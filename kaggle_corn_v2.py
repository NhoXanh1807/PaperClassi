#!/usr/bin/env python3
"""
Training pipeline v2 — CORN ordinal loss + multi-model + multi-seed.

Designed to run on a rented Vast.ai GPU VM (or any CUDA box). For each model:
  - K-fold StratifiedKFold (default 5 folds)
  - For each fold: train SEEDS_PER_MODEL seeds, average probs
  - Layer-wise learning rate decay (LLRD)
  - CORN ordinal loss (Shi et al. 2023) — directly optimizes for ordinal QWK
  - Early stopping on val QWK
  - Saves OOF probs + test probs as NPZ

Output files (saved to --out-dir, default ./outputs):
  - {model_slug}_probs.npz   {oof: (N_train, 5), test: (N_test, 5)}
  - meta.json                {models, fold_qwks, oof_qwks}

Usage (basic):
    python kaggle_corn_v2.py --data-dir ./data --out-dir ./outputs

Common knobs:
    --models scibert specter2          # subset of preset models
    --folds 5 --seeds 3                 # k-fold and seeds-per-fold
    --epochs-small 10 --epochs-large 6  # epochs for small / large backbones
    --batch-small 16 --batch-large 8
    --max-len 256
    --resume                            # skip models whose NPZ already exists

Data layout expected:
    <data-dir>/train.csv
    <data-dir>/public_test.csv
    <data-dir>/private_test.csv
"""

import os, ssl, json, gc, random, argparse, warnings
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
PRESET_MODELS = {
    # key: (hf_id, size_class)   size_class ∈ {"small", "large"}
    "scibert":    ("allenai/scibert_scivocab_uncased", "small"),
    "specter2":   ("allenai/specter2_base",            "small"),
    "deberta-v3": ("microsoft/deberta-v3-large",       "large"),
    "e5-large":   ("intfloat/e5-large-v2",             "large"),
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "."),
                   help="Folder containing train.csv, public_test.csv, private_test.csv (default: cwd)")
    p.add_argument("--out-dir",  default=os.environ.get("OUT_DIR",  "./outputs"),
                   help="Folder for NPZ probs + meta.json (default: ./outputs)")
    p.add_argument("--models", nargs="+", default=list(PRESET_MODELS.keys()),
                   choices=list(PRESET_MODELS.keys()),
                   help="Which preset models to train")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seeds", type=int, default=3, help="Seeds averaged per fold")
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--epochs-small", type=int, default=10)
    p.add_argument("--epochs-large", type=int, default=6)
    p.add_argument("--batch-small", type=int, default=16)
    p.add_argument("--batch-large", type=int, default=8)
    p.add_argument("--lr-head-small", type=float, default=3e-4)
    p.add_argument("--lr-backbone-small", type=float, default=2e-5)
    p.add_argument("--lr-head-large", type=float, default=2e-4)
    p.add_argument("--lr-backbone-large", type=float, default=1e-5)
    p.add_argument("--llrd-decay", type=float, default=0.9, help="1.0 disables LLRD")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers. 0 = no multiprocessing (safest on Py 3.14 forkserver). "
                        "For this dataset size workers add little speedup.")
    p.add_argument("--resume", action="store_true",
                   help="Skip a model if its NPZ already exists in out-dir")
    p.add_argument("--hf-token", default=os.environ.get("HF_ACCESS_TOKEN") or os.environ.get("HF_TOKEN"))
    return p.parse_args()

args = parse_args()

DATA_DIR = Path(args.data_dir).expanduser().resolve()
OUT_DIR  = Path(args.out_dir).expanduser().resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

for f in ("train.csv", "public_test.csv", "private_test.csv"):
    if not (DATA_DIR / f).exists():
        raise FileNotFoundError(f"Missing {DATA_DIR / f}. Pass --data-dir or set DATA_DIR.")

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
USE_AMP = DEVICE == "cuda"
print(f"Device: {DEVICE}  | AMP: {USE_AMP}  | Data: {DATA_DIR}  | Out: {OUT_DIR}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}  | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────────────────────────────────────
N_FOLDS         = args.folds
SEEDS_PER_MODEL = args.seeds
NUM_LABELS      = 5
MAX_LEN         = args.max_len
PATIENCE        = args.patience
WARMUP_RATIO    = args.warmup_ratio
WEIGHT_DECAY    = args.weight_decay
LLRD_DECAY      = args.llrd_decay
GRAD_CLIP       = args.grad_clip

# Build per-model config from CLI: (hf_id, batch_size, lr_head, lr_backbone, epochs)
def _cfg(key):
    hf_id, klass = PRESET_MODELS[key]
    if klass == "small":
        return (hf_id, args.batch_small, args.lr_head_small, args.lr_backbone_small, args.epochs_small)
    return (hf_id, args.batch_large, args.lr_head_large, args.lr_backbone_large, args.epochs_large)

MODELS = [_cfg(k) for k in args.models]
HF_TOKEN = args.hf_token

# ──────────────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────────────
train = pd.read_csv(DATA_DIR / "train.csv")
pub   = pd.read_csv(DATA_DIR / "public_test.csv")
priv  = pd.read_csv(DATA_DIR / "private_test.csv")
print(f"Train: {len(train)} | Public: {len(pub)} | Private: {len(priv)}")

def make_text(df, model_name):
    """Add year + venue + title; prefix 'query:' for E5 models per their docs."""
    title  = df["title"].fillna("").astype(str)
    venue  = df["venue"].fillna("").astype(str)
    year   = df["year"].fillna(0).astype(int).astype(str)
    text   = "venue: " + venue + " | year: " + year + " | title: " + title
    if "e5" in model_name.lower():
        text = "query: " + text
    return text.tolist()

y_train_idx = train["Label"].values - 1   # 0-indexed for CORN

# ──────────────────────────────────────────────────────────────────────────────
# CORN ordinal loss (Shi et al., 2023) — cumulative link, conditional training
# ──────────────────────────────────────────────────────────────────────────────
def corn_loss(logits, y, num_classes=NUM_LABELS):
    """
    logits: (B, K-1) raw outputs.
    y     : (B,) targets in [0, K-1].
    Each task i predicts P(y > i | y > i-1). Train task i only on examples
    with y > i-1 (conditional). BCE-with-logits per task, averaged.
    """
    K = num_classes
    losses, n_active = [], 0
    for i in range(K - 1):
        mask = (y >= i) if i > 0 else torch.ones_like(y, dtype=torch.bool)
        if mask.sum() == 0:
            continue
        # Note: y >= i means y > i-1, so mask out samples that already failed previous task
        target = (y[mask] > i).float()
        logit  = logits[mask, i]
        losses.append(F.binary_cross_entropy_with_logits(logit, target, reduction="sum"))
        n_active += mask.sum().item()
    return torch.stack(losses).sum() / max(n_active, 1)


def corn_logits_to_probs(logits):
    """
    Convert (B, K-1) CORN logits to (B, K) class probability vector.
    Joint P(y > k) = prod_{i=0..k} sigmoid(logit_i) by conditional cascade.
    """
    sig = torch.sigmoid(logits)               # (B, K-1) conditional probs
    cum = torch.cumprod(sig, dim=1)           # (B, K-1) joint P(y > k)
    K   = logits.shape[1] + 1
    probs = torch.zeros(logits.shape[0], K, device=logits.device, dtype=logits.dtype)
    probs[:, 0]   = 1.0 - cum[:, 0]
    for k in range(1, K - 1):
        probs[:, k] = cum[:, k - 1] - cum[:, k]
    probs[:, K-1] = cum[:, K - 2]
    probs = probs.clamp(min=1e-7)
    probs = probs / probs.sum(dim=1, keepdim=True)
    return probs


# ──────────────────────────────────────────────────────────────────────────────
# QWK helpers
# ──────────────────────────────────────────────────────────────────────────────
LABEL_VALS = np.array([1, 2, 3, 4, 5], dtype=float)

def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")

def probs_to_label_thr(probs, thresholds):
    expected = probs @ LABEL_VALS
    out = np.ones(len(expected), dtype=int)
    for i, t in enumerate(thresholds):
        out[expected > t] = i + 2
    return out

def optimize_thresholds(probs, y_true_1to5):
    def neg(thr):
        return -qwk(y_true_1to5, probs_to_label_thr(probs, np.sort(thr)))
    res = minimize(neg, [1.5, 2.5, 3.5, 4.5], method="Nelder-Mead",
                   options={"maxiter": 3000, "xatol": 1e-5})
    return np.sort(res.x)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset / collate
# ──────────────────────────────────────────────────────────────────────────────
class PaperDS(Dataset):
    def __init__(self, texts, labels=None):
        self.texts, self.labels = texts, labels
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        item = {"text": self.texts[i]}
        if self.labels is not None: item["label"] = int(self.labels[i])
        return item

class Collate:
    """Picklable collate (Python 3.14 forkserver needs this — closures can't pickle)."""
    def __init__(self, tok, max_len):
        self.tok = tok
        self.max_len = max_len
    def __call__(self, batch):
        enc = self.tok([b["text"] for b in batch], truncation=True, padding=True,
                       max_length=self.max_len, return_tensors="pt")
        if "label" in batch[0]:
            enc["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        return enc


# ──────────────────────────────────────────────────────────────────────────────
# Backbone + CORN head
# ──────────────────────────────────────────────────────────────────────────────
class CORNModel(nn.Module):
    def __init__(self, hf_id, num_classes=NUM_LABELS, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(hf_id, token=HF_TOKEN)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        # K-1 binary heads (one per cumulative threshold)
        self.head = nn.Linear(hidden, num_classes - 1)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, **kwargs):
        out = self.backbone(**{k: v for k, v in kwargs.items() if k != "labels"})
        # Use CLS pooling (works for BERT family + DeBERTa + E5)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            pooled = out.last_hidden_state[:, 0]
        return self.head(self.dropout(pooled))


# ──────────────────────────────────────────────────────────────────────────────
# Layer-wise LR decay
# ──────────────────────────────────────────────────────────────────────────────
def make_llrd_params(model, lr_head, lr_backbone, decay=LLRD_DECAY, weight_decay=WEIGHT_DECAY):
    """
    Head gets lr_head; backbone layers get lr_backbone * decay^(n_layers - layer_idx),
    so deeper layers (closer to head) train fastest.
    """
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    params = []

    # Head
    head_params = [(n, p) for n, p in model.head.named_parameters() if p.requires_grad]
    params.append({
        "params": [p for n, p in head_params if not any(nd in n for nd in no_decay)],
        "lr": lr_head, "weight_decay": weight_decay,
    })
    params.append({
        "params": [p for n, p in head_params if any(nd in n for nd in no_decay)],
        "lr": lr_head, "weight_decay": 0.0,
    })

    # Find encoder layers (works for BERT/DeBERTa/E5)
    layers = None
    for attr in ["encoder.layer", "encoder.layers", "transformer.layer"]:
        obj = model.backbone
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            layers = obj
            break
        except AttributeError:
            continue
    n_layers = len(layers) if layers is not None else 0

    # Embeddings + non-layer params (lowest lr)
    backbone_named = list(model.backbone.named_parameters())
    layer_param_ids = set()
    if n_layers > 0:
        for i, layer in enumerate(layers):
            lr_i = lr_backbone * (decay ** (n_layers - 1 - i))
            layer_params = [(n, p) for n, p in layer.named_parameters() if p.requires_grad]
            params.append({
                "params": [p for n, p in layer_params if not any(nd in n for nd in no_decay)],
                "lr": lr_i, "weight_decay": weight_decay,
            })
            params.append({
                "params": [p for n, p in layer_params if any(nd in n for nd in no_decay)],
                "lr": lr_i, "weight_decay": 0.0,
            })
            for _, p in layer_params:
                layer_param_ids.add(id(p))

    other = [(n, p) for n, p in backbone_named if p.requires_grad and id(p) not in layer_param_ids]
    if other:
        lr_low = lr_backbone * (decay ** n_layers)
        params.append({
            "params": [p for n, p in other if not any(nd in n for nd in no_decay)],
            "lr": lr_low, "weight_decay": weight_decay,
        })
        params.append({
            "params": [p for n, p in other if any(nd in n for nd in no_decay)],
            "lr": lr_low, "weight_decay": 0.0,
        })

    # Filter empty groups
    return [g for g in params if len(g["params"]) > 0]


# ──────────────────────────────────────────────────────────────────────────────
# Train / eval one fold-seed
# ──────────────────────────────────────────────────────────────────────────────
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def train_one(model_cfg, tr_texts, tr_y, val_texts, val_y, test_texts, seed):
    hf_id, batch, lr_head, lr_bb, epochs = model_cfg
    set_seed(seed)

    tok = AutoTokenizer.from_pretrained(hf_id, token=HF_TOKEN, use_fast=True)
    model = CORNModel(hf_id).to(DEVICE)

    coll = Collate(tok, MAX_LEN)
    nw = args.num_workers
    tr_loader   = DataLoader(PaperDS(tr_texts, tr_y),   batch_size=batch, shuffle=True,  collate_fn=coll, num_workers=nw)
    val_loader  = DataLoader(PaperDS(val_texts, val_y), batch_size=32,    shuffle=False, collate_fn=coll, num_workers=nw)
    test_loader = DataLoader(PaperDS(test_texts),       batch_size=32,    shuffle=False, collate_fn=coll, num_workers=nw)

    param_groups = make_llrd_params(model, lr_head, lr_bb)
    optim = AdamW(param_groups)
    total_steps = len(tr_loader) * epochs
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=int(total_steps * WARMUP_RATIO), num_training_steps=total_steps,
    )
    scaler = GradScaler(enabled=USE_AMP)

    best_qwk, best_val_probs, best_test_probs = -1.0, None, None
    wait = 0

    for ep in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        for batch_data in tr_loader:
            batch_data = {k: v.to(DEVICE, non_blocking=True) for k, v in batch_data.items()}
            labels = batch_data.pop("labels")
            optim.zero_grad(set_to_none=True)
            with autocast(enabled=USE_AMP):
                logits = model(**batch_data)
                loss   = corn_loss(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optim)
            scaler.update()
            sched.step()
            run_loss += loss.item()

        # Eval
        model.eval()
        val_probs, test_probs = [], []
        with torch.no_grad():
            for b in val_loader:
                b = {k: v.to(DEVICE, non_blocking=True) for k, v in b.items() if k != "labels"}
                with autocast(enabled=USE_AMP):
                    p = corn_logits_to_probs(model(**b))
                val_probs.append(p.cpu().float().numpy())
            val_probs = np.vstack(val_probs)

            val_thr   = optimize_thresholds(val_probs, val_y + 1)
            val_preds = probs_to_label_thr(val_probs, val_thr)
            val_qwk   = qwk(val_y + 1, val_preds)

        marker = ""
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            wait = 0
            marker = " ★"
            # Cache test probs only when improved (saves compute)
            with torch.no_grad():
                tp = []
                for b in test_loader:
                    b = {k: v.to(DEVICE, non_blocking=True) for k, v in b.items()}
                    with autocast(enabled=USE_AMP):
                        p = corn_logits_to_probs(model(**b))
                    tp.append(p.cpu().float().numpy())
                best_test_probs = np.vstack(tp)
            best_val_probs = val_probs
        else:
            wait += 1
        print(f"     ep{ep:2d} loss={run_loss/len(tr_loader):.4f}  val_QWK={val_qwk:.4f}{marker}")
        if wait >= PATIENCE:
            print(f"     early-stop @ ep{ep}")
            break

    del model, optim, sched, scaler, tok
    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    elif DEVICE == "mps": torch.mps.empty_cache()
    return best_val_probs, best_test_probs, best_qwk


# ──────────────────────────────────────────────────────────────────────────────
# Main loop  (guarded for Python 3.14 forkserver — workers re-import this module,
# and would otherwise re-enter training and dead-loop)
# ──────────────────────────────────────────────────────────────────────────────
def main():
  meta = {"models": [], "global": {}}

  for cfg in MODELS:
    hf_id = cfg[0]
    slug  = hf_id.replace("/", "_")
    npz_path = OUT_DIR / f"{slug}_probs.npz"
    if npz_path.exists():
        print(f"\n=== SKIP {hf_id} (already saved at {npz_path}) ===")
        continue

    print(f"\n=== Model: {hf_id} ===")
    texts_train = make_text(train, hf_id)
    texts_test  = make_text(pub, hf_id) + make_text(priv, hf_id)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof_probs  = np.zeros((len(train), NUM_LABELS))
    test_probs = np.zeros((len(texts_test), NUM_LABELS))
    fold_qwks  = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(texts_train, y_train_idx), 1):
        print(f"\n  ── Fold {fold}/{N_FOLDS} ──")
        tr_t  = [texts_train[i] for i in tr_idx]
        val_t = [texts_train[i] for i in val_idx]
        tr_y, val_y = y_train_idx[tr_idx], y_train_idx[val_idx]

        seed_val, seed_test, seed_qwks = None, None, []
        for s_idx in range(SEEDS_PER_MODEL):
            seed = 42 + 1000 * s_idx
            print(f"   seed {seed}")
            vp, tp, q = train_one(cfg, tr_t, tr_y, val_t, val_y, texts_test, seed)
            seed_val  = vp if seed_val is None else seed_val + vp
            seed_test = tp if seed_test is None else seed_test + tp
            seed_qwks.append(q)
        seed_val  /= SEEDS_PER_MODEL
        seed_test /= SEEDS_PER_MODEL

        # Renormalize after averaging (probs averaged independently can drift slightly)
        seed_val  = seed_val  / seed_val.sum(axis=1, keepdims=True)
        seed_test = seed_test / seed_test.sum(axis=1, keepdims=True)

        oof_probs[val_idx]  = seed_val
        test_probs         += seed_test / N_FOLDS

        thr_f  = optimize_thresholds(seed_val, val_y + 1)
        qwk_f  = qwk(val_y + 1, probs_to_label_thr(seed_val, thr_f))
        fold_qwks.append(qwk_f)
        print(f"   fold QWK (seed-avg): {qwk_f:.4f}  (per-seed mean: {np.mean(seed_qwks):.4f})")

    # OOF metrics
    thr_global = optimize_thresholds(oof_probs, y_train_idx + 1)
    oof_qwk    = qwk(y_train_idx + 1, probs_to_label_thr(oof_probs, thr_global))
    print(f"\n  >> {hf_id}  OOF QWK = {oof_qwk:.4f}  (fold mean {np.mean(fold_qwks):.4f})")

    np.savez(npz_path, oof=oof_probs, test=test_probs, thresholds=thr_global)
    meta["models"].append({
        "id": hf_id,
        "oof_qwk": float(oof_qwk),
        "fold_qwks": [float(x) for x in fold_qwks],
        "thresholds": [float(x) for x in thr_global],
    })

  # ────────────────────────────────────────────────────────────────────────────
  # Simple weighted ensemble baseline (so we have a submission even without stacking)
  # ────────────────────────────────────────────────────────────────────────────
  all_npz = [(m["id"], np.load(OUT_DIR / f"{m['id'].replace('/', '_')}_probs.npz"))
             for m in meta["models"]]
  if all_npz:
      print("\n=== Quick equal-weight ensemble (baseline) ===")
      oof_mix  = np.mean([n["oof"]  for _, n in all_npz], axis=0)
      test_mix = np.mean([n["test"] for _, n in all_npz], axis=0)
      thr = optimize_thresholds(oof_mix, y_train_idx + 1)
      q   = qwk(y_train_idx + 1, probs_to_label_thr(oof_mix, thr))
      print(f"  OOF QWK: {q:.4f}  thr={np.round(thr,3)}")
      meta["global"]["equal_weight_oof_qwk"] = float(q)

      final = probs_to_label_thr(test_mix, thr)
      pub_sub  = pd.DataFrame({"id": pub["id"],  "Label": final[:len(pub)]})
      priv_sub = pd.DataFrame({"id": priv["id"], "Label": final[len(pub):]})
      pd.concat([pub_sub, priv_sub], ignore_index=True).to_csv(OUT_DIR / "submission_corn_avg.csv", index=False)
      pub_sub.to_csv(OUT_DIR / "public_submission_corn_avg.csv", index=False)
      priv_sub.to_csv(OUT_DIR / "private_submission_corn_avg.csv", index=False)
      print(f"  Wrote submission_corn_avg.csv")

  with open(OUT_DIR / "meta.json", "w") as f:
      json.dump(meta, f, indent=2)
  print(f"\nDone. NPZ files + meta.json saved to {OUT_DIR}")
  print("Download all *_probs.npz and meta.json, then run solve_stacking_v2.py locally.")


if __name__ == "__main__":
    main()
