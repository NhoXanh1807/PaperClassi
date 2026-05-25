#!/usr/bin/env python3
"""
Paper Classification — Fine-tuned SPECTER2 + Ordinal Threshold Optimization
5-fold CV → OOF probs → optimize thresholds → ensemble fold models on test
"""

import os, ssl
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"]     = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
import warnings
warnings.filterwarnings("ignore")

HF_TOKEN   = os.environ.get("HF_ACCESS_TOKEN", "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")
DATA_DIR   = Path("/Users/quangnguyen/Desktop/paper-classi")
MODEL_NAME = "allenai/specter2_base"
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"

# ── Hyperparameters ────────────────────────────────────────────────────────────
EPOCHS      = 10
BATCH_SIZE  = 16
LR          = 2e-5
MAX_LEN     = 256
N_FOLDS     = 5
SEED        = 42
NUM_LABELS  = 5
FREEZE_BELOW = 8   # freeze transformer layers 0-7, fine-tune 8-11 + head

print(f"Device: {DEVICE} | Model: {MODEL_NAME}")

# ── Load data ──────────────────────────────────────────────────────────────────
train = pd.read_csv(DATA_DIR / "train.csv")
pub   = pd.read_csv(DATA_DIR / "public_test.csv")
priv  = pd.read_csv(DATA_DIR / "private_test.csv")

print(f"Train: {len(train)} | Public: {len(pub)} | Private: {len(priv)}")

def make_text(df):
    # title + venue gives SPECTER2 the most useful context
    return (df["title"].fillna("") + " " + df["venue"].fillna("")).tolist()

train_texts = make_text(train)
test_texts  = make_text(pub) + make_text(priv)
y_train     = train["Label"].values - 1  # 0-indexed for CE

# Class weights to handle label imbalance (label 1 >> label 5)
counts       = np.bincount(y_train)
class_weights = torch.tensor(
    (1.0 / counts) / (1.0 / counts).sum() * NUM_LABELS, dtype=torch.float
)

# ── Dataset ────────────────────────────────────────────────────────────────────
class PaperDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts  = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {"text": self.texts[idx]}
        if self.labels is not None:
            item["label"] = int(self.labels[idx])
        return item

def make_collate(tokenizer):
    def collate(batch):
        enc = tokenizer(
            [b["text"] for b in batch],
            truncation=True, padding=True,
            max_length=MAX_LEN, return_tensors="pt",
        )
        if "label" in batch[0]:
            enc["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        return enc
    return collate

# ── Train / eval helpers ───────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer=None, scheduler=None, criterion=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss = 0.0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = batch.pop("labels", None)
            out    = model(**batch)

            if criterion is not None and labels is not None:
                loss = criterion(out.logits, labels)
            elif labels is not None:
                loss = out.loss
            else:
                loss = None

            if training and loss is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()

    return total_loss / max(len(loader), 1)

@torch.no_grad()
def get_probs(model, loader):
    model.eval()
    probs = []
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}
        logits = model(**batch).logits
        probs.append(torch.softmax(logits, dim=-1).cpu().float().numpy())
    return np.vstack(probs)

# ── QWK / threshold helpers ───────────────────────────────────────────────────
LABEL_VALS = np.array([1, 2, 3, 4, 5], dtype=float)

def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")

def probs_to_label(probs, thresholds):
    expected = probs @ LABEL_VALS
    out = np.ones(len(expected), dtype=int)
    for i, t in enumerate(thresholds):
        out[expected > t] = i + 2
    return out

def optimize_thresholds(probs, y_true):
    def neg_qwk(thr):
        return -qwk(y_true, probs_to_label(probs, np.sort(thr)))
    res = minimize(neg_qwk, [1.5, 2.5, 3.5, 4.5], method="Nelder-Mead",
                   options={"maxiter": 3000, "xatol": 1e-5})
    return np.sort(res.x)

# ── 5-fold cross-validation ───────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
skf       = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

oof_probs  = np.zeros((len(train), NUM_LABELS))
test_probs = np.zeros((len(test_texts), NUM_LABELS))

for fold, (tr_idx, val_idx) in enumerate(skf.split(train_texts, y_train), 1):
    print(f"\n── Fold {fold}/{N_FOLDS} ──────────────────────────")

    tr_ds   = PaperDataset([train_texts[i] for i in tr_idx],  y_train[tr_idx])
    val_ds  = PaperDataset([train_texts[i] for i in val_idx], y_train[val_idx])
    test_ds = PaperDataset(test_texts)
    coll    = make_collate(tokenizer)

    tr_loader   = DataLoader(tr_ds,   batch_size=BATCH_SIZE, shuffle=True,  collate_fn=coll)
    val_loader  = DataLoader(val_ds,  batch_size=32,         shuffle=False, collate_fn=coll)
    test_loader = DataLoader(test_ds, batch_size=32,         shuffle=False, collate_fn=coll)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS, token=HF_TOKEN,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)

    # Freeze lower transformer layers
    for name, param in model.named_parameters():
        for part in name.split("."):
            if part.isdigit() and int(part) < FREEZE_BELOW:
                param.requires_grad = False
                break

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01,
    )
    total_steps = len(tr_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    best_qwk, best_state = 0.0, None
    patience, wait       = 3, 0

    for epoch in range(1, EPOCHS + 1):
        loss      = run_epoch(model, tr_loader, optimizer, scheduler, criterion)
        val_probs = get_probs(model, val_loader)
        val_preds = val_probs.argmax(1) + 1
        val_qwk   = qwk(y_train[val_idx] + 1, val_preds)
        marker    = " ★" if val_qwk > best_qwk else ""
        print(f"  Epoch {epoch:2d}  loss={loss:.4f}  val_QWK={val_qwk:.4f}{marker}")

        if val_qwk > best_qwk:
            best_qwk  = val_qwk
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stop (patience={patience})")
                break

    model.load_state_dict(best_state)
    oof_probs[val_idx]  = get_probs(model, val_loader)
    test_probs         += get_probs(model, test_loader) / N_FOLDS

    del model, optimizer, scheduler, best_state
    if DEVICE == "mps":
        torch.mps.empty_cache()

# ── OOF evaluation & threshold optimization ────────────────────────────────────
oof_base = oof_probs.argmax(1) + 1
qwk_base = qwk(y_train + 1, oof_base)
print(f"\nOOF baseline QWK : {qwk_base:.4f}")

opt_thr   = optimize_thresholds(oof_probs, y_train + 1)
oof_opt   = probs_to_label(oof_probs, opt_thr)
qwk_opt   = qwk(y_train + 1, oof_opt)
print(f"OOF threshold QWK: {qwk_opt:.4f}")
print(f"Thresholds       : {np.round(opt_thr, 3)}")

# ── Generate predictions ──────────────────────────────────────────────────────
use_thr     = qwk_opt > qwk_base
final_preds = probs_to_label(test_probs, opt_thr) if use_thr else test_probs.argmax(1) + 1
final_qwk   = qwk_opt if use_thr else qwk_base

pub_preds  = final_preds[:len(pub)]
priv_preds = final_preds[len(pub):]

pub_sub  = pd.DataFrame({"id": pub["id"],  "Label": pub_preds})
priv_sub = pd.DataFrame({"id": priv["id"], "Label": priv_preds})
combined = pd.concat([pub_sub, priv_sub], ignore_index=True)

combined.to_csv(DATA_DIR / "submission_finetuned.csv",         index=False)
pub_sub.to_csv(DATA_DIR  / "public_submission_finetuned.csv",  index=False)
priv_sub.to_csv(DATA_DIR / "private_submission_finetuned.csv", index=False)

np.savez(DATA_DIR / "specter2_probs.npz", oof=oof_probs, test=test_probs)

print(f"\n{'='*45}")
print(f"Final OOF QWK : {final_qwk:.4f}")
print(f"Submission    : submission_finetuned.csv ({len(combined)} rows)")
print("Pred dist:")
print(combined["Label"].value_counts().sort_index().to_string())
