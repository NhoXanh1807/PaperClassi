#!/usr/bin/env python3
"""
Paper Classification — SPECTER2 embeddings + LinearSVC
allenai/specter2 encodes academic paper titles into 768-dim vectors
optimized for scientific document understanding.
"""

import os, ssl
# Bypass corporate proxy self-signed cert
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"]    = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import torch
import warnings
warnings.filterwarnings("ignore")

HF_TOKEN = os.environ.get("HF_ACCESS_TOKEN", "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")
DATA_DIR  = Path("/Users/quangnguyen/Desktop/paper-classi")
EMB_CACHE = DATA_DIR / "specter2_embeddings.npz"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ─── Load data ────────────────────────────────────────────────────────────────
train = pd.read_csv(DATA_DIR / "train.csv")
pub   = pd.read_csv(DATA_DIR / "public_test.csv")
priv  = pd.read_csv(DATA_DIR / "private_test.csv")

print(f"Train: {len(train)} | Public: {len(pub)} | Private: {len(priv)}")

# ─── Build input text for SPECTER2 ────────────────────────────────────────────
# SPECTER2 format: title [SEP] abstract — no abstract, so just use title
# Optionally append venue for domain context
def specter_text(df):
    return (df["title"].fillna("") + " " + df["venue"].fillna("")).tolist()

all_texts = specter_text(train) + specter_text(pub) + specter_text(priv)

# ─── Encode with SPECTER2 (cached) ────────────────────────────────────────────
if EMB_CACHE.exists():
    print(f"Loading cached embeddings from {EMB_CACHE}")
    cache     = np.load(EMB_CACHE)
    all_embs  = cache["embeddings"]
else:
    print("Encoding with allenai/specter2_base ...")
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base", token=HF_TOKEN)
    hf_model  = AutoModel.from_pretrained("allenai/specter2_base", token=HF_TOKEN)
    hf_model  = hf_model.to(DEVICE).eval()

    def encode_batch(texts, batch_size=32):
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc   = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            with torch.no_grad():
                out = hf_model(**enc)
            # CLS token embedding
            vecs = out.last_hidden_state[:, 0, :].cpu().float().numpy()
            all_vecs.append(vecs)
            if (i // batch_size) % 10 == 0:
                print(f"  {i}/{len(texts)}", end="\r")
        print()
        return np.vstack(all_vecs)

    all_embs = encode_batch(all_texts)
    # L2 normalize
    norms    = np.linalg.norm(all_embs, axis=1, keepdims=True)
    all_embs = all_embs / np.clip(norms, 1e-8, None)
    np.savez(EMB_CACHE, embeddings=all_embs)
    print(f"  Cached to {EMB_CACHE}")

n_train = len(train)
n_pub   = len(pub)
emb_train = all_embs[:n_train]
emb_pub   = all_embs[n_train:n_train + n_pub]
emb_priv  = all_embs[n_train + n_pub:]

print(f"Embedding shape: {emb_train.shape}")

# ─── Add structured features ──────────────────────────────────────────────────
VENUES    = ["cav", "iclp", "kr", "lics", "lpnmr"]
YEAR_MEAN = train["year"].mean()
YEAR_STD  = train["year"].std()

def make_struct(df):
    venue_dummies = pd.get_dummies(df["venue"], prefix="v")
    for v in VENUES:
        col = f"v_{v}"
        if col not in venue_dummies:
            venue_dummies[col] = 0
    venue_mat = venue_dummies[[f"v_{v}" for v in VENUES]].values.astype(float)
    year_sc   = ((df["year"] - YEAR_MEAN) / YEAR_STD).values.reshape(-1, 1)
    return np.hstack([venue_mat, year_sc])

struct_train = make_struct(train)
struct_pub   = make_struct(pub)
struct_priv  = make_struct(priv)

# Scale struct features to same magnitude as normalized embeddings
scaler = StandardScaler()
struct_train = scaler.fit_transform(struct_train)
struct_pub   = scaler.transform(struct_pub)
struct_priv  = scaler.transform(struct_priv)

# Weight venue features more heavily (they are highly predictive)
STRUCT_WEIGHT = 3.0
X_train = np.hstack([emb_train, struct_train * STRUCT_WEIGHT])
X_pub   = np.hstack([emb_pub,   struct_pub   * STRUCT_WEIGHT])
X_priv  = np.hstack([emb_priv,  struct_priv  * STRUCT_WEIGHT])
y_train = train["Label"].values

print(f"Final feature dim: {X_train.shape[1]}")

# ─── Tune C ───────────────────────────────────────────────────────────────────
def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nTuning C:")
best_c, best_score = 0.1, 0.0
for C in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]:
    preds = cross_val_predict(
        LinearSVC(C=C, max_iter=3000, dual=True), X_train, y_train, cv=skf
    )
    score = qwk(y_train, preds)
    marker = " ←" if score > best_score else ""
    print(f"  C={C:<5}  QWK={score:.4f}{marker}")
    if score > best_score:
        best_score = score
        best_c = C

print(f"\nBest: C={best_c}  CV QWK={best_score:.4f}")

# ─── Ordinal threshold optimization ───────────────────────────────────────────
print("\nOptimizing thresholds on OOF probabilities...")

calibrated = CalibratedClassifierCV(
    LinearSVC(C=best_c, max_iter=3000, dual=True), cv=skf, method="isotonic"
)
oof_probs = cross_val_predict(calibrated, X_train, y_train, cv=skf, method="predict_proba")

LABELS = np.array([1, 2, 3, 4, 5])

def probs_to_label(probs, thresholds):
    expected = probs @ LABELS
    preds = np.ones(len(expected), dtype=int)
    for i, t in enumerate(thresholds):
        preds[expected > t] = i + 2
    return preds

def neg_qwk(thresholds, probs, y_true):
    preds = probs_to_label(probs, np.sort(thresholds))
    return -qwk(y_true, preds)

result = minimize(
    neg_qwk,
    np.array([1.5, 2.5, 3.5, 4.5]),
    args=(oof_probs, y_train),
    method="Nelder-Mead",
    options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-4},
)
opt_thresholds = np.sort(result.x)
qwk_opt = qwk(y_train, probs_to_label(oof_probs, opt_thresholds))

oof_base  = cross_val_predict(
    LinearSVC(C=best_c, max_iter=3000, dual=True), X_train, y_train, cv=skf
)
qwk_base = qwk(y_train, oof_base)

print(f"  Baseline CV QWK : {qwk_base:.4f}")
print(f"  Threshold CV QWK: {qwk_opt:.4f}")
print(f"  Thresholds      : {np.round(opt_thresholds, 3)}")

# ─── Train final model & predict ──────────────────────────────────────────────
use_threshold = qwk_opt > qwk_base
final_qwk     = qwk_opt if use_threshold else qwk_base

if use_threshold:
    final_model = CalibratedClassifierCV(
        LinearSVC(C=best_c, max_iter=3000, dual=True), method="isotonic"
    )
    final_model.fit(X_train, y_train)
    def predict(X):
        return probs_to_label(final_model.predict_proba(X), opt_thresholds)
else:
    final_model = LinearSVC(C=best_c, max_iter=3000, dual=True)
    final_model.fit(X_train, y_train)
    predict = final_model.predict

pub_preds  = predict(X_pub)
priv_preds = predict(X_priv)

# ─── Submission ───────────────────────────────────────────────────────────────
pub_sub  = pd.DataFrame({"id": pub["id"],  "Label": pub_preds})
priv_sub = pd.DataFrame({"id": priv["id"], "Label": priv_preds})
combined = pd.concat([pub_sub, priv_sub], ignore_index=True)

combined.to_csv(DATA_DIR / "submission_specter2.csv",        index=False)
pub_sub.to_csv(DATA_DIR  / "public_submission_specter2.csv", index=False)
priv_sub.to_csv(DATA_DIR / "private_submission_specter2.csv",index=False)

print(f"\n{'='*40}")
print(f"Final CV QWK  : {final_qwk:.4f}")
print(f"Submission    : submission_specter2.csv ({len(combined)} rows)")
print("Pred dist:")
print(combined["Label"].value_counts().sort_index().to_string())
