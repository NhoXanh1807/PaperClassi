#!/usr/bin/env python3
"""
Stacking ensemble v2 — combines base-model OOF/test probs with meta features
through a LightGBM ordinal regressor, then optimizes QWK thresholds.

Expects in DATA_DIR (downloaded from Kaggle output of kaggle_corn_v2.py):
  - {model_slug}_probs.npz  with arrays  oof (N, 5), test (M, 5)
  - meta.json               (optional, for per-model OOF QWK printout)

Existing legacy files (specter2_probs.npz, scibert_probs.npz from solve_finetune*.py)
are also picked up automatically and stacked together.

Outputs:
  - submission_stacking_v2.csv         (combined pub + priv)
  - public_submission_stacking_v2.csv
  - private_submission_stacking_v2.csv
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize

import lightgbm as lgb

DATA_DIR   = Path("/Users/quangnguyen/Desktop/paper-classi")
NUM_LABELS = 5
N_FOLDS    = 5
SEED       = 42
LABEL_VALS = np.array([1, 2, 3, 4, 5], dtype=float)


# ── QWK helpers ──────────────────────────────────────────────────────────────
def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")

def expected_value(probs):
    return probs @ LABEL_VALS

def thr_to_label(expected, thresholds):
    out = np.ones(len(expected), dtype=int)
    for i, t in enumerate(thresholds):
        out[expected > t] = i + 2
    return out

def probs_to_label_thr(probs, thresholds):
    return thr_to_label(expected_value(probs), thresholds)

def optimize_thresholds(values, y_true_1to5, init=(1.5, 2.5, 3.5, 4.5)):
    """`values` can be expected value (1D) or full probs (2D)."""
    if values.ndim == 2:
        expected = expected_value(values)
    else:
        expected = values
    def neg(thr):
        return -qwk(y_true_1to5, thr_to_label(expected, np.sort(thr)))
    res = minimize(neg, init, method="Nelder-Mead",
                   options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6})
    return np.sort(res.x)


# ── Load base-model probs ────────────────────────────────────────────────────
train = pd.read_csv(DATA_DIR / "train.csv")
pub   = pd.read_csv(DATA_DIR / "public_test.csv")
priv  = pd.read_csv(DATA_DIR / "private_test.csv")
y     = train["Label"].values                       # 1..5

npz_files = sorted(DATA_DIR.glob("*_probs.npz"))
if not npz_files:
    raise SystemExit(f"No *_probs.npz files found in {DATA_DIR}. "
                     "Run kaggle_corn_v2.py on Kaggle first and download the NPZs.")

print(f"Found {len(npz_files)} base-model prob files:")
oof_stack, test_stack, names = [], [], []
for f in npz_files:
    d = np.load(f)
    if d["oof"].shape != (len(train), NUM_LABELS):
        print(f"  SKIP {f.name}: shape mismatch {d['oof'].shape}")
        continue
    expected_test_len = len(pub) + len(priv)
    if d["test"].shape[0] != expected_test_len:
        print(f"  SKIP {f.name}: test rows {d['test'].shape[0]} != {expected_test_len}")
        continue

    # Per-model OOF QWK with its own threshold (sanity check)
    thr_m = optimize_thresholds(d["oof"], y)
    q_m   = qwk(y, probs_to_label_thr(d["oof"], thr_m))
    print(f"  {f.name:50s}  OOF QWK={q_m:.4f}")

    oof_stack.append(d["oof"])
    test_stack.append(d["test"])
    names.append(f.stem)

# Concatenate across models: (N, 5*M) probs + (N, M) expected-value summaries
oof_probs  = np.concatenate(oof_stack,  axis=1)
test_probs = np.concatenate(test_stack, axis=1)
oof_expv   = np.stack([expected_value(p) for p in oof_stack],  axis=1)
test_expv  = np.stack([expected_value(p) for p in test_stack], axis=1)


# ── Meta features (year, venue OHE, num_authors, title length, has_doi) ──────
def meta_features(df):
    venues = ["cav", "iclp", "kr", "lics", "lpnmr"]
    ven    = df["venue"].fillna("").str.lower().values.reshape(-1, 1)
    ohe    = OneHotEncoder(categories=[venues], handle_unknown="ignore", sparse_output=False)
    ven_oh = ohe.fit_transform(ven)
    year   = df["year"].fillna(2020).astype(int).values.reshape(-1, 1)
    year_n = (year - 2020) / 5.0
    n_auth = df["authors"].fillna("").apply(lambda s: 0 if not s else s.count(",") + 1).values.reshape(-1, 1)
    title_len_chars = df["title"].fillna("").str.len().values.reshape(-1, 1)
    title_len_tokens = df["title"].fillna("").str.split().apply(len).values.reshape(-1, 1)
    has_doi = (~df["doi"].fillna("").eq("")).astype(int).values.reshape(-1, 1)
    return np.hstack([ven_oh, year_n, n_auth, title_len_chars, title_len_tokens, has_doi]).astype(np.float32)

meta_tr   = meta_features(train)
meta_test = meta_features(pd.concat([pub, priv], ignore_index=True))

X_tr   = np.hstack([oof_probs,  oof_expv,  meta_tr]).astype(np.float32)
X_test = np.hstack([test_probs, test_expv, meta_test]).astype(np.float32)
print(f"\nStacked feature matrix: train {X_tr.shape}  test {X_test.shape}")


# ── LightGBM ordinal regression (predict expected score directly) ────────────
# Target = label as float; we regress, then map to label via optimized thresholds.
y_reg = y.astype(np.float32)

oof_reg  = np.zeros(len(train), dtype=np.float32)
test_reg = np.zeros(len(X_test), dtype=np.float32)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

lgb_params = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.03,
    num_leaves=31,
    min_data_in_leaf=20,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=5,
    lambda_l2=1.0,
    verbose=-1,
    seed=SEED,
)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr, y), 1):
    dtr  = lgb.Dataset(X_tr[tr_idx],  label=y_reg[tr_idx])
    dval = lgb.Dataset(X_tr[val_idx], label=y_reg[val_idx])
    booster = lgb.train(
        lgb_params, dtr, num_boost_round=3000,
        valid_sets=[dval], valid_names=["val"],
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
    )
    oof_reg[val_idx] = booster.predict(X_tr[val_idx], num_iteration=booster.best_iteration)
    test_reg       += booster.predict(X_test,       num_iteration=booster.best_iteration) / N_FOLDS
    print(f"  fold {fold}: best_iter={booster.best_iteration:4d}")


# ── Optimize thresholds on stacked OOF ───────────────────────────────────────
thr      = optimize_thresholds(oof_reg, y)
oof_pred = thr_to_label(oof_reg, thr)
oof_qwk  = qwk(y, oof_pred)
print(f"\nStacked OOF QWK: {oof_qwk:.4f}")
print(f"Thresholds     : {np.round(thr, 4)}")
print("OOF confusion:")
print(pd.crosstab(pd.Series(y, name="true"), pd.Series(oof_pred, name="pred")))


# ── Compare against best single base model and simple-average baseline ───────
print("\n— Baselines for context —")
# (a) equal weighting of base prob vectors
oof_avg  = np.mean([p for p in oof_stack],  axis=0)
test_avg = np.mean([p for p in test_stack], axis=0)
thr_avg  = optimize_thresholds(oof_avg, y)
print(f"  Equal-avg OOF QWK : {qwk(y, probs_to_label_thr(oof_avg, thr_avg)):.4f}")

# (b) best individual model
best_q, best_name = -1, None
for name, p in zip(names, oof_stack):
    t = optimize_thresholds(p, y)
    q = qwk(y, probs_to_label_thr(p, t))
    if q > best_q:
        best_q, best_name = q, name
print(f"  Best single model : {best_name}  OOF QWK = {best_q:.4f}")


# ── Final submission ─────────────────────────────────────────────────────────
final = thr_to_label(test_reg, thr)
pub_preds  = final[:len(pub)]
priv_preds = final[len(pub):]

pub_sub  = pd.DataFrame({"id": pub["id"],  "Label": pub_preds})
priv_sub = pd.DataFrame({"id": priv["id"], "Label": priv_preds})
combined = pd.concat([pub_sub, priv_sub], ignore_index=True)

combined.to_csv(DATA_DIR / "submission_stacking_v2.csv",         index=False)
pub_sub.to_csv (DATA_DIR / "public_submission_stacking_v2.csv",  index=False)
priv_sub.to_csv(DATA_DIR / "private_submission_stacking_v2.csv", index=False)

print(f"\nFinal submission: submission_stacking_v2.csv  ({len(combined)} rows)")
print("Pred dist:")
print(combined["Label"].value_counts().sort_index().to_string())
