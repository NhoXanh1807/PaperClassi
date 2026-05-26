#!/usr/bin/env python3
"""
Generate 5 submission CSVs from outputs/*_probs.npz so you can submit all to
Kaggle and pick the best-performing one on the leaderboard.

Strategies (sorted simple -> complex):
  A) equal_avg_all4      — average probs of 4 CORN models
  B) equal_avg_drop_deb  — drop DeBERTa-v3-large (weakest @ 0.582), avg 3 best
  C) qwk_weighted_top3   — weight ∝ OOF_QWK^4 over the 3 best models
  D) lgbm_stack_all4     — LightGBM regressor on probs+meta features (all 4)
  E) lgbm_stack_top3     — Same as D, but drop DeBERTa

For each: optimize 4 thresholds on OOF, then map test expected-score → label.

Outputs (in OUT_DIR):
  submission_A_equal_avg_all4.csv
  submission_B_equal_avg_drop_deb.csv
  submission_C_qwk_weighted_top3.csv
  submission_D_lgbm_stack_all4.csv
  submission_E_lgbm_stack_top3.csv

Prints OOF QWK for each so you know the expected ranking.
"""

import argparse, json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize

import lightgbm as lgb


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=".",
               help="Folder with train.csv, public_test.csv, private_test.csv")
p.add_argument("--probs-dir", default="./outputs",
               help="Folder with *_probs.npz from kaggle_corn_v2.py")
p.add_argument("--out-dir", default=".",
               help="Where to write submission_*.csv")
p.add_argument("--folds", type=int, default=5)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

DATA_DIR  = Path(args.data_dir).resolve()
PROBS_DIR = Path(args.probs_dir).resolve()
OUT_DIR   = Path(args.out_dir).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_LABELS = 5
LABEL_VALS = np.array([1, 2, 3, 4, 5], dtype=float)


# ──────────────────────────────────────────────────────────────────────────────
# QWK + threshold helpers
# ──────────────────────────────────────────────────────────────────────────────
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
    expected = expected_value(values) if values.ndim == 2 else values
    def neg(thr):
        return -qwk(y_true_1to5, thr_to_label(expected, np.sort(thr)))
    res = minimize(neg, init, method="Nelder-Mead",
                   options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6})
    return np.sort(res.x)


# ──────────────────────────────────────────────────────────────────────────────
# Load data + probs
# ──────────────────────────────────────────────────────────────────────────────
train = pd.read_csv(DATA_DIR / "train.csv")
pub   = pd.read_csv(DATA_DIR / "public_test.csv")
priv  = pd.read_csv(DATA_DIR / "private_test.csv")
y     = train["Label"].values
n_train = len(train)
n_test  = len(pub) + len(priv)

MODEL_MAP = {
    "scibert":         "allenai_scibert_scivocab_uncased_probs.npz",
    "specter2":        "allenai_specter2_base_probs.npz",
    "deberta-v3":      "microsoft_deberta-v3-large_probs.npz",
    "e5-large":        "intfloat_e5-large-v2_probs.npz",
    "deberta-v3-base": "microsoft_deberta-v3-base_probs.npz",
    "specter-v1":      "allenai_specter_probs.npz",
    "mpnet-base":      "microsoft_mpnet-base_probs.npz",
    "roberta-base":    "FacebookAI_roberta-base_probs.npz",
}

probs = {}
for key, fname in MODEL_MAP.items():
    fp = PROBS_DIR / fname
    if not fp.exists():
        continue
    d = np.load(fp)
    assert d["oof"].shape  == (n_train, NUM_LABELS), f"{fname} oof shape {d['oof'].shape}"
    assert d["test"].shape == (n_test,  NUM_LABELS), f"{fname} test shape {d['test'].shape}"
    probs[key] = {"oof": d["oof"], "test": d["test"]}

if not probs:
    raise SystemExit(f"No NPZ files found in {PROBS_DIR}")
print(f"Loaded {len(probs)} models: {list(probs)}")

# Per-model OOF QWK + ranking
oof_qwks = {}
for k, d in probs.items():
    thr = optimize_thresholds(d["oof"], y)
    oof_qwks[k] = qwk(y, probs_to_label_thr(d["oof"], thr))
    print(f"  {k:16s}  OOF QWK = {oof_qwks[k]:.4f}")

# Drop models with OOF below threshold relative to best (within 0.07 of max)
best_q   = max(oof_qwks.values())
QWK_CUT  = best_q - 0.07          # eg if best 0.66, cut at 0.59
strong   = [k for k, q in oof_qwks.items() if q >= QWK_CUT]
weak     = [k for k in oof_qwks if k not in strong]
print(f"\nStrong models (OOF ≥ {QWK_CUT:.3f}): {strong}")
if weak:
    print(f"Weak (excluded from B/E): {weak}")

top_n   = min(3, len(strong))
topN    = sorted(strong, key=oof_qwks.get, reverse=True)[:top_n]
print(f"Top-{top_n}: {topN}")


# ──────────────────────────────────────────────────────────────────────────────
# Submission helpers
# ──────────────────────────────────────────────────────────────────────────────
def write_submission(name, test_pred_labels, oof_qwk_val):
    """test_pred_labels: int array of length n_test (1..5)."""
    sub = pd.DataFrame({
        "id":    list(pub["id"]) + list(priv["id"]),
        "Label": test_pred_labels.astype(int),
    })
    fp = OUT_DIR / f"submission_{name}.csv"
    sub.to_csv(fp, index=False)
    print(f"  wrote {fp.name}  (OOF QWK ≈ {oof_qwk_val:.4f})  "
          f"dist={dict(sub['Label'].value_counts().sort_index())}")


def avg_and_submit(name, keys, weights=None):
    """Average probs (optionally weighted), optimize thr, predict, write."""
    if weights is None:
        weights = np.ones(len(keys)) / len(keys)
    weights = np.asarray(weights, dtype=float) / np.sum(weights)
    oof_mix  = sum(w * probs[k]["oof"]  for k, w in zip(keys, weights))
    test_mix = sum(w * probs[k]["test"] for k, w in zip(keys, weights))
    thr = optimize_thresholds(oof_mix, y)
    q   = qwk(y, probs_to_label_thr(oof_mix, thr))
    test_pred = probs_to_label_thr(test_mix, thr)
    write_submission(name, test_pred, q)
    return q


# ──────────────────────────────────────────────────────────────────────────────
# A) Equal-weight ALL models
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n[A] equal_avg_all  ({list(probs.keys())})")
qA = avg_and_submit("A_equal_avg_all", list(probs.keys()))

# ──────────────────────────────────────────────────────────────────────────────
# B) Equal-weight, drop weak models (OOF < cutoff)
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n[B] equal_avg_strong  ({strong})")
qB = avg_and_submit("B_equal_avg_strong", strong)

# ──────────────────────────────────────────────────────────────────────────────
# C) QWK-weighted top-N (strongest only)
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n[C] qwk_weighted_top{top_n}  ({topN})")
weights_C = np.array([oof_qwks[k] ** 4 for k in topN])
print(f"    weights ∝ qwk^4 → {dict(zip(topN, np.round(weights_C / weights_C.sum(), 3)))}")
qC = avg_and_submit(f"C_qwk_weighted_top{top_n}", topN, weights_C)


# ──────────────────────────────────────────────────────────────────────────────
# Meta features for D / E
# ──────────────────────────────────────────────────────────────────────────────
def meta_features(df):
    venues = ["cav", "iclp", "kr", "lics", "lpnmr"]
    ven    = np.asarray(df["venue"].fillna("").str.lower().to_numpy(), dtype=object).reshape(-1, 1)
    ohe    = OneHotEncoder(categories=[venues], handle_unknown="ignore", sparse_output=False)
    ven_oh = ohe.fit_transform(ven)
    year   = df["year"].fillna(2020).astype(int).to_numpy().reshape(-1, 1)
    year_n = (year - 2020) / 5.0
    n_auth = df["authors"].fillna("").apply(lambda s: 0 if not s else s.count(",") + 1).to_numpy().reshape(-1, 1)
    title_len_chars  = df["title"].fillna("").str.len().to_numpy().reshape(-1, 1)
    title_len_tokens = df["title"].fillna("").str.split().apply(len).to_numpy().reshape(-1, 1)
    has_doi = (~df["doi"].fillna("").eq("")).astype(int).to_numpy().reshape(-1, 1)
    return np.hstack([ven_oh, year_n, n_auth, title_len_chars, title_len_tokens, has_doi]).astype(np.float32)

meta_tr   = meta_features(train)
meta_test = meta_features(pd.concat([pub, priv], ignore_index=True))


def lgbm_stack(name, keys):
    """Stack given model keys via LightGBM regression on probs+expv+meta features."""
    oof_stack  = [probs[k]["oof"]  for k in keys]
    test_stack = [probs[k]["test"] for k in keys]
    oof_probs  = np.concatenate(oof_stack,  axis=1)
    test_probs = np.concatenate(test_stack, axis=1)
    oof_expv   = np.stack([expected_value(p) for p in oof_stack],  axis=1)
    test_expv  = np.stack([expected_value(p) for p in test_stack], axis=1)

    X_tr   = np.hstack([oof_probs,  oof_expv,  meta_tr  ]).astype(np.float32)
    X_test = np.hstack([test_probs, test_expv, meta_test]).astype(np.float32)
    y_reg  = y.astype(np.float32)

    oof_reg  = np.zeros(n_train, dtype=np.float32)
    test_reg = np.zeros(n_test,  dtype=np.float32)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    lgb_params = dict(
        objective="regression", metric="rmse",
        learning_rate=0.03, num_leaves=31, min_data_in_leaf=20,
        feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
        lambda_l2=1.0, verbose=-1, seed=args.seed,
    )
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr, y), 1):
        dtr  = lgb.Dataset(X_tr[tr_idx],  label=y_reg[tr_idx])
        dval = lgb.Dataset(X_tr[val_idx], label=y_reg[val_idx])
        b = lgb.train(lgb_params, dtr, num_boost_round=3000,
                      valid_sets=[dval], valid_names=["val"],
                      callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
        oof_reg[val_idx] = b.predict(X_tr[val_idx],  num_iteration=b.best_iteration)
        test_reg        += b.predict(X_test,        num_iteration=b.best_iteration) / args.folds

    thr = optimize_thresholds(oof_reg, y)
    q   = qwk(y, thr_to_label(oof_reg, thr))
    test_pred = thr_to_label(test_reg, thr)
    write_submission(name, test_pred, q)
    return q


# ──────────────────────────────────────────────────────────────────────────────
# D) LGBM stacking ALL models
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n[D] lgbm_stack_all  ({list(probs.keys())})")
qD = lgbm_stack("D_lgbm_stack_all", list(probs.keys()))

# ──────────────────────────────────────────────────────────────────────────────
# E) LGBM stacking, strong models only
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n[E] lgbm_stack_strong  ({strong})")
qE = lgbm_stack("E_lgbm_stack_strong", strong)


# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 60)
print("SUMMARY — sort submissions by OOF QWK desc:")
print("═" * 60)
ranking = sorted(
    [("A_equal_avg_all", qA), ("B_equal_avg_strong", qB),
     (f"C_qwk_weighted_top{top_n}", qC), ("D_lgbm_stack_all", qD),
     ("E_lgbm_stack_strong", qE)],
    key=lambda x: x[1], reverse=True,
)
for i, (name, q) in enumerate(ranking, 1):
    print(f"  {i}. submission_{name}.csv   OOF QWK = {q:.4f}")
print("\nNộp tất cả 5 lên Kaggle, public LB sẽ cho biết cái nào generalize tốt nhất.")
