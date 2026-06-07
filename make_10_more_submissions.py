#!/usr/bin/env python3
"""
10 more Kaggle submissions — DIFFERENT strategies than A-E in make_5_submissions.py.

Different = different aggregation math, weighting, or post-processing, so they
won't all collapse to the same CSV. Goal: gamble on LB with diverse tactics.

F) geomean_all          — geometric mean of probs (vs arith mean in A)
G) rank_avg_all         — rank-based averaging (robust to scale)
H) lgbm_classif_all     — LGBM multiclass classification (vs regression in D)
I) lgbm_noprobs         — LGBM on expected-value + meta only (no raw probs)
J) specter_only         — single best model standalone
K) blend_specter_heavy  — manual 0.4/0.25/0.20/0.15 (specter2-dominated)
L) argmax_no_thr        — argmax of arith-mean (skip threshold opt)
M) top3_x_deberta_lite  — 0.9 * top3_avg + 0.1 * deberta (light DeBERTa infusion)
N) softmax_sharp        — equal-avg then sharpen with temperature 0.5
O) arith_geo_blend      — 0.5 * arith + 0.5 * geomean
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize
from scipy.stats import rankdata

import lightgbm as lgb


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--data-dir",  default=".")
p.add_argument("--probs-dir", default="./outputs")
p.add_argument("--out-dir",   default="./submissions",
               help="Root folder; writes to {out-dir}/{full,public,private}/")
p.add_argument("--folds", type=int, default=5)
p.add_argument("--seed",  type=int, default=42)
args = p.parse_args()

DATA_DIR  = Path(args.data_dir).resolve()
PROBS_DIR = Path(args.probs_dir).resolve()
OUT_DIR   = Path(args.out_dir).resolve()
for sub in ("full", "public", "private"):
    (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

NUM_LABELS = 5
LABEL_VALS = np.array([1, 2, 3, 4, 5], dtype=float)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
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

def normalize(p):
    p = np.clip(p, 1e-9, 1.0)
    return p / p.sum(axis=1, keepdims=True)


# ──────────────────────────────────────────────────────────────────────────────
# Load
# ──────────────────────────────────────────────────────────────────────────────
train = pd.read_csv(DATA_DIR / "train.csv")
pub   = pd.read_csv(DATA_DIR / "public_test.csv")
priv  = pd.read_csv(DATA_DIR / "private_test.csv")
y     = train["Label"].values
n_train, n_test = len(train), len(pub) + len(priv)

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
    probs[key] = {"oof": d["oof"], "test": d["test"]}

if not probs:
    raise SystemExit(f"No NPZ in {PROBS_DIR}")
keys_all = list(probs.keys())
print(f"Loaded {len(probs)} models: {keys_all}")

oof_qwks = {}
for k, d in probs.items():
    thr = optimize_thresholds(d["oof"], y)
    oof_qwks[k] = qwk(y, probs_to_label_thr(d["oof"], thr))
    print(f"  {k:16s}  OOF QWK = {oof_qwks[k]:.4f}")

best_key = max(oof_qwks, key=oof_qwks.get)
top3 = sorted(oof_qwks, key=oof_qwks.get, reverse=True)[:3]
print(f"\nBest single: {best_key}   Top-3: {top3}")


_PUB_IDS  = set(pub["id"])
_PRIV_IDS = set(priv["id"])

def write_sub(name, labels, oof_qwk_val):
    """Write full + public-only + private-only CSV variants."""
    sub = pd.DataFrame({
        "id":    list(pub["id"]) + list(priv["id"]),
        "Label": labels.astype(int),
    })
    fname = f"submission_{name}.csv"
    sub.to_csv(OUT_DIR / "full" / fname, index=False)
    sub[sub["id"].isin(_PUB_IDS )].to_csv(OUT_DIR / "public"  / fname, index=False)
    sub[sub["id"].isin(_PRIV_IDS)].to_csv(OUT_DIR / "private" / fname, index=False)
    print(f"  wrote {fname}  OOF={oof_qwk_val:.4f}  "
          f"dist={dict(sub['Label'].value_counts().sort_index())}")


# ──────────────────────────────────────────────────────────────────────────────
# F) geomean_all — geometric mean of probs (different aggregation)
# ──────────────────────────────────────────────────────────────────────────────
print("\n[F] geomean_all")
log_oof  = np.mean([np.log(np.clip(probs[k]["oof"],  1e-9, 1)) for k in keys_all], axis=0)
log_test = np.mean([np.log(np.clip(probs[k]["test"], 1e-9, 1)) for k in keys_all], axis=0)
oof_F  = normalize(np.exp(log_oof))
test_F = normalize(np.exp(log_test))
thr = optimize_thresholds(oof_F, y)
qF = qwk(y, probs_to_label_thr(oof_F, thr))
write_sub("F_geomean_all", probs_to_label_thr(test_F, thr), qF)


# ──────────────────────────────────────────────────────────────────────────────
# G) rank_avg_all — rank-based averaging of expected values
# ──────────────────────────────────────────────────────────────────────────────
print("\n[G] rank_avg_all")
ev_oof_stack  = np.stack([expected_value(probs[k]["oof"])  for k in keys_all], axis=1)
ev_test_stack = np.stack([expected_value(probs[k]["test"]) for k in keys_all], axis=1)
rank_oof  = np.mean([rankdata(ev_oof_stack[:, i],  method="average") for i in range(len(keys_all))], axis=0)
rank_test = np.mean([rankdata(ev_test_stack[:, i], method="average") for i in range(len(keys_all))], axis=0)
# Scale ranks to [1, 5] range for threshold optimization
rank_oof  = 1 + 4 * (rank_oof  - rank_oof.min())  / (rank_oof.max()  - rank_oof.min() + 1e-9)
rank_test = 1 + 4 * (rank_test - rank_test.min()) / (rank_test.max() - rank_test.min() + 1e-9)
thr = optimize_thresholds(rank_oof, y)
qG = qwk(y, thr_to_label(rank_oof, thr))
write_sub("G_rank_avg_all", thr_to_label(rank_test, thr), qG)


# ──────────────────────────────────────────────────────────────────────────────
# Meta features for H / I
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


# ──────────────────────────────────────────────────────────────────────────────
# H) lgbm_classif_all — LGBM multiclass classification (vs regression in D)
# ──────────────────────────────────────────────────────────────────────────────
print("\n[H] lgbm_classif_all")
oof_concat  = np.concatenate([probs[k]["oof"]  for k in keys_all], axis=1)
test_concat = np.concatenate([probs[k]["test"] for k in keys_all], axis=1)
ev_oof  = np.stack([expected_value(probs[k]["oof"])  for k in keys_all], axis=1)
ev_test = np.stack([expected_value(probs[k]["test"]) for k in keys_all], axis=1)
X_tr   = np.hstack([oof_concat,  ev_oof,  meta_tr  ]).astype(np.float32)
X_test = np.hstack([test_concat, ev_test, meta_test]).astype(np.float32)
y_cls  = (y - 1).astype(np.int32)  # 0..4

oof_p_H  = np.zeros((n_train, NUM_LABELS), dtype=np.float32)
test_p_H = np.zeros((n_test,  NUM_LABELS), dtype=np.float32)
skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
params_H = dict(objective="multiclass", num_class=NUM_LABELS, metric="multi_logloss",
                learning_rate=0.05, num_leaves=31, min_data_in_leaf=20,
                feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
                lambda_l2=1.0, verbose=-1, seed=args.seed)
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr, y), 1):
    dtr  = lgb.Dataset(X_tr[tr_idx],  label=y_cls[tr_idx])
    dval = lgb.Dataset(X_tr[val_idx], label=y_cls[val_idx])
    b = lgb.train(params_H, dtr, num_boost_round=2000,
                  valid_sets=[dval], valid_names=["val"],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    oof_p_H[val_idx]  = b.predict(X_tr[val_idx], num_iteration=b.best_iteration)
    test_p_H         += b.predict(X_test,        num_iteration=b.best_iteration) / args.folds
thr = optimize_thresholds(oof_p_H, y)
qH = qwk(y, probs_to_label_thr(oof_p_H, thr))
write_sub("H_lgbm_classif_all", probs_to_label_thr(test_p_H, thr), qH)


# ──────────────────────────────────────────────────────────────────────────────
# I) lgbm_noprobs — LGBM on expected-values + meta only (no raw probs)
# ──────────────────────────────────────────────────────────────────────────────
print("\n[I] lgbm_noprobs")
X_tr_I   = np.hstack([ev_oof,  meta_tr  ]).astype(np.float32)
X_test_I = np.hstack([ev_test, meta_test]).astype(np.float32)
y_reg = y.astype(np.float32)
oof_reg  = np.zeros(n_train, dtype=np.float32)
test_reg = np.zeros(n_test,  dtype=np.float32)
params_I = dict(objective="regression", metric="rmse", learning_rate=0.03,
                num_leaves=15, min_data_in_leaf=20, feature_fraction=0.9,
                bagging_fraction=0.9, bagging_freq=5, lambda_l2=1.0,
                verbose=-1, seed=args.seed)
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr_I, y), 1):
    dtr  = lgb.Dataset(X_tr_I[tr_idx],  label=y_reg[tr_idx])
    dval = lgb.Dataset(X_tr_I[val_idx], label=y_reg[val_idx])
    b = lgb.train(params_I, dtr, num_boost_round=3000,
                  valid_sets=[dval], valid_names=["val"],
                  callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
    oof_reg[val_idx]  = b.predict(X_tr_I[val_idx], num_iteration=b.best_iteration)
    test_reg         += b.predict(X_test_I,        num_iteration=b.best_iteration) / args.folds
thr = optimize_thresholds(oof_reg, y)
qI = qwk(y, thr_to_label(oof_reg, thr))
write_sub("I_lgbm_noprobs", thr_to_label(test_reg, thr), qI)


# ──────────────────────────────────────────────────────────────────────────────
# J) specter_only — single best model alone
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n[J] {best_key}_only")
oof_J  = probs[best_key]["oof"]
test_J = probs[best_key]["test"]
thr = optimize_thresholds(oof_J, y)
qJ = qwk(y, probs_to_label_thr(oof_J, thr))
write_sub(f"J_{best_key.replace('-','_')}_only", probs_to_label_thr(test_J, thr), qJ)


# ──────────────────────────────────────────────────────────────────────────────
# K) blend_specter_heavy — manual specter2-dominated weights
# ──────────────────────────────────────────────────────────────────────────────
print("\n[K] blend_specter_heavy")
manual_weights = {"specter2": 0.40, "e5-large": 0.25, "scibert": 0.20, "deberta-v3": 0.15}
# Only apply to models we have; renormalize
weights_K = {k: w for k, w in manual_weights.items() if k in probs}
total = sum(weights_K.values())
weights_K = {k: w / total for k, w in weights_K.items()}
print(f"    weights → {weights_K}")
oof_K  = sum(w * probs[k]["oof"]  for k, w in weights_K.items())
test_K = sum(w * probs[k]["test"] for k, w in weights_K.items())
thr = optimize_thresholds(oof_K, y)
qK = qwk(y, probs_to_label_thr(oof_K, thr))
write_sub("K_blend_specter_heavy", probs_to_label_thr(test_K, thr), qK)


# ──────────────────────────────────────────────────────────────────────────────
# L) argmax_no_thr — argmax of arith-mean (no threshold optimization)
# ──────────────────────────────────────────────────────────────────────────────
print("\n[L] argmax_no_thr")
oof_L  = np.mean([probs[k]["oof"]  for k in keys_all], axis=0)
test_L = np.mean([probs[k]["test"] for k in keys_all], axis=0)
pred_L_oof  = np.argmax(oof_L,  axis=1) + 1
pred_L_test = np.argmax(test_L, axis=1) + 1
qL = qwk(y, pred_L_oof)
write_sub("L_argmax_no_thr", pred_L_test, qL)


# ──────────────────────────────────────────────────────────────────────────────
# M) top3_x_deberta_lite — 0.9 * top3_avg + 0.1 * deberta (light infusion)
# ──────────────────────────────────────────────────────────────────────────────
if "deberta-v3" in probs and len(top3) >= 3:
    print("\n[M] top3_x_deberta_lite")
    top3_avg_oof  = np.mean([probs[k]["oof"]  for k in top3], axis=0)
    top3_avg_test = np.mean([probs[k]["test"] for k in top3], axis=0)
    deb_oof  = probs["deberta-v3"]["oof"]
    deb_test = probs["deberta-v3"]["test"]
    oof_M  = 0.9 * top3_avg_oof  + 0.1 * deb_oof  if "deberta-v3" not in top3 else top3_avg_oof
    test_M = 0.9 * top3_avg_test + 0.1 * deb_test if "deberta-v3" not in top3 else top3_avg_test
    thr = optimize_thresholds(oof_M, y)
    qM = qwk(y, probs_to_label_thr(oof_M, thr))
    write_sub("M_top3_x_deberta_lite", probs_to_label_thr(test_M, thr), qM)
else:
    qM = None
    print("\n[M] SKIP (need deberta-v3 + 3 strong models)")


# ──────────────────────────────────────────────────────────────────────────────
# N) softmax_sharp — equal-avg then sharpen probs with temperature 0.5
# ──────────────────────────────────────────────────────────────────────────────
print("\n[N] softmax_sharp  (T=0.5)")
T = 0.5
sharp_oof  = normalize(np.power(np.clip(np.mean([probs[k]["oof"]  for k in keys_all], axis=0), 1e-9, 1), 1.0 / T))
sharp_test = normalize(np.power(np.clip(np.mean([probs[k]["test"] for k in keys_all], axis=0), 1e-9, 1), 1.0 / T))
thr = optimize_thresholds(sharp_oof, y)
qN = qwk(y, probs_to_label_thr(sharp_oof, thr))
write_sub("N_softmax_sharp", probs_to_label_thr(sharp_test, thr), qN)


# ──────────────────────────────────────────────────────────────────────────────
# O) arith_geo_blend — 0.5 * arith_mean + 0.5 * geomean
# ──────────────────────────────────────────────────────────────────────────────
print("\n[O] arith_geo_blend")
arith_oof  = np.mean([probs[k]["oof"]  for k in keys_all], axis=0)
arith_test = np.mean([probs[k]["test"] for k in keys_all], axis=0)
oof_O  = normalize(0.5 * arith_oof  + 0.5 * oof_F)
test_O = normalize(0.5 * arith_test + 0.5 * test_F)
thr = optimize_thresholds(oof_O, y)
qO = qwk(y, probs_to_label_thr(oof_O, thr))
write_sub("O_arith_geo_blend", probs_to_label_thr(test_O, thr), qO)


# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 60)
print("SUMMARY (OOF QWK desc):")
print("═" * 60)
ranking = [("F_geomean_all", qF), ("G_rank_avg_all", qG),
           ("H_lgbm_classif_all", qH), ("I_lgbm_noprobs", qI),
           (f"J_{best_key.replace('-','_')}_only", qJ), ("K_blend_specter_heavy", qK),
           ("L_argmax_no_thr", qL), ("M_top3_x_deberta_lite", qM),
           ("N_softmax_sharp", qN), ("O_arith_geo_blend", qO)]
ranking = [(n, q) for n, q in ranking if q is not None]
ranking.sort(key=lambda x: x[1], reverse=True)
for i, (name, q) in enumerate(ranking, 1):
    print(f"  {i:2d}. submission_{name}.csv   OOF QWK = {q:.4f}")
print("\nLưu ý: OOF chỉ là proxy. Submit lên Kaggle để biết LB thật.")
