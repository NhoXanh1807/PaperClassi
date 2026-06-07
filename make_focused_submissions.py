#!/usr/bin/env python3
"""
5 focused submissions targeting subspace near A=0.72727 (current best LB).

Lessons learned from prior LB:
  - Equal weight (0.25 x 4) > all manual weights tried (diversity beats QWK-tuning)
  - DeBERTa is essential (dropping it costs 0.016 LB)
  - LGBM stacking consistently worse than simple averaging
  - Specter-heavy K (0.72124) close to A — slight asymmetry might help

Untested directions (DIFFERENT from existing A-O):
  P) equal_drop_e5         - drop e5-large instead of deberta (probe importance)
  Q) equal_drop_scibert    - drop scibert (probe importance)
  R) weight_specter_plus   - (0.20, 0.30, 0.25, 0.25) — gentle specter emphasis
  S) weight_deberta_plus   - (0.20, 0.20, 0.35, 0.25) — counter-intuitive deberta boost
  T) median_label_all      - per-row median of 4 single-model labels (ordinal-friendly)
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize


# ──────────────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--data-dir",  default=".")
p.add_argument("--probs-dir", default="./outputs")
p.add_argument("--out-dir",   default="./submissions",
               help="Root folder; writes to {out-dir}/{full,public,private}/")
args = p.parse_args()

DATA_DIR  = Path(args.data_dir).resolve()
PROBS_DIR = Path(args.probs_dir).resolve()
OUT_DIR   = Path(args.out_dir).resolve()
for sub in ("full", "public", "private"):
    (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

NUM_LABELS = 5
LABEL_VALS = np.array([1, 2, 3, 4, 5], dtype=float)

def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")

def expected_value(p):
    return p @ LABEL_VALS

def thr_to_label(expected, thr):
    out = np.ones(len(expected), dtype=int)
    for i, t in enumerate(thr):
        out[expected > t] = i + 2
    return out

def probs_to_label(p, thr):
    return thr_to_label(expected_value(p), thr)

def optimize_thr(values, y_true, init=(1.5, 2.5, 3.5, 4.5)):
    expected = expected_value(values) if values.ndim == 2 else values
    def neg(t):
        return -qwk(y_true, thr_to_label(expected, np.sort(t)))
    return np.sort(minimize(neg, init, method="Nelder-Mead",
                            options={"maxiter": 5000, "xatol": 1e-6}).x)


# ──────────────────────────────────────────────────────────────────────────────
train = pd.read_csv(DATA_DIR / "train.csv")
pub   = pd.read_csv(DATA_DIR / "public_test.csv")
priv  = pd.read_csv(DATA_DIR / "private_test.csv")
y     = train["Label"].values

MODEL_MAP = {
    "scibert":    "allenai_scibert_scivocab_uncased_probs.npz",
    "specter2":   "allenai_specter2_base_probs.npz",
    "deberta-v3": "microsoft_deberta-v3-large_probs.npz",
    "e5-large":   "intfloat_e5-large-v2_probs.npz",
}
probs = {}
for k, fname in MODEL_MAP.items():
    fp = PROBS_DIR / fname
    if fp.exists():
        d = np.load(fp)
        probs[k] = {"oof": d["oof"], "test": d["test"]}

assert len(probs) == 4, f"Need all 4 models, got {list(probs)}"
print(f"Loaded: {list(probs)}\n")


_PUB_IDS  = set(pub["id"])
_PRIV_IDS = set(priv["id"])

def write_sub(name, labels, qoof):
    """Write full + public-only + private-only CSV variants."""
    sub = pd.DataFrame({"id": list(pub["id"]) + list(priv["id"]),
                        "Label": labels.astype(int)})
    fname = f"submission_{name}.csv"
    sub.to_csv(OUT_DIR / "full" / fname, index=False)
    sub[sub["id"].isin(_PUB_IDS )].to_csv(OUT_DIR / "public"  / fname, index=False)
    sub[sub["id"].isin(_PRIV_IDS)].to_csv(OUT_DIR / "private" / fname, index=False)
    print(f"  wrote {fname}  OOF={qoof:.4f}  "
          f"dist={dict(sub['Label'].value_counts().sort_index())}")


def weighted_blend(name, weights):
    """weights: dict model->weight."""
    w = np.array(list(weights.values()), dtype=float)
    w /= w.sum()
    keys = list(weights.keys())
    oof  = sum(wi * probs[k]["oof"]  for wi, k in zip(w, keys))
    test = sum(wi * probs[k]["test"] for wi, k in zip(w, keys))
    thr = optimize_thr(oof, y)
    q   = qwk(y, probs_to_label(oof, thr))
    write_sub(name, probs_to_label(test, thr), q)
    return q


# P) equal_drop_e5 — (1/3 each: scibert + specter2 + deberta)
print("[P] equal_drop_e5  (scibert + specter2 + deberta)")
qP = weighted_blend("P_equal_drop_e5",
    {"scibert": 1, "specter2": 1, "deberta-v3": 1})

# Q) equal_drop_scibert — (1/3 each: specter2 + deberta + e5)
print("\n[Q] equal_drop_scibert  (specter2 + deberta + e5)")
qQ = weighted_blend("Q_equal_drop_scibert",
    {"specter2": 1, "deberta-v3": 1, "e5-large": 1})

# R) weight_specter_plus — (0.20, 0.30, 0.25, 0.25)
print("\n[R] weight_specter_plus  (0.20/0.30/0.25/0.25)")
qR = weighted_blend("R_weight_specter_plus",
    {"scibert": 0.20, "specter2": 0.30, "deberta-v3": 0.25, "e5-large": 0.25})

# S) weight_deberta_plus — (0.20, 0.20, 0.35, 0.25)
print("\n[S] weight_deberta_plus  (0.20/0.20/0.35/0.25)  — counter-intuitive")
qS = weighted_blend("S_weight_deberta_plus",
    {"scibert": 0.20, "specter2": 0.20, "deberta-v3": 0.35, "e5-large": 0.25})

# T) median_label_all — per-row median of each model's single-model label
print("\n[T] median_label_all  (median of 4 individual predictions)")
# Each model gets own threshold; predict label per model per row; median across models
single_labels_oof  = []
single_labels_test = []
for k, d in probs.items():
    thr_k = optimize_thr(d["oof"], y)
    single_labels_oof.append (probs_to_label(d["oof"],  thr_k))
    single_labels_test.append(probs_to_label(d["test"], thr_k))
# Stack and take median per row (for ordinal labels, median is well-defined)
stack_oof  = np.stack(single_labels_oof,  axis=1)   # (N, 4)
stack_test = np.stack(single_labels_test, axis=1)
median_oof  = np.median(stack_oof,  axis=1).round().astype(int)
median_test = np.median(stack_test, axis=1).round().astype(int)
qT = qwk(y, median_oof)
write_sub("T_median_label_all", median_test, qT)


# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 60)
print("SUMMARY (OOF QWK desc):")
print("═" * 60)
ranking = sorted([("P_equal_drop_e5", qP), ("Q_equal_drop_scibert", qQ),
                  ("R_weight_specter_plus", qR), ("S_weight_deberta_plus", qS),
                  ("T_median_label_all", qT)],
                 key=lambda x: x[1], reverse=True)
for i, (n, q) in enumerate(ranking, 1):
    print(f"  {i}. submission_{n}.csv   OOF QWK = {q:.4f}")
