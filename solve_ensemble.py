#!/usr/bin/env python3
"""
Ensemble: average OOF probabilities from SPECTER2 + SciBERT,
re-optimize thresholds on the averaged OOF, then apply to averaged test probs.

Requires:
  - scibert_probs.npz  (saved by solve_finetune_scibert.py)
  - specter2_probs.npz (saved by solve_finetune.py — re-run if missing)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize

DATA_DIR = Path("/Users/quangnguyen/Desktop/paper-classi")
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

# ── Load ──────────────────────────────────────────────────────────────────────
train = pd.read_csv(DATA_DIR / "train.csv")
pub   = pd.read_csv(DATA_DIR / "public_test.csv")
priv  = pd.read_csv(DATA_DIR / "private_test.csv")
y_true = train["Label"].values

spec = np.load(DATA_DIR / "specter2_probs.npz")
sci  = np.load(DATA_DIR / "scibert_probs.npz")

print(f"SPECTER2 OOF shape: {spec['oof'].shape}")
print(f"SciBERT  OOF shape: {sci['oof'].shape}")

# ── Per-model OOF baseline ────────────────────────────────────────────────────
print("\n— Per-model OOF QWK —")
for name, p in [("SPECTER2", spec), ("SciBERT", sci)]:
    base_thr = optimize_thresholds(p["oof"], y_true)
    base_qwk = qwk(y_true, probs_to_label(p["oof"], base_thr))
    print(f"  {name:9s}: {base_qwk:.4f}  thr={np.round(base_thr,3)}")

# ── Try multiple ensemble weights ─────────────────────────────────────────────
print("\n— Weighted ensemble (w_specter, w_scibert) —")
best_w, best_qwk, best_thr = (0.5, 0.5), 0.0, None
for w_spec in np.linspace(0.3, 0.9, 7):
    w_sci = 1.0 - w_spec
    oof_mix = w_spec * spec["oof"] + w_sci * sci["oof"]
    thr = optimize_thresholds(oof_mix, y_true)
    q   = qwk(y_true, probs_to_label(oof_mix, thr))
    marker = " ←" if q > best_qwk else ""
    print(f"  spec={w_spec:.2f} sci={w_sci:.2f}  QWK={q:.4f}{marker}")
    if q > best_qwk:
        best_qwk = q
        best_w   = (w_spec, w_sci)
        best_thr = thr

w_spec, w_sci = best_w
print(f"\nBest weights: SPECTER2={w_spec:.2f}, SciBERT={w_sci:.2f}")
print(f"Best OOF QWK: {best_qwk:.4f}")
print(f"Thresholds  : {np.round(best_thr,3)}")

# ── Apply to test ─────────────────────────────────────────────────────────────
test_mix   = w_spec * spec["test"] + w_sci * sci["test"]
test_preds = probs_to_label(test_mix, best_thr)

pub_preds  = test_preds[:len(pub)]
priv_preds = test_preds[len(pub):]

pub_sub  = pd.DataFrame({"id": pub["id"],  "Label": pub_preds})
priv_sub = pd.DataFrame({"id": priv["id"], "Label": priv_preds})
combined = pd.concat([pub_sub, priv_sub], ignore_index=True)

combined.to_csv(DATA_DIR / "submission_ensemble.csv",        index=False)
pub_sub.to_csv(DATA_DIR  / "public_submission_ensemble.csv", index=False)
priv_sub.to_csv(DATA_DIR / "private_submission_ensemble.csv",index=False)

print(f"\nSubmission: submission_ensemble.csv ({len(combined)} rows)")
print("Pred dist:")
print(combined["Label"].value_counts().sort_index().to_string())
