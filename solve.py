#!/usr/bin/env python3
"""
Paper Classification — LinearSVC + TF-IDF + Ordinal Threshold Optimization
Metric: Quadratic Weighted Kappa (QWK)
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import cohen_kappa_score
from scipy.sparse import hstack, csr_matrix
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "/Users/quangnguyen/Desktop/paper-classi"

# ─── Load ─────────────────────────────────────────────────────────────────────
train = pd.read_csv(f"{DATA_DIR}/train.csv")
pub   = pd.read_csv(f"{DATA_DIR}/public_test.csv")
priv  = pd.read_csv(f"{DATA_DIR}/private_test.csv")

print(f"Train: {len(train)} | Public: {len(pub)} | Private: {len(priv)}")

# ─── Feature helpers ──────────────────────────────────────────────────────────
def make_text(df):
    title   = df["title"].fillna("")
    authors = df["authors"].fillna("")
    venue   = df["venue"].fillna("")
    return title + " " + venue + " " + venue + " " + authors

VENUES    = ["cav", "iclp", "kr", "lics", "lpnmr"]
YEAR_MEAN = train["year"].mean()
YEAR_STD  = train["year"].std()

def make_numeric(df):
    # Venue one-hot
    venue_dummies = pd.get_dummies(df["venue"], prefix="v")
    for v in VENUES:
        col = f"v_{v}"
        if col not in venue_dummies:
            venue_dummies[col] = 0
    venue_mat = venue_dummies[[f"v_{v}" for v in VENUES]].values.astype(float)

    # Year normalized
    year_scaled = ((df["year"] - YEAR_MEAN) / YEAR_STD).values.reshape(-1, 1)

    # DOI source
    doi_src = df["doi"].fillna("").apply(
        lambda x: 1 if "semanticscholar" in x else (2 if "doi.org" in x or str(x).startswith("10.") else 0)
    ).values.reshape(-1, 1)

    # Author count
    author_count = df["authors"].fillna("").apply(
        lambda x: len(x.split(",")) if x else 0
    ).values.reshape(-1, 1)

    # Title length (word count)
    title_len = df["title"].fillna("").apply(
        lambda x: len(x.split())
    ).values.reshape(-1, 1)

    return np.hstack([venue_mat, year_scaled, doi_src, author_count, title_len])

# ─── Build TF-IDF on all data (transductive) ──────────────────────────────────
all_text = pd.concat([make_text(train), make_text(pub), make_text(priv)], ignore_index=True)

tfidf_word = TfidfVectorizer(
    ngram_range=(1, 3),
    min_df=2,
    max_features=60000,
    sublinear_tf=True,
    analyzer="word",
    token_pattern=r"(?u)\b\w+\b",
)
tfidf_char = TfidfVectorizer(
    ngram_range=(3, 5),
    min_df=3,
    max_features=40000,
    sublinear_tf=True,
    analyzer="char_wb",
)

tfidf_word.fit(all_text)
tfidf_char.fit(all_text)

def build_X(df):
    text   = make_text(df)
    X_word = tfidf_word.transform(text)
    X_char = tfidf_char.transform(text)
    X_num  = csr_matrix(make_numeric(df))
    return hstack([X_word, X_char, X_num], format="csr")

X_train = build_X(train)
y_train = train["Label"].values
X_pub   = build_X(pub)
X_priv  = build_X(priv)

print(f"Feature matrix: {X_train.shape}")

# ─── Tune C ───────────────────────────────────────────────────────────────────
def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nTuning C:")
best_c, best_score = 0.1, 0.0
for C in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
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
# Use CalibratedClassifierCV to get OOF probabilities, then find optimal
# thresholds for the ordinal expected value that maximizes QWK.

print("\nOptimizing thresholds on OOF probabilities...")

calibrated = CalibratedClassifierCV(
    LinearSVC(C=best_c, max_iter=3000, dual=True), cv=skf, method="isotonic"
)
oof_probs = cross_val_predict(calibrated, X_train, y_train, cv=skf, method="predict_proba")
# oof_probs shape: (n_samples, 5) — columns correspond to labels 1-5

LABELS = np.array([1, 2, 3, 4, 5])

def probs_to_label(probs, thresholds):
    # Compute expected label then apply thresholds
    expected = probs @ LABELS                         # shape (n,)
    preds = np.ones(len(expected), dtype=int)
    for i, t in enumerate(thresholds):
        preds[expected > t] = i + 2
    return preds

def neg_qwk(thresholds, probs, y_true):
    thresholds = np.sort(thresholds)
    preds = probs_to_label(probs, thresholds)
    return -qwk(y_true, preds)

# Initial thresholds: evenly spaced between 1 and 5
init_thresholds = np.array([1.5, 2.5, 3.5, 4.5])
result = minimize(
    neg_qwk,
    init_thresholds,
    args=(oof_probs, y_train),
    method="Nelder-Mead",
    options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-4},
)
opt_thresholds = np.sort(result.x)
oof_preds_opt  = probs_to_label(oof_probs, opt_thresholds)
qwk_opt        = qwk(y_train, oof_preds_opt)

# Compare to baseline (no threshold opt)
oof_preds_base = cross_val_predict(
    LinearSVC(C=best_c, max_iter=3000, dual=True), X_train, y_train, cv=skf
)
qwk_base = qwk(y_train, oof_preds_base)

print(f"  Baseline CV QWK : {qwk_base:.4f}")
print(f"  Threshold CV QWK: {qwk_opt:.4f}")
print(f"  Thresholds      : {np.round(opt_thresholds, 3)}")

use_threshold = qwk_opt > qwk_base

# ─── Train final model ────────────────────────────────────────────────────────
if use_threshold:
    final_model = CalibratedClassifierCV(
        LinearSVC(C=best_c, max_iter=3000, dual=True), method="isotonic"
    )
    final_model.fit(X_train, y_train)

    def predict(X):
        probs = final_model.predict_proba(X)
        return probs_to_label(probs, opt_thresholds)
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

combined.to_csv(f"{DATA_DIR}/submission.csv", index=False)
pub_sub.to_csv(f"{DATA_DIR}/public_submission.csv",   index=False)
priv_sub.to_csv(f"{DATA_DIR}/private_submission.csv", index=False)

final_qwk = qwk_opt if use_threshold else qwk_base
print(f"\n{'='*40}")
print(f"Final CV QWK : {final_qwk:.4f}")
print(f"Submission   : submission.csv ({len(combined)} rows)")
print("Prediction distribution:")
print(combined["Label"].value_counts().sort_index().to_string())
