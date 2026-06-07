# PaperClassi — Ordinal Paper Classification with CORN + Transformer Ensemble

A reproducible pipeline for a 5-class **ordinal** paper-classification Kaggle
competition (2,494 train / 596 test). The final selected submission scored
**Public LB = 0.72727** (Quadratic Weighted Kappa) using a simple
equal-weight ensemble of four CORN-trained transformer backbones.

---

## TL;DR

```bash
# 1) On a rented GPU VM (recommended: Vast.ai 3090 / 4090, ~$0.30/hr, ~5h)
git clone https://github.com/NhoXanh1807/PaperClassi.git
cd PaperClassi
pip install -r requirements.txt
python kaggle_corn_v2.py --data-dir . --out-dir ./outputs

# 2) Locally, after copying outputs/*.npz back to ./outputs/
python make_5_submissions.py        # 5 ensembling strategies
python make_10_more_submissions.py  # 10 diverse aggregation strategies
python make_focused_submissions.py  # 5 focused weight perturbations

# Submit any CSV from ./submissions/full/ to Kaggle
```

---

## Repository layout

```
PaperClassi/
├── README.md                        # this file
├── requirements.txt
├── kaggle_corn_v2.py                # main training script (GPU)
├── make_5_submissions.py            # 5 baseline ensembles (A–E)
├── make_10_more_submissions.py      # 10 alternative aggregations (F–O)
├── make_focused_submissions.py      # 5 focused weight perturbations (P–T)
├── train.csv                        # 2,494 labeled papers
├── public_test.csv                  # 298 public-test rows
├── private_test.csv                 # 298 private-test rows
├── outputs/                         # base-model OOF + test probabilities
│   ├── meta.json
│   ├── allenai_scibert_scivocab_uncased_probs.npz
│   ├── allenai_specter2_base_probs.npz
│   ├── intfloat_e5-large-v2_probs.npz
│   └── microsoft_deberta-v3-large_probs.npz
└── submissions/
    ├── full/                        # combined pub + priv (Kaggle expects this)
    ├── public/                      # public-test rows only
    └── private/                     # private-test rows only
```

---

## Pipeline overview

The pipeline has two stages:

1. **Stage 1 — Base model training (GPU)** — `kaggle_corn_v2.py`
   Fine-tunes four pre-trained transformer encoders with a **CORN ordinal
   head**, **5-fold stratified CV**, and **3 seeds per fold**, then writes
   out-of-fold (OOF) and test probabilities to `outputs/*.npz`.

2. **Stage 2 — Ensembling (CPU, local)** — `make_*_submissions.py`
   Reads the NPZ files and produces multiple submission CSVs using various
   aggregation strategies (equal averaging, weighted averaging, LightGBM
   stacking, geometric mean, rank averaging, etc.).

The two stages are decoupled so you only need to pay for the GPU once.

---

## 1. Hardware & environment

### Minimum requirements

| Component       | Recommended                                                              |
|-----------------|---------------------------------------------------------------------------|
| GPU             | 1× RTX 3090 / 4090 / A5000 (≥16 GB VRAM)                                  |
| RAM             | ≥16 GB                                                                    |
| Disk            | ≥40 GB (HF cache for 4 backbones ≈12 GB)                                  |
| Docker image    | `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime` (Vast template works)     |
| CUDA            | 12.x                                                                      |
| Python          | 3.10+ (tested on 3.14)                                                    |

### Time and cost (full run: 4 models × 5 folds × 3 seeds × CORN)

| GPU              | Wall-clock | $/hr (approx.) | Total $   |
|------------------|------------|-----------------|-----------|
| RTX 3090 (24 GB) | ~5–7 h     | $0.20–0.30      | $1.5–2.0  |
| RTX 4090 (24 GB) | ~3.5–4 h   | $0.40–0.55      | $1.5–2.0  |
| A100 40 GB       | ~2.5–3 h   | $0.80–1.20      | $2–3.5    |

To go faster, set `--seeds 1` (3× speed-up, ~–0.005 to –0.01 OOF QWK).

### Renting a Vast.ai instance

1. Sign up at [vast.ai](https://vast.ai) and add credit.
2. Search for `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime` or any
   "PyTorch (cuDNN Devel)" template.
3. Filter: GPU = RTX 3090/4090, VRAM ≥ 16 GB, Disk ≥ 40 GB, Net ≥ 100 Mbps.
4. Click **Rent** and wait for the instance to enter the *Running* state.

---

## 2. Stage 1 — Train base models (GPU)

SSH into the rented VM (Vast UI shows the exact command), then:

```bash
# Clone
cd /workspace
git clone https://github.com/NhoXanh1807/PaperClassi.git
cd PaperClassi

# Verify GPU
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Install deps (the PyTorch image already ships torch — do NOT reinstall torch)
pip install --no-cache-dir -r requirements.txt

# (Optional) Hugging Face token if a model is gated
export HF_ACCESS_TOKEN=hf_xxxxxxxxxxxxxxxxx

# (Optional) Move HF cache to a larger disk
export HF_HOME=/workspace/hf_cache
mkdir -p $HF_HOME

# Train inside tmux so the run survives SSH drops
tmux new -s train
python kaggle_corn_v2.py --data-dir . --out-dir ./outputs 2>&1 | tee train.log
```

Detach tmux with `Ctrl+B` then `D`. Re-attach with `tmux a -t train`.

### Useful flags

```bash
# Quick smoke test (2 folds, 1 seed, 1 epoch)
python kaggle_corn_v2.py --models scibert --folds 2 --seeds 1 \
       --epochs-small 1 --epochs-large 1

# Train only specific models (skip ones that already have NPZ)
python kaggle_corn_v2.py --models specter2 e5-large

# Retrain a single model with updated hyperparameters (auto-backups old NPZ)
python kaggle_corn_v2.py --models deberta-v3 --overwrite \
       --epochs-large 12 --amp-dtype bf16

# Lower-VRAM GPU
python kaggle_corn_v2.py --batch-large 4

# All flags
python kaggle_corn_v2.py --help
```

### Copy results back to your machine

When training finishes, copy `outputs/` back to your local repository:

```bash
# Run this on your local machine, NOT on the VM
scp -P <port> -r root@<host>:/workspace/PaperClassi/outputs \
    /path/to/local/PaperClassi/
```

**Destroy the Vast instance immediately afterwards** to stop billing.

---

## 3. Stage 2 — Generate submissions (local CPU)

Each of the three scripts loads `outputs/*_probs.npz` and writes
submission CSVs to `submissions/{full,public,private}/`. They are
independent and can be run in any order.

```bash
# 5 baseline ensembles (A–E):
#   A) equal_avg_all          F) – J)
#   B) equal_avg_strong       (drops weak models below OOF cutoff)
#   C) qwk_weighted_topN
#   D) lgbm_stack_all
#   E) lgbm_stack_strong
python make_5_submissions.py

# 10 alternative aggregations (F–O):
#   geometric mean, rank averaging, LightGBM classification,
#   LightGBM-no-probs, single-model standalone, manual weighting,
#   argmax without thresholds, light DeBERTa infusion, softmax sharpening,
#   arithmetic+geometric blend
python make_10_more_submissions.py

# 5 focused weight perturbations (P–T):
#   drop-one-model probes, gentle weight tweaks, ordinal median voting
python make_focused_submissions.py
```

Each script prints the OOF QWK per submission and the predicted-label
distribution. The **`full/`** CSVs are what you upload to Kaggle.

### Picking a submission

| Goal                                 | Recommended submission                       |
|--------------------------------------|----------------------------------------------|
| Best public LB on this dataset       | `submission_A_equal_avg_all4.csv` (LB 0.72727) |
| Best OOF QWK                         | `submission_B_equal_avg_strong.csv` (OOF 0.6733) |
| Diversity probe                      | `submission_F_geomean_all.csv`                 |
| Single best model                    | `submission_J_specter2_only.csv`               |

See `submissions/full/` for the full list.

---

## 4. Models and methodology

### Base models (Stage 1)

| Key             | HF identifier                          | Params | OOF QWK |
|-----------------|----------------------------------------|--------|---------|
| `scibert`       | `allenai/scibert_scivocab_uncased`     | 110 M  | 0.6476  |
| `specter2`      | `allenai/specter2_base`                | 110 M  | 0.6630  |
| `deberta-v3`    | `microsoft/deberta-v3-large`           | 380 M  | 0.5824  |
| `e5-large`      | `intfloat/e5-large-v2`                 | 355 M  | 0.6601  |

Additional preset slots are wired up for `deberta-v3-base`, `specter-v1`,
`mpnet-base`, and `roberta-base` if you want to extend the ensemble.

### Training recipe

- **Loss:** CORN ordinal loss (Shi et al., 2023) — decomposes the 5-class
  problem into 4 conditional binary tasks, directly aligning the
  objective with QWK.
- **CV:** 5-fold stratified, fixed seed (`42`); 3 training seeds per fold
  averaged.
- **Optimizer:** AdamW with `eps=1e-6` (prevents fp16 NaN with
  DeBERTa-v3-large), weight-decay `0.01`, gradient clipping `1.0`.
- **LR schedule:** Layer-wise LR decay (`0.9`) on the backbone, head LR
  10× higher; linear warm-up for the first 10 % of steps then linear
  decay.
- **AMP:** `bfloat16` autocast on Ampere+ GPUs, `fp16` with GradScaler
  otherwise.
- **Early stop:** patience 3 on validation QWK.

### Threshold optimization

CORN logits are mapped to a 5-class probability vector; the expected
score `E[y] = Σ k·p_k` is fitted against four ordering thresholds
optimised with Nelder–Mead on the OOF set to maximise QWK.

### Ensembling

We tried ~20 distinct strategies (`make_*_submissions.py`). The
**equal-weight arithmetic mean of all four base models** scored
**LB 0.72727**, beating every weighted and stacked alternative — the
diversity supplied by the (weak) DeBERTa-v3-large component proved
essential despite its low OOF score.

---

## 5. Troubleshooting

| Symptom                                          | Fix                                                                  |
|--------------------------------------------------|----------------------------------------------------------------------|
| `CUDA out of memory`                             | `--batch-large 4` or `--max-len 192`                                  |
| `ModuleNotFoundError: sentencepiece`             | `pip install -r requirements.txt`                                     |
| HF download slow/fails                           | Set `HF_HOME` to a large disk; retry; for gated models pass `HF_ACCESS_TOKEN` |
| `FileNotFoundError: train.csv`                   | Pass `--data-dir /path/to/folder` containing the CSVs                 |
| SSH drops mid-training                           | Always run inside `tmux` or `nohup`                                   |
| Training resumes from start instead of skipping  | Default behaviour skips models whose NPZ exists; pass `--overwrite` to retrain |
| Re-train one model only                          | `--models <key> --overwrite` (old NPZ backed up to `*.npz.bak`)       |
| `val_QWK = 0` at epoch 1 (DeBERTa)              | Already fixed by default (`AdamW eps=1e-6`, bf16). Fallback: `--amp-dtype fp32` |
| Python 3.14 multiprocessing pickle error         | `--num-workers 0` (default already 0)                                 |

---

## References

- Shi, X., Cao, W., & Raschka, S. (2023). *Deep Neural Networks for
  Rank-Consistent Ordinal Regression Based on Conditional Probabilities*.
  Pattern Analysis and Applications.
- Beltagy, I., Lo, K., & Cohan, A. (2019). *SciBERT: A Pretrained
  Language Model for Scientific Text*. EMNLP.
- Cohan, A., Feldman, S., Beltagy, I., Downey, D., & Weld, D. (2020).
  *SPECTER: Document-Level Representation Learning Using Citation-Informed
  Transformers*. ACL.
- He, P., Gao, J., & Chen, W. (2021). *DeBERTaV3: Improving DeBERTa Using
  ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding
  Sharing*. arXiv:2111.09543.
- Wang, L., Yang, N., Huang, X., et al. (2024). *Text Embeddings by
  Weakly-Supervised Contrastive Pre-training* (E5).
