# Paper Classification — v2 Pipeline (CORN ordinal + stacking)

Mục tiêu: đẩy QWK ensemble từ **0.70 → cao hơn** bằng cách:

1. **Thay CrossEntropy bằng CORN ordinal loss** (Shi et al. 2023) — tối ưu trực tiếp cho ordinal metric như QWK.
2. **Thêm base models mới**: DeBERTa-v3-large, E5-large-v2 ngoài SPECTER2 + SciBERT.
3. **Multi-seed per fold** (3 seeds) — giảm variance.
4. **Layer-wise LR decay (LLRD)** — fine-tune ổn định model lớn.
5. **Stacking với LightGBM + meta features** (year, venue, num_authors, title length, has_doi) thay cho weighted average.

## Cấu trúc file

| File | Chạy ở đâu | Mục đích |
|---|---|---|
| `kaggle_corn_v2.py` | **Kaggle (GPU T4/P100)** | Train tất cả base models, lưu OOF + test probs |
| `solve_stacking_v2.py` | Local | Stacking + threshold optimization → submission cuối |

## Bước 1 — Chạy trên Kaggle

1. Tạo Kaggle dataset chứa 3 file: `train.csv`, `public_test.csv`, `private_test.csv`. Đặt tên slug là `paper-classi`.
2. Tạo notebook mới, bật **GPU T4 x2** (hoặc P100). Internet **ON** (để tải HF models).
3. Add dataset `paper-classi` vào notebook (sẽ mount tại `/kaggle/input/paper-classi/`).
4. Copy toàn bộ `kaggle_corn_v2.py` vào cell đầu, Run All.
5. Thời gian dự kiến:
   - SciBERT + SPECTER2: ~30 phút/model (5 fold × 3 seed × 10 ep)
   - DeBERTa-v3-large + E5-large-v2: ~90 phút/model (5 fold × 3 seed × 6 ep)
   - **Tổng: ~4-5 tiếng** cho full config

   Nếu cần nhanh hơn, giảm `SEEDS_PER_MODEL = 1` (mất ~0.005 QWK nhưng nhanh gấp 3).

6. Output trong `/kaggle/working/`:
   - `allenai_scibert_scivocab_uncased_probs.npz`
   - `allenai_specter2_base_probs.npz`
   - `microsoft_deberta-v3-large_probs.npz`
   - `intfloat_e5-large-v2_probs.npz`
   - `meta.json` (tóm tắt OOF QWK mỗi model)
   - `submission_corn_avg.csv` (baseline equal-average — dùng làm fallback)

7. Download tất cả `*_probs.npz` về `/Users/quangnguyen/Desktop/paper-classi/`.

## Bước 2 — Stacking local

```bash
cd /Users/quangnguyen/Desktop/paper-classi
pip install lightgbm scipy scikit-learn pandas
python3 solve_stacking_v2.py
```

Script sẽ:
- Đọc mọi `*_probs.npz` trong folder (kể cả các file cũ `specter2_probs.npz`, `scibert_probs.npz` từ pipeline v1)
- Build feature matrix: probs từ mọi model + expected value + meta features
- Train LGBM 5-fold regression → predict expected score
- Optimize 4 threshold cho QWK trên OOF
- In ra OOF QWK + confusion matrix
- Ghi `submission_stacking_v2.csv` + public/private

## Kỳ vọng cải thiện

| Thay đổi | Lift kỳ vọng (cộng dồn) |
|---|---|
| CORN ordinal loss thay CE | +0.01 ~ 0.03 |
| Thêm DeBERTa-v3-large vào ensemble | +0.01 ~ 0.02 |
| Thêm E5-large-v2 | +0.005 ~ 0.015 |
| Multi-seed averaging (3 seeds) | +0.005 ~ 0.01 |
| LLRD (ổn định fine-tune model lớn) | +0.005 |
| Stacking LGBM + meta features | +0.01 ~ 0.02 |
| **Tổng** | **+0.04 ~ 0.10** |

Mức 0.70 → kỳ vọng **0.74 ~ 0.78** sau full pipeline. Nếu chỉ thêm CORN + stacking (không thêm model mới) thì khoảng **+0.02 ~ 0.04**.

## Tinh chỉnh thêm (nếu muốn đẩy tiếp)

- **Pseudo-labeling**: lấy test rows có max prob > 0.9 từ stacking, gộp vào train, train lại 1-2 model — thường +0.005~0.01.
- **MLM continued pretraining**: pretrain DeBERTa-v3 thêm vài epoch trên toàn bộ title+venue (train+test) trước khi fine-tune — +0.005~0.015.
- **Threshold optimization per-fold rồi avg thay vì global** — đôi khi giảm overfit.
- **Test-Time Augmentation**: tokenize test với nhiều max_len khác nhau, average probs.

Báo lại OOF QWK sau khi chạy Bước 1 nếu muốn tinh chỉnh thêm.
