# Paper Classification — CORN ordinal + stacking

Pipeline 2 bước:
1. **`kaggle_corn_v2.py`** — train 4 base models (SciBERT, SPECTER2, DeBERTa-v3-large, E5-large-v2) trên GPU, mỗi model 5-fold × 3 seed, dùng CORN ordinal loss + LLRD. Xuất OOF & test probs ra NPZ.
2. **`solve_stacking_v2.py`** — chạy local, đọc tất cả NPZ, stack bằng LightGBM + meta features, optimize threshold cho QWK, sinh submission cuối.

Bước 1 nặng GPU → thuê máy trên [Vast.ai](https://vast.ai). Bước 2 chạy local trên Mac/PC bình thường.

---

## 1. Cấu hình máy Vast.ai

### Yêu cầu tối thiểu
| Thành phần | Cấu hình khuyến nghị |
|---|---|
| GPU | **1× RTX 3090 / 4090 / A5000 (24 GB VRAM)** hoặc tốt hơn |
| VRAM tối thiểu | 16 GB (đủ cho DeBERTa-v3-large + batch 8) |
| RAM | ≥ 16 GB |
| Disk | ≥ 40 GB (HF cache cho 4 model ~12 GB) |
| Image | **`pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime`** hoặc image Vast `PyTorch (cuDNN Devel)` |
| CUDA | 12.x (12.1 hoặc 12.4 đều OK) |

### Ước tính thời gian & chi phí
| GPU | Thời gian full run (4 model, 5 fold × 3 seed) | $/h tham khảo | Tổng $ |
|---|---|---|---|
| RTX 3090 (24 GB) | ~6-7h | $0.20-0.30 | ~$1.5-2 |
| RTX 4090 (24 GB) | ~3.5-4h | $0.40-0.55 | ~$1.5-2 |
| A100 40 GB | ~2.5-3h | $0.80-1.20 | ~$2-3.5 |

Nếu muốn tiết kiệm, dùng `--seeds 1` (giảm 3× thời gian, mất ~0.005-0.01 QWK).

### Bước chọn máy trên Vast.ai
1. Login [vast.ai](https://vast.ai), nạp tiền vào balance.
2. Vào **Templates** → chọn **PyTorch (cuDNN Devel)** (hoặc Search image `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime`).
3. Vào **Search** → filter:
   - **GPU**: `RTX 3090` hoặc `RTX 4090`
   - **VRAM**: ≥ 16 GB
   - **Disk**: ≥ 40 GB
   - **DLPerf**: cao càng tốt
   - **Inet down**: ≥ 100 Mbps (để pull HF model nhanh)
4. **Rent** instance → đợi 1-2 phút cho instance "Running".

---

## 2. Setup môi trường trên VM

SSH vào máy (Vast UI hiện sẵn lệnh `ssh -p <port> root@<host>`):

```bash
# 1) Clone repo
cd /workspace   # hoặc /root
git clone https://github.com/NhoXanh1807/PaperClassi.git
cd PaperClassi

# 2) Verify GPU
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 3) Install deps (image đã có torch + CUDA → KHÔNG cài torch lại)
pip install --no-cache-dir -r requirements.txt

# 4) (Tuỳ chọn) Set HF token nếu cần model gated
export HF_ACCESS_TOKEN=hf_xxxxxxxxxxxxxxxxx

# 5) (Tuỳ chọn) Chỉnh HF cache sang volume disk lớn để khỏi hết chỗ
export HF_HOME=/workspace/hf_cache
mkdir -p $HF_HOME
```

> **Nếu image KHÔNG sẵn torch** (image base ubuntu trắng), cài torch trước:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

---

## 3. Chạy training

### Chạy full pipeline (mặc định)
```bash
# Nên chạy trong tmux/screen để không mất session khi SSH ngắt
tmux new -s train

python kaggle_corn_v2.py --data-dir . --out-dir ./outputs 2>&1 | tee train.log
```

Detach: `Ctrl+B` rồi `D`. Attach lại: `tmux a -t train`.

### Các option hay dùng

```bash
# Chạy nhanh — chỉ 2 model nhỏ, 1 seed
python kaggle_corn_v2.py --models scibert specter2 --seeds 1

# Skip model đã có NPZ (mặc định) — chỉ chạy model mới
python kaggle_corn_v2.py

# Retrain 1 model cụ thể (tự backup NPZ cũ thành .npz.bak)
python kaggle_corn_v2.py --models deberta-v3 --overwrite \
       --epochs-large 12 --amp-dtype bf16

# GPU yếu / VRAM < 16 GB: giảm batch DeBERTa
python kaggle_corn_v2.py --batch-large 4

# Smoke test (1 fold, 1 seed, 1 epoch nhỏ)
python kaggle_corn_v2.py --folds 2 --seeds 1 --epochs-small 1 --epochs-large 1 \
                        --models scibert
```

### Tất cả flag
```bash
python kaggle_corn_v2.py --help
```

---

## 4. Download kết quả về local

Sau khi train xong, trên VM `./outputs/` sẽ có:

```
outputs/
├── allenai_scibert_scivocab_uncased_probs.npz
├── allenai_specter2_base_probs.npz
├── microsoft_deberta-v3-large_probs.npz
├── intfloat_e5-large-v2_probs.npz
├── meta.json
├── submission_corn_avg.csv         # baseline equal-weight
├── public_submission_corn_avg.csv
└── private_submission_corn_avg.csv
```

Download về Mac (chạy ở **máy local**, không phải VM):

```bash
# Vast UI hiện sẵn host + port SSH
scp -P <port> -r root@<host>:/workspace/PaperClassi/outputs \
    /Users/quangnguyen/Desktop/paper-classi/
```

**Đừng quên destroy instance sau khi xong để khỏi mất tiền** (Vast UI → instance → nút Destroy).

---

## 5. Stacking local

Trên máy của bạn (sau khi đã copy `outputs/*.npz` về `/Users/quangnguyen/Desktop/paper-classi/`):

```bash
cd /Users/quangnguyen/Desktop/paper-classi
pip install lightgbm scipy scikit-learn pandas
python3 solve_stacking_v2.py
```

Output: `submission_stacking_v2.csv` — file submit cuối cùng.

---

## 6. Troubleshooting

| Lỗi | Nguyên nhân & fix |
|---|---|
| `CUDA out of memory` | Giảm `--batch-large 4` hoặc `--batch-small 8`. Hoặc giảm `--max-len 192`. |
| `ModuleNotFoundError: sentencepiece` | Quên cài deps: `pip install -r requirements.txt`. |
| HF model tải chậm/fail | Set `HF_HOME` ra disk lớn; thử lại. Nếu model gated thì cần `HF_ACCESS_TOKEN`. |
| `FileNotFoundError: train.csv` | CSV không ở `--data-dir`. Pass `--data-dir /đường/đến/folder`. |
| SSH disconnect làm mất train | Luôn dùng `tmux` hoặc `nohup`. |
| Train dừng giữa chừng | Mặc định đã skip model có NPZ. Chỉ chạy lại model thiếu. |
| Cần retrain 1 model | `--models <key> --overwrite`. NPZ cũ → `*.npz.bak`. |
| NaN loss / val_QWK=0 ở ep1 (DeBERTa) | Đã fix mặc định (AdamW eps=1e-6, bf16). Nếu vẫn: `--amp-dtype fp32`. |

---

## Cấu trúc repo

```
PaperClassi/
├── kaggle_corn_v2.py          # Script train chính (CHẠY TRÊN VAST)
├── solve_stacking_v2.py       # Stacking local (CHẠY LOCAL)
├── solve.py, solve_*.py       # Pipeline v1 (legacy, optional)
├── train.csv                  # ~3.4K papers training
├── public_test.csv            # Public leaderboard
├── private_test.csv           # Private leaderboard
├── requirements.txt
└── README.md                  # File này
```

---

## Tham khảo thêm

- Chi tiết thuật toán / lift kỳ vọng / tinh chỉnh thêm: xem [`README_v2.md`](README_v2.md).
- CORN ordinal loss: Shi et al. 2023, "Deep Neural Networks for Rank-Consistent Ordinal Regression".
