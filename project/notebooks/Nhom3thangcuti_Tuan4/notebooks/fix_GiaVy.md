# YÊU CẦU SỬA FILE: Tuan4_GiaVy_Dinov3+miniML.ipynb

## BỐI CẢNH

File này dùng DINOv3 + MiniLM để truy xuất ảnh sản phẩm Shopee.
Pipeline: DINOv3 (ảnh) + MiniLM (text) → Weighted Fusion → pHash Boost → FAISS → Metric
Cần sửa 5 vấn đề để đạt yêu cầu tuần 4.

---

## SỬA 1: Đổi MiniLM tiếng Anh sang Multilingual

**Tìm dòng này (xuất hiện 2 lần trong file):**

```python
minilm_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
```

**Thay cả 2 chỗ bằng:**

```python
minilm_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)
```

**Tìm dòng lưu file text features:**

```python
np.save(os.path.join(output_dir, "minilm_text_features.npy"), text_features)
```

**Thay bằng:**

```python
np.save(os.path.join(output_dir, "multilingual_minilm_text_features.npy"), text_features)
```

**Tìm dòng load text features:**

```python
minilm_text_features = np.load(os.path.join(output_dir, "minilm_text_features.npy")).astype('float32')
```

**Thay bằng:**

```python
minilm_text_features = np.load(os.path.join(output_dir, "multilingual_minilm_text_features.npy")).astype('float32')
```

**Lý do:** `all-MiniLM-L6-v2` chỉ hỗ trợ tiếng Anh. Dataset Shopee có tiêu đề tiếng Indonesia, Thái, Việt → `paraphrase-multilingual-MiniLM-L12-v2` hỗ trợ 50+ ngôn ngữ, phù hợp hơn nhiều.

---

## SỬA 2: Thêm chia tập Validation / Test

**Thêm cell mới TRƯỚC cell tính metric (cell có vòng lặp `for i in tqdm(range(num_samples))`).**

Nội dung cell mới:

```python
# ============================================================
# CHIA TẬP VALIDATION / TEST
# Validation (20%): dùng để tuning alpha, pHash threshold
# Test (80%): CHỈ dùng để báo kết quả cuối — không tune!
# ============================================================
from sklearn.model_selection import train_test_split

print('=' * 55)
print('CHIA TẬP VALIDATION / TEST')
print('=' * 55)

# Lọc bỏ nhóm chỉ có 1 ảnh (không đánh giá được)
label_counts_all = candidate_df['label_group'].value_counts()
valid_mask = candidate_df['label_group'].isin(
    label_counts_all[label_counts_all >= 2].index
)
valid_indices = candidate_df[valid_mask].index.tolist()

# Chia val/test — không dùng stratify vì nhiều nhóm chỉ có 2 ảnh
val_idx, test_idx = train_test_split(
    valid_indices,
    test_size=0.8,
    random_state=42
)

print(f'Gallery (toàn bộ)  : {len(candidate_df):,} ảnh')
print(f'Validation set     : {len(val_idx):,} ảnh (20%) → dùng để tuning')
print(f'Test set           : {len(test_idx):,} ảnh (80%) → báo kết quả cuối')

# Lưu lại để tái sử dụng
import json
split_info = {'val_idx': val_idx, 'test_idx': test_idx}
with open(os.path.join(output_dir, 'split_indices.json'), 'w') as f:
    json.dump(split_info, f)
print('Đã lưu split_indices.json!')
```

---

## SỬA 3: Thêm Grid Search Alpha trên Validation Set

**Thêm cell mới SAU cell chia tập, TRƯỚC cell tính metric cuối cùng.**

Nội dung cell mới:

```python
# ============================================================
# GRID SEARCH ALPHA TRÊN VALIDATION SET
# ============================================================
print('=' * 55)
print('GRID SEARCH ALPHA (VAL SET)')
print('=' * 55)

alphas = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
best_alpha    = 0.7
best_val_map  = -1.0

image_tensor_val = torch.tensor(
    dinov3_image_features[val_idx]).to(device)
text_tensor_val  = torch.tensor(
    minilm_text_features[val_idx]).to(device)
image_norm_val   = image_tensor_val / image_tensor_val.norm(dim=-1, keepdim=True)
text_norm_val    = text_tensor_val  / text_tensor_val.norm(dim=-1, keepdim=True)

image_norm_all = torch.tensor(dinov3_image_features).to(device)
text_norm_all  = torch.tensor(minilm_text_features).to(device)
image_norm_all = image_norm_all / image_norm_all.norm(dim=-1, keepdim=True)
text_norm_all  = text_norm_all  / text_norm_all.norm(dim=-1, keepdim=True)

phash_strings = candidate_df["image_phash"].values
phash_ints    = np.array([int(h, 16) for h in phash_strings], dtype=np.uint64)
labels_all    = candidate_df["label_group"].values

for alpha in alphas:
    ap_scores = []
    for q_pos, i in enumerate(val_idx):
        query_label = labels_all[i]
        gt_indices  = np.where(labels_all == query_label)[0]
        gt_indices  = gt_indices[gt_indices != i]
        if len(gt_indices) == 0:
            continue

        # Tính fusion score
        img_sim = torch.matmul(
            image_norm_val[q_pos].unsqueeze(0),
            image_norm_all.T).squeeze().cpu().numpy()
        txt_sim = torch.matmul(
            text_norm_val[q_pos].unsqueeze(0),
            text_norm_all.T).squeeze().cpu().numpy()
        sim = alpha * img_sim + (1 - alpha) * txt_sim

        # pHash boost
        q_phash = phash_ints[i]
        x = q_phash ^ phash_ints
        x = (x & 0x5555555555555555) + ((x >> 1) & 0x5555555555555555)
        x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
        x = (x & 0x0F0F0F0F0F0F0F0F) + ((x >> 4) & 0x0F0F0F0F0F0F0F0F)
        x = (x & 0x00FF00FF00FF00FF) + ((x >> 8) & 0x00FF00FF00FF00FF)
        x = (x & 0x0000FFFF0000FFFF) + ((x >> 16) & 0x0000FFFF0000FFFF)
        ham = ((x & 0x00000000FFFFFFFF) + (x >> 32)).astype(np.uint8)
        sim[ham <= 2] += 0.5
        sim[ham == 0] += 0.5
        sim[i]         = -999  # loại self

        top5 = np.argsort(-sim)[:5]
        is_rel = np.isin(top5, gt_indices)
        hits = np.where(is_rel)[0] + 1
        if len(hits) > 0:
            ap = np.sum(np.arange(1, len(hits)+1) / hits) / min(5, len(gt_indices))
        else:
            ap = 0.0
        ap_scores.append(ap)

    val_map = round(float(np.mean(ap_scores)), 4)
    print(f'  alpha={alpha:.2f} → val mAP@5 = {val_map:.4f}')

    if val_map > best_val_map:
        best_val_map  = val_map
        best_alpha    = alpha

print(f'\n🏆 BEST alpha = {best_alpha}, val mAP@5 = {best_val_map:.4f}')
print(f'Dùng alpha = {best_alpha} để đánh giá trên Test Set')
```

---

## SỬA 4: Sửa Cell Tính Metric — Dùng Test Set + Sửa Lỗi gt_len=0

**Tìm cell tính metric có vòng lặp:**

```python
for i in tqdm(range(num_samples)):
    retrieved_indices = all_sorted_indices[i]
    query_label = labels[i]
    ground_truth_indices = group_to_indices[query_label]
```

**Thay toàn bộ cell đó bằng:**

```python
# ============================================================
# TÍNH METRIC TRÊN TEST SET (80%)
# Dùng best_alpha từ grid search ở trên
# ============================================================
import os
import gc
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

csv_path   = "/content/drive/MyDrive/DoAnPython/DuLieuPython/train.csv"
output_dir = "/content/drive/MyDrive/DoAnPython/DuLieuPython"

candidate_df = pd.read_csv(csv_path)
labels       = candidate_df["label_group"].values

dinov3_image_features  = np.load(os.path.join(output_dir, "dinov3_image_features.npy")).astype('float32')
minilm_text_features   = np.load(os.path.join(output_dir, "multilingual_minilm_text_features.npy")).astype('float32')
phash_strings          = candidate_df["image_phash"].values
phash_ints             = np.array([int(h, 16) for h in phash_strings], dtype=np.uint64)

# Load test indices
with open(os.path.join(output_dir, 'split_indices.json'), 'r') as f:
    split_info = json.load(f)
test_idx = split_info['test_idx']

device = "cuda" if torch.cuda.is_available() else "cpu"

image_norm = torch.tensor(dinov3_image_features).to(device)
text_norm  = torch.tensor(minilm_text_features).to(device)
image_norm = image_norm / image_norm.norm(dim=-1, keepdim=True)
text_norm  = text_norm  / text_norm.norm(dim=-1, keepdim=True)

# Dùng best_alpha từ grid search — nếu chưa chạy grid search thì dùng 0.7
ALPHA = best_alpha if 'best_alpha' in dir() else 0.7
print(f'Đánh giá test set với alpha = {ALPHA}')
print(f'Số query test: {len(test_idx):,}')
print('=' * 55)

ap5_scores = []
p1, r1 = [], []
p3, r3 = [], []
p5, r5 = [], []
p10,r10 = [], []

for i in tqdm(test_idx, desc='Tính metric (Test Set)'):
    query_label = labels[i]
    gt_indices  = np.where(labels == query_label)[0]
    gt_indices  = gt_indices[gt_indices != i]  # loại chính nó
    gt_len      = len(gt_indices)

    # ← QUAN TRỌNG: bỏ qua query không có ảnh liên quan
    # KHÔNG dùng append(0) vì sẽ kéo mAP xuống giả tạo
    if gt_len == 0:
        continue

    # Tính fusion score
    img_sim = torch.matmul(
        image_norm[i].unsqueeze(0), image_norm.T
    ).squeeze().cpu().numpy()
    txt_sim = torch.matmul(
        text_norm[i].unsqueeze(0), text_norm.T
    ).squeeze().cpu().numpy()
    sim = ALPHA * img_sim + (1 - ALPHA) * txt_sim

    # pHash boost (giữ nguyên thuật toán bit-twiddling của Hưng)
    q_phash   = phash_ints[i]
    x         = q_phash ^ phash_ints
    x = (x & 0x5555555555555555) + ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x & 0x0F0F0F0F0F0F0F0F) + ((x >> 4) & 0x0F0F0F0F0F0F0F0F)
    x = (x & 0x00FF00FF00FF00FF) + ((x >> 8) & 0x00FF00FF00FF00FF)
    x = (x & 0x0000FFFF0000FFFF) + ((x >> 16) & 0x0000FFFF0000FFFF)
    ham       = ((x & 0x00000000FFFFFFFF) + (x >> 32)).astype(np.uint8)
    sim[ham <= 2] += 0.5
    sim[ham == 0] += 0.5
    sim[i]         = -999  # loại self

    # Lấy top-10
    top10      = np.argsort(-sim)[:10]
    is_rel     = np.isin(top10, gt_indices)

    h1  = is_rel[:1].sum();  p1.append(h1/1);   r1.append(h1/gt_len)
    h3  = is_rel[:3].sum();  p3.append(h3/3);   r3.append(h3/gt_len)
    h5  = is_rel[:5].sum();  p5.append(h5/5);   r5.append(h5/gt_len)
    h10 = is_rel[:10].sum(); p10.append(h10/10); r10.append(h10/gt_len)

    # AP@5
    hits = np.where(is_rel[:5])[0] + 1
    if len(hits) > 0:
        ap = np.sum(np.arange(1, len(hits)+1) / hits) / min(5, gt_len)
    else:
        ap = 0.0
    ap5_scores.append(ap)

# Tổng hợp kết quả
summary_metrics = pd.DataFrame({
    "K"        : [1, 3, 5, 10],
    "Precision": [np.mean(p1), np.mean(p3), np.mean(p5), np.mean(p10)],
    "Recall"   : [np.mean(r1), np.mean(r3), np.mean(r5), np.mean(r10)],
})
summary_metrics[["Precision", "Recall"]] = summary_metrics[["Precision", "Recall"]].round(4)
map_at_5 = round(float(np.mean(ap5_scores)), 4)

print('\n=== KẾT QUẢ CUỐI — DINOv3 + Multilingual MiniLM + pHash ===')
print(f'Tập đánh giá: Test set ({len(test_idx):,} query, 80%)')
print(f'Alpha        : {ALPHA}')
print(summary_metrics.to_string(index=False))
print(f'mAP@5        : {map_at_5}')

# Lưu metric ra CSV để file chung (BTCT_Tuan4_33.ipynb) đọc
summary_metrics['mAP@5']  = map_at_5
summary_metrics['method'] = f'DINOv3 + MultilingualMiniLM + pHash (alpha={ALPHA})'
summary_metrics.to_csv(
    os.path.join(output_dir, 'metrics_dinov3_minilm.csv'), index=False)
print('\nĐã lưu metrics_dinov3_minilm.csv!')
```

---

## SỬA 5: Thêm 2 cell cuối — Ghi chú AI và Kế hoạch tuần 5

**Thêm 2 markdown cell vào cuối notebook:**

**Cell cuối 1:**

```markdown
## GHI CHÚ AI HỖ TRỢ

_(Bắt buộc theo yêu cầu thầy)_

| Phần                    | AI hỗ trợ như thế nào                                             | Người kiểm tra    |
| ----------------------- | ----------------------------------------------------------------- | ----------------- |
| Load DINOv3 (timm)      | Cursor gợi ý dùng `resolve_model_data_config` lấy transform chuẩn | Nguyễn Khánh Hưng |
| Bit-twiddling Hamming   | Cursor gợi ý thuật toán tối ưu không cần vòng lặp                 | Nguyễn Khánh Hưng |
| Grid search alpha       | Claude gợi ý quy trình val/test split                             | Nguyễn Khánh Hưng |
| Sửa gt_len=0            | Claude phát hiện lỗi append(0) → đổi thành continue               | Nguyễn Khánh Hưng |
| Đổi multilingual MiniLM | Claude gợi ý model phù hợp Shopee đa ngôn ngữ                     | Nguyễn Khánh Hưng |
```

**Cell cuối 2:**

```markdown
## KẾ HOẠCH TUẦN 5

| Nội dung                        | Phương pháp                                    | Mục tiêu                  |
| ------------------------------- | ---------------------------------------------- | ------------------------- |
| Fine-tune DINOv3 trên Shopee    | Triplet Loss với hard negative mining          | Tăng mAP so với zero-shot |
| Thử EfficientNet-B4 thay DINOv3 | Nhẹ hơn, dễ fine-tune hơn                      | So sánh với DINOv3        |
| Tối ưu pHash threshold          | Grid search threshold trên val set             | Tìm threshold tốt nhất    |
| Kết hợp DINOv3 + TF-IDF         | Thay MiniLM bằng TF-IDF (đã chứng minh tốt T3) | Có thể cao hơn MiniLM     |
```

---

## KIỂM TRA SAU KHI SỬA

Chạy đoạn này để kiểm tra features không bị lỗi:

```python
print('=== KIỂM TRA FEATURES ===')
print(f'DINOv3 features[0] norm: {np.linalg.norm(dinov3_image_features[0]):.4f}')
print(f'MiniLM features[0] norm: {np.linalg.norm(minilm_text_features[0]):.4f}')
print(f'DINOv3 std: {dinov3_image_features.std():.4f}')
print(f'MiniLM std: {minilm_text_features.std():.4f}')
# Tất cả phải > 0, nếu = 0 thì features bị lỗi
```

---

## KẾT QUẢ MONG ĐỢI SAU KHI SỬA

| Metric      | Trước sửa (toàn bộ) | Sau sửa (test set 80%) |
| ----------- | ------------------- | ---------------------- |
| Precision@1 | 0.7729              | ~0.75–0.80             |
| Recall@5    | 0.7119              | ~0.70–0.75             |
| mAP@5       | 0.7388              | ~0.72–0.78             |

**Lưu ý:** Metric trên test set sẽ hơi thấp hơn toàn bộ — đây là bình thường và đúng hơn về mặt học thuật.

---

## THỨ TỰ CHẠY SAU KHI SỬA

```
Cell 1: pip install
Cell 2: import thư viện
Cell 3: mount Drive + đường dẫn
Cell 4: load CSV + df.info()
Cell 5: load candidate_df
Cell 6: pip install sentence-transformers + load DINOv3 + MiniLM multilingual
Cell 7: unzip train_images
Cell 8: trích xuất DINOv3 + MiniLM features → lưu .npy
Cell 9: load .npy + tính similarity + pHash → lưu all_sorted_indices.npy
Cell MỚI 1: chia val/test split → lưu split_indices.json
Cell MỚI 2: grid search alpha trên val set → tìm best_alpha
Cell CUỐI: tính metric trên test set với best_alpha → lưu metrics_dinov3_minilm.csv
```
