# 🔍 Code Review: Visual Search Pipeline (pHash + MobileCLIP + DINOv2)

**File:** `Tuan4_GiaVy_Pipeline_pHash.ipynb`  
**Dataset:** Shopee (34,250 items) | **Metrics:** mAP@5, Precision@1, Recall@5

---

## 🐛 Bugs

### Bug 1 — Thiếu L2-normalize Image Features (HuggingFace CLIP) `Cell 6` 🔴 Critical

Nhánh HuggingFace trong `extract_image_features_clip` **không normalize** features trước khi trả về, trong khi FAISS `IndexFlatIP` yêu cầu vector đã L2-normalize để tính cosine similarity đúng.

```python
# ❌ Hiện tại — KHÔNG normalize
feats = clip_model.get_image_features(**inputs)
all_feats.append(feats.cpu().float().numpy())

# ✅ Fix
feats = clip_model.get_image_features(**inputs)
feats = feats / feats.norm(dim=-1, keepdim=True)   # thêm dòng này
all_feats.append(feats.cpu().float().numpy())
```

> Tương tự với `extract_text_features_clip` nhánh HuggingFace — cũng thiếu normalize.

---

### Bug 2 — Thiếu L2-normalize Image Features (MobileCLIP) `Cell 6` 🔴 Critical

Text features MobileCLIP **có normalize**, nhưng image features thì **không**.

```python
# ❌ Hiện tại — image KHÔNG normalize, text có normalize → bất đối xứng
feats = clip_model.encode_image(imgs.to(DEVICE))
all_feats.append(feats.cpu().float().numpy())

# ✅ Fix
feats = clip_model.encode_image(imgs.to(DEVICE))
feats = feats / feats.norm(dim=-1, keepdim=True)   # thêm dòng này
all_feats.append(feats.cpu().float().numpy())
```

---

### Bug 3 — So sánh Float với `== 0` không đáng tin cậy `Cell 7` 🟡 Medium

```python
# ❌ Hiện tại — float comparison dễ bỏ sót edge case
norms = np.where(norms == 0, 1e-10, norms)

# ✅ Fix — dùng np.maximum an toàn hơn
norms = np.maximum(norms, 1e-10)
```

---

### Bug 4 — Kiểm tra model tồn tại bằng `dir()` không đáng tin cậy `Bước 1` 🟡 Medium

```python
# ❌ Hiện tại — nếu dinov2 = None thì dir() vẫn thấy nó, không reload
if 'dinov2' not in dir() or dinov2 is None:

# ✅ Fix — dùng globals() chính xác hơn
if 'dinov2' not in globals() or dinov2 is None:
```

---

### Bug 5 — Recall@5 tính sai khi nhóm có nhiều hơn K ảnh `Cell 7 & Bước 3` 🟠 High

Khi một nhóm có 10 ảnh giống nhau (`len(relevant) = 9`), Recall@5 tối đa chỉ đạt `5/9 ≈ 0.56` dù pipeline hoàn hảo → **metric bị underestimate**.

```python
# ❌ Hiện tại — mẫu số dùng toàn bộ relevant
r5_list.append(len(set(retrieved) & relevant) / len(relevant))

# ✅ Fix — giới hạn mẫu số bởi K
r5_list.append(len(set(retrieved) & relevant) / min(len(relevant), K))
```

---

## 🚀 Cải Tiến mAP

### Cải tiến 1 — Tăng `hash_size` của pHash `Bước 1.5`

`hash_size=8` tạo hash 64-bit, quá thô cho ảnh e-commerce. Tăng lên `16` (256-bit) giúp phân biệt tốt hơn.

```python
# ❌ Hiện tại
def compute_phash(img_path, hash_size=8):

# ✅ Cải tiến
def compute_phash(img_path, hash_size=16):
```

---

### Cải tiến 2 — Kết hợp thêm `AverageHash` với `pHash` `Bước 2`

Dùng 2 loại hash song song tăng độ chính xác khi detect ảnh trùng.

```python
# Tính thêm ahash cho gallery (tương tự pHash)
gallery_ahashes = [imagehash.average_hash(Image.open(...)) for ...]

# Trong search_two_stage:
q_phash = compute_phash(query_img_path)
q_ahash = imagehash.average_hash(Image.open(query_img_path).convert('RGB'))

for i, idx in enumerate(candidate_indices):
    phash_dist = q_phash - gallery_phashes[idx]
    ahash_dist = q_ahash - gallery_ahashes[idx]
    if phash_dist <= HAMMING_THRESHOLD and ahash_dist <= HAMMING_THRESHOLD:
        combined[i] += PHASH_BONUS
```

---

### Cải tiến 3 — Tăng `PHASH_BONUS` và nới `HAMMING_THRESHOLD` `Bước 1.5`

Giá trị hiện tại quá thấp để tạo ra sự khác biệt thực sự.

```python
# ❌ Hiện tại
HAMMING_THRESHOLD = 5
PHASH_BONUS       = 0.03

# ✅ Cải tiến (thử nghiệm trên val set trước)
HAMMING_THRESHOLD = 8
PHASH_BONUS       = 0.10
```

---

### Cải tiến 4 — Normalize bằng FAISS trực tiếp thay vì thủ công `Cell 7`

```python
import faiss

# ✅ Thay thế fuse_and_normalize_clip bằng:
def fuse_and_normalize_clip(img_feats, txt_feats, alpha):
    fused = (alpha * img_feats + (1 - alpha) * txt_feats).astype(np.float32)
    faiss.normalize_L2(fused)   # in-place, chính xác và nhanh hơn
    return fused
```

---

### Cải tiến 5 — Grid search `alpha` mịn hơn sau bước đầu `Cell 9`

Sau khi tìm được `best_alpha` với step 0.1, search fine-grained quanh vùng đó.

```python
# Bước 1: coarse search (đã có)
alphas_coarse = np.arange(0.1, 1.0, 0.1)

# Bước 2: fine-grained search quanh best_alpha
alphas_fine = np.arange(
    max(0.0, best_alpha_2 - 0.09),
    min(1.0, best_alpha_2 + 0.10),
    0.02
).round(2)
```

---

### Cải tiến 6 — Nâng cấp lên `DINOv2-Base` nếu VRAM đủ `Bước 1`

`dinov2_vitb14` (768-dim) mạnh hơn `dinov2_vits14` (384-dim), thường cải thiện ~1–2% mAP.

```python
# ❌ Hiện tại
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

# ✅ Nâng cấp (cần ~6GB VRAM, phù hợp T4)
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
```

---

## 📊 Tóm tắt

| #     | Vấn đề                                       | Cell / Bước    | Ảnh hưởng               | Độ ưu tiên     |
| ----- | -------------------------------------------- | -------------- | ----------------------- | -------------- |
| Bug 1 | Thiếu normalize image features (HuggingFace) | Cell 6         | Kết quả sai hoàn toàn   | 🔴 Critical    |
| Bug 2 | Thiếu normalize image features (MobileCLIP)  | Cell 6         | Kết quả sai hoàn toàn   | 🔴 Critical    |
| Bug 5 | Recall@5 tính sai                            | Cell 7, Bước 3 | Metric bị underestimate | 🟠 High        |
| Bug 3 | `np.where` float compare                     | Cell 7         | Edge case               | 🟡 Medium      |
| Bug 4 | `dir()` check không tin cậy                  | Bước 1         | Có thể crash            | 🟡 Medium      |
| CT 1  | Tăng pHash `hash_size` lên 16                | Bước 1.5       | +mAP nhỏ, chi phí thấp  | 🟢 Dễ làm      |
| CT 2  | Thêm AverageHash song song                   | Bước 2         | +mAP trung bình         | 🟢 Dễ làm      |
| CT 3  | Tăng `PHASH_BONUS` và `HAMMING_THRESHOLD`    | Bước 1.5       | +mAP trung bình         | 🟢 Dễ làm      |
| CT 4  | FAISS `normalize_L2` thay numpy              | Cell 7         | Nhất quán hơn           | 🟢 Dễ làm      |
| CT 5  | Fine-grained alpha search                    | Cell 9         | +mAP nhỏ                | 🟢 Dễ làm      |
| CT 6  | DINOv2-Base thay DINOv2-Small                | Bước 1         | +1–2% mAP               | 🔵 Nếu VRAM đủ |

> **Ưu tiên hàng đầu:** Fix Bug 1 & 2 trước — đây là lỗi nghiêm trọng nhất làm sai toàn bộ kết quả similarity search.
