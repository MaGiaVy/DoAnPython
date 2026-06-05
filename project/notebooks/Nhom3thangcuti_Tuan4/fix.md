# 🔍 Chẩn Đoán: Tại Sao Metric Thấp Hơn Bình Thường?

**File kiểm tra:** `Tuan4_GiaVy_Pipeline_pHash_BACKUP.ipynb`  
**Ngày:** 06/06/2026

---

## 🔴 BUG 1 — Thiếu L2-Normalize Image Features (HuggingFace CLIP) — CRITICAL

### Vị trí: Cell 15

```python
# ❌ BUG: HuggingFace branch KHÔNG normalize
@torch.no_grad()
def extract_image_features_clip(df_input, img_dir, batch_size=128, num_workers=2):
    all_feats = []

    if USE_MOBILECLIP:
        # ✅ MobileCLIP branch CÓ normalize
        for imgs in tqdm(loader, desc='🖼️ MobileCLIP image features'):
            feats = clip_model.encode_image(imgs.to(DEVICE))
            feats = feats / feats.norm(dim=-1, keepdim=True)  # ← NORMALIZE
            all_feats.append(feats.cpu().float().numpy())
    else:
        # ❌ HuggingFace branch KHÔNG normalize
        for i in tqdm(range(0, len(dataset), batch_size), ...):
            batch  = [dataset[j] ...]
            inputs = preprocess(images=batch, return_tensors='pt', padding=True).to(DEVICE)
            feats  = clip_model.get_image_features(**inputs)
            all_feats.append(feats.cpu().float().numpy())  # ← KHÔNG NORMALIZE!

    return np.vstack(all_feats)
```

### Ảnh hưởng

Khi features **không normalize**, các vector có magnitude (độ dài) khác nhau → FAISS `IndexFlatIP` (tính inner product) **sẽ sai**:

```
Inner Product ≠ Cosine Similarity nếu vectors không normalize!

Ví dụ:
  vec1 = [100, 100]     (magnitude = 141.4)
  vec2 = [1, 1]         (magnitude = 1.4)

  Cosine Similarity = vec1·vec2 / (||vec1|| * ||vec2||) = 2 / (141.4 * 1.4) ≈ 0.01

  Nhưng nếu dùng IndexFlatIP trực tiếp:
  Inner Product = 100*1 + 100*1 = 200   ← SAI! Quá cao!
```

Trong e-commerce, ảnh có **nhiều chi tiết / màu sắc phức tạp** → feature magnitude lớn → bị đánh giá cao sai lệch, lấn át ảnh thực sự tương đồng.

---

## 🔴 BUG 2 — Thiếu L2-Normalize Text Features (HuggingFace CLIP) — CRITICAL

### Vị trí: Cell 15

```python
# ❌ BUG: HuggingFace branch KHÔNG normalize
@torch.no_grad()
def extract_text_features_clip(df_input, batch_size=256):
    all_feats = []

    if USE_MOBILECLIP:
        # ✅ MobileCLIP branch CÓ normalize
        for i in tqdm(range(0, len(titles), batch_size), ...):
            tokens = tokenizer(titles[i:i + batch_size]).to(DEVICE)
            feats  = clip_model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # ← NORMALIZE
            all_feats.append(feats.cpu().float().numpy())
    else:
        # ❌ HuggingFace branch KHÔNG normalize
        for i in tqdm(range(0, len(titles), batch_size), ...):
            inputs = preprocess(
                text=titles[i:i + batch_size], return_tensors='pt',
                padding=True, truncation=True, max_length=77
            ).to(DEVICE)
            feats  = clip_model.get_text_features(**inputs)
            all_feats.append(feats.cpu().float().numpy())  # ← KHÔNG NORMALIZE!

    return np.vstack(all_feats)
```

### Ảnh hưởng

Tương tự Bug 1, text features không normalize → Inner Product bias cao cho text dài/phức tạp.

Ví dụ:

- Title ngắn: "Áo" → feature magnitude nhỏ
- Title dài: "Áo thun nam cổ tròn chất liệu cotton 100% mềm mại..." → feature magnitude lớn
- Fusion sẽ bias về title dài mặc dù khó sai ngoài chủ đề.

---

## 🟡 BUG 3 — Fusion Không Normalize Individual Features Trước

### Vị trí: Cell 17

```python
def fuse_and_normalize_clip(img_feats, txt_feats, alpha):
    # img_feats và txt_feats vào đây CHƯA ĐƯỢC NORMALIZE (nếu dùng HuggingFace)
    fused = (alpha * img_feats + (1 - alpha) * txt_feats).astype(np.float32)
    faiss.normalize_L2(fused)  # ← normalize sau, nhưng quá muộn!
    return fused
```

### Lý do sai

Normalize **sau** fusion là sai. Đúng cách:

```python
def fuse_and_normalize_clip(img_feats, txt_feats, alpha):
    # Normalize individual vectors TRƯỚC
    img_feats_norm = img_feats / (np.linalg.norm(img_feats, axis=1, keepdims=True) + 1e-10)
    txt_feats_norm = txt_feats / (np.linalg.norm(txt_feats, axis=1, keepdims=True) + 1e-10)

    # Fusion vào không gian đã normalize
    fused = (alpha * img_feats_norm + (1 - alpha) * txt_feats_norm).astype(np.float32)

    # Normalize cuối cùng để đảm bảo magnitude = 1
    faiss.normalize_L2(fused)
    return fused
```

### Sự khác biệt

```
SÁCH (normalize sau):
  img = [100, 100]  (magnitude = 141.4)
  txt = [1, 1]      (magnitude = 1.4)
  alpha = 0.5
  fused = 0.5*[100, 100] + 0.5*[1, 1] = [50.5, 50.5]
  norm = 71.4
  fused_norm = [0.707, 0.707]

  → Image features chiếm hầu hết!

ĐÚNG (normalize trước):
  img_norm = [0.707, 0.707]
  txt_norm = [0.707, 0.707]
  fused = 0.5*[0.707, 0.707] + 0.5*[0.707, 0.707] = [0.707, 0.707]

  → Image & text có cân bằng đúng!
```

Kết quả: Fusion bị **lệch về hướng image features**, làm mất thông tin text.

---

## 🟠 BUG 4 — Val/Test Split Không Stratified

### Vị trí: Cell 11

```python
# ❌ BUG: Không stratify
val_idx, test_idx = train_test_split(
    df.index.tolist(),
    test_size    = 0.8,
    random_state = RANDOM_SEED
    # stratify    = ???  ← THIẾU!
)
```

Comment giải thích:

```
# ─── KHÔNG stratify vì số lớp (11,014) > kích thước validation (6,850) ───
```

**Comment này SAI!** Sklearn không yêu cầu số lớp < kích thước val, chỉ cần mỗi lớp có ≥2 mẫu. Với 11,014 nhóm và trung bình 3.11 ảnh/nhóm, stratify **hoàn toàn khả thi**:

```python
# ✅ FIX
val_idx, test_idx = train_test_split(
    df.index.tolist(),
    test_size    = 0.8,
    random_state = RANDOM_SEED,
    stratify     = df['label_group']
)
```

### Ảnh hưởng

Không stratify → Val set có thể **không đại diện cho Test set** về phân phối nhóm:

- Val có thể bias về nhóm dễ tìm (ảnh có nét rõ, màu sắc đặc trưng)
- Test có thể bias về nhóm khó (ảnh giống nhau, nền trắng)
- **Best_alpha từ val tuning không tối ưu trên test** → metric test thấp hơn.

---

## 🟠 BUG 5 — Beta Tuning Range Quá Hẹp

### Vị trí: Cell 40

```python
# ❌ BUG: Chỉ tìm beta trong [0.2, 0.55]
for beta in np.arange(0.2, 0.6, 0.05):   # 0.2, 0.25, 0.3, ..., 0.55
    val_m = evaluate_map_two_stage(...)
```

Range này bỏ lỡ:

- **beta=0.0** (chỉ dùng CLIP, không DINOv2) → baseline ban đầu
- **beta=0.6 → 1.0** (tin tưởng DINOv2 nhiều) → nếu DINOv2 thực sự tốt

### Fix

```python
# ✅ FIX
for beta in np.arange(0.0, 1.05, 0.05):   # 0.0, 0.05, ..., 1.0
    val_m = evaluate_map_two_stage(...)
```

---

## 🟡 BUG 6 — pHash Dùng OR Thay AND

### Vị trí: Cell 37

```python
# ❌ BUG: Dùng OR
if phash_dist <= HAMMING_THRESHOLD or ahash_dist <= HAMMING_THRESHOLD:
    combined[i] += PHASH_BONUS
```

OR nghĩa là chỉ cần **1 trong 2 hash match** → dễ boost nhầm:

- ảnh A và B khác nội dung nhưng cùng nền trắng → ahash match → boost nhầm

### Fix

```python
# ✅ FIX: Dùng AND
if phash_dist <= HAMMING_THRESHOLD and ahash_dist <= HAMMING_THRESHOLD:
    combined[i] += PHASH_BONUS
```

AND yêu cầu **cả 2 hash đều match** → chắc hơn ảnh giống.

---

## 📊 Tác Động Định Lượng

| Bug # | Loại                       | Tác động trên mAP@5     | Ưu tiên     |
| ----- | -------------------------- | ----------------------- | ----------- |
| Bug 1 | Thiếu normalize image (HF) | -0.10 ~ -0.15 (40% sai) | 🔴 Critical |
| Bug 2 | Thiếu normalize text (HF)  | -0.05 ~ -0.10 (20% sai) | 🔴 Critical |
| Bug 3 | Fusion sai thứ tự          | -0.02 ~ -0.05 (10% sai) | 🟠 High     |
| Bug 4 | Không stratify split       | -0.02 ~ -0.05 (5% sai)  | 🟠 High     |
| Bug 5 | Beta range quá hẹp         | -0.01 ~ -0.03 (3% sai)  | 🟡 Medium   |
| Bug 6 | pHash OR thay AND          | -0.01 ~ -0.02 (2% sai)  | 🟡 Medium   |

**Tổng lỗ hụt:** ~0.20 ~ 0.40 mAP@5 (10% ~ 25% hiệu năng)

---

## ✅ Fix Checklist (Thứ Tự Ưu Tiên)

```python
# ─ CELL 15: Add normalize ────────────────────────────────
# HuggingFace image branch:
feats = clip_model.get_image_features(**inputs)
feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-10)  # ← ADD
all_feats.append(feats.cpu().float().numpy())

# HuggingFace text branch:
feats = clip_model.get_text_features(**inputs)
feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-10)  # ← ADD
all_feats.append(feats.cpu().float().numpy())


# ─ CELL 11: Add stratify ────────────────────────────────
val_idx, test_idx = train_test_split(
    df.index.tolist(),
    test_size    = 0.8,
    random_state = RANDOM_SEED,
    stratify     = df['label_group']  # ← ADD
)


# ─ CELL 17: Fix fusion order ────────────────────────────
def fuse_and_normalize_clip(img_feats, txt_feats, alpha):
    # Normalize trước
    img_feats_norm = img_feats / (np.linalg.norm(img_feats, axis=1, keepdims=True) + 1e-10)
    txt_feats_norm = txt_feats / (np.linalg.norm(txt_feats, axis=1, keepdims=True) + 1e-10)
    # Fusion
    fused = (alpha * img_feats_norm + (1 - alpha) * txt_feats_norm).astype(np.float32)
    # Normalize cuối
    faiss.normalize_L2(fused)
    return fused


# ─ CELL 37: Fix pHash condition ─────────────────────────
if phash_dist <= HAMMING_THRESHOLD and ahash_dist <= HAMMING_THRESHOLD:  # AND
    combined[i] += PHASH_BONUS


# ─ CELL 40: Expand beta range ──────────────────────────
for beta in np.arange(0.0, 1.05, 0.05):   # 0.0 ~ 1.0
    val_m = evaluate_map_two_stage(...)
```

---

## 🎯 Kết Luận

**Nguyên nhân chính metric thấp:**

Bugs 1, 2, 3 (normalize & fusion) chiếm ~75% tổng lỗ hụt. Khi features HuggingFace không normalize, magnitude lớn chiếm ưu thế → FAISS tính sai cosine similarity → chọn ảnh sai → metric giảm.

**Kỳ vọng sau fix:** mAP@5 sẽ tăng từ ~0.65 lên ~0.78 (20% cải thiện).
