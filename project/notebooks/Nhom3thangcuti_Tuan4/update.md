# 🔧 Upgrade Guide — `Tuan4_GiaVy_Pipeline1.ipynb`

Tài liệu này liệt kê **4 thay đổi** cần áp dụng theo thứ tự ưu tiên.  
Mỗi mục ghi rõ: cell nào, thay đoạn nào bằng đoạn nào, tại sao.

---

## Fix 1 — Normalize image features (Bug ảnh hưởng metric)

**Cell:** `Cell 6 — Hàm trích xuất Image & Text Features`  
**Vấn đề:** `extract_image_features_clip` trả về raw embeddings chưa normalize, trong khi `extract_text_features_clip` (nhánh MobileCLIP) đã normalize. Khi fusion `α × img + (1-α) × txt`, hai vector có scale khác nhau → alpha grid search tìm ra giá trị lệch, metric không đại diện thực chất.

**Thay hàm `extract_image_features_clip` bằng đoạn sau:**

```python
@torch.no_grad()
def extract_image_features_clip(df_input, img_dir, batch_size=128, num_workers=2):
    all_feats = []

    if USE_MOBILECLIP:
        dataset = ShopeeImageDataset(df_input, img_dir,
                                     transform=preprocess, return_tensor=True)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
        for imgs in tqdm(loader, desc='🖼️ MobileCLIP image features'):
            feats = clip_model.encode_image(imgs.to(DEVICE))
            # ✅ FIX: normalize để scale đồng nhất với text features
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu().float().numpy())
    else:
        dataset = ShopeeImageDataset(df_input, img_dir, return_tensor=False)
        for i in tqdm(range(0, len(dataset), batch_size),
                      desc='🖼️ CLIP HuggingFace image features'):
            batch  = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
            inputs = preprocess(images=batch, return_tensors='pt',
                                padding=True).to(DEVICE)
            feats  = clip_model.get_image_features(**inputs)
            # ✅ FIX: normalize
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu().float().numpy())

    return np.vstack(all_feats)
```

**Đồng thời, thêm normalize cho nhánh HuggingFace của `extract_text_features_clip`:**

```python
# Trong nhánh else (HuggingFace):
        for i in tqdm(range(0, len(titles), batch_size),
                      desc='📝 CLIP HuggingFace text features'):
            inputs = preprocess(
                text=titles[i:i + batch_size], return_tensors='pt',
                padding=True, truncation=True, max_length=77
            ).to(DEVICE)
            feats  = clip_model.get_text_features(**inputs)
            # ✅ FIX: normalize (nhánh MobileCLIP đã có, HuggingFace chưa có)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu().float().numpy())
```

> **Lưu ý quan trọng:** Sau khi fix xong, **phải chạy lại Cell 8** (trích xuất lại toàn bộ features) trước khi grid search. Các features cũ đã bị lưu trong RAM với scale sai.

---

## Fix 2 — Pre-compute DINOv2 query features (Bug performance nghiêm trọng)

**Cell:** `Bước 1` và `Bước 3`  
**Vấn đề:** Hiện tại `search_two_stage` gọi `get_dinov2_embedding(query_img_path)` bên trong vòng loop của mỗi query — tức là **1 forward pass riêng lẻ mỗi query**. Với 6,850 val queries × 4 giá trị `retrieval_k` × 9 giá trị `beta` = ~246,600 forward pass đơn lẻ. Trên T4, mỗi pass đơn lẻ tốn overhead CUDA dispatch → toàn bộ tuning mất hàng tiếng, dễ timeout session.

**Bước 2.1 — Thêm hàm batch extract DINOv2 sau Cell "Bước 1":**

```python
# ── Bổ sung: Extract DINOv2 features theo batch cho tập query ──────────────
@torch.no_grad()
def extract_dinov2_features_batch(df_input, img_dir, batch_size=64):
    """Trích xuất DINOv2 features theo batch — nhanh hơn gọi từng ảnh ~20-50x."""
    paths = [os.path.join(img_dir, fname) for fname in df_input['image']]
    all_feats = []

    for i in tqdm(range(0, len(paths), batch_size), desc='🦕 DINOv2 batch features'):
        batch_imgs = []
        for p in paths[i:i + batch_size]:
            try:
                img = Image.open(p).convert('RGB')
            except Exception:
                img = Image.new('RGB', (224, 224), (128, 128, 128))
            batch_imgs.append(transform_dino(img))

        batch_tensor = torch.stack(batch_imgs).to(DEVICE)
        feats = dinov2.forward_features(batch_tensor)['x_norm_clstoken']
        feats = F.normalize(feats, dim=-1)
        all_feats.append(feats.cpu().float().numpy())

    return np.vstack(all_feats)


# Tính và cache val + test DINOv2 features
VAL_DINO_CACHE  = os.path.join(FEAT_DIR, 'dinov2_val.npy')
TEST_DINO_CACHE = os.path.join(FEAT_DIR, 'dinov2_test.npy')

if not os.path.exists(VAL_DINO_CACHE):
    print('⏳ Đang tính DINOv2 val features...')
    val_dino_feats = extract_dinov2_features_batch(df_val, IMG_DIR)
    np.save(VAL_DINO_CACHE, val_dino_feats)
    print(f'✅ Đã lưu: {VAL_DINO_CACHE}  shape={val_dino_feats.shape}')
else:
    val_dino_feats = np.load(VAL_DINO_CACHE)
    print(f'✅ Loaded val DINOv2 cache: {val_dino_feats.shape}')

if not os.path.exists(TEST_DINO_CACHE):
    print('⏳ Đang tính DINOv2 test features...')
    test_dino_feats = extract_dinov2_features_batch(df_test, IMG_DIR)
    np.save(TEST_DINO_CACHE, test_dino_feats)
    print(f'✅ Đã lưu: {TEST_DINO_CACHE}  shape={test_dino_feats.shape}')
else:
    test_dino_feats = np.load(TEST_DINO_CACHE)
    print(f'✅ Loaded test DINOv2 cache: {test_dino_feats.shape}')
```

**Bước 2.2 — Thay hàm `search_two_stage` để nhận pre-computed DINOv2 vector:**

```python
def search_two_stage(query_img_feat, query_txt_feat, query_dino_feat,
                     alpha, query_idx=None, retrieval_k=100, final_k=5,
                     beta=0.3):
    """
    Pipeline 2 giai đoạn — dùng pre-computed DINOv2 feature thay vì load ảnh từng lần.
    Đã bỏ tham số query_img_path (không cần nữa).
    """
    # ─── Giai đoạn 1: MobileCLIP FAISS ──────────────────────────────────────
    q_fused = fuse_single_clip(query_img_feat, query_txt_feat, alpha)
    raw_indices, raw_scores = search_mobileclip_stage1(q_fused, top_k=retrieval_k + 1)

    candidate_indices, candidate_clip_scores = [], []
    for idx, score in zip(raw_indices, raw_scores):
        if idx != query_idx:
            candidate_indices.append(idx)
            candidate_clip_scores.append(score)
        if len(candidate_indices) == retrieval_k:
            break

    # ─── Giai đoạn 2: DINOv2 Re-ranking (dùng pre-computed vector) ──────────
    gallery_dino_cands = gallery_dino[candidate_indices]          # (retrieval_k, 384)
    dino_scores = (query_dino_feat @ gallery_dino_cands.T).tolist()  # vectorized dot product

    # ─── Fusion điểm ─────────────────────────────────────────────────────────
    combined  = beta * np.array(dino_scores) + (1 - beta) * np.array(candidate_clip_scores)
    new_order = np.argsort(combined)[::-1]
    final_indices = [candidate_indices[i] for i in new_order[:final_k]]
    final_scores  = [combined[i]          for i in new_order[:final_k]]
    return final_indices, final_scores

print('✅ Hàm search_two_stage (vectorized) sẵn sàng!')
```

**Bước 2.3 — Cập nhật `evaluate_map_two_stage` để truyền pre-computed DINOv2:**

```python
def evaluate_map_two_stage(query_df, alpha, retrieval_k, final_k=5,
                           query_img_feats=None, query_txt_feats=None,
                           query_dino_feats=None,   # ← tham số mới
                           beta=0.3):
    gallery_pids     = df_gallery['posting_id'].tolist()
    pid_to_gallery_idx = {pid: idx for idx, pid in enumerate(gallery_pids)}
    gt_dict          = get_ground_truth_dict(df_gallery)
    ap_list, p1_list, r5_list = [], [], []

    for i, row in tqdm(
        enumerate(query_df.itertuples()),
        total=len(query_df),
        desc=f'🔍 Eval 2-stage (k={retrieval_k}, β={beta:.2f})'
    ):
        qid      = row.posting_id
        relevant = gt_dict.get(qid, set()) - {qid}
        if not relevant:
            continue

        q_img_feat  = query_img_feats[i]
        q_txt_feat  = query_txt_feats[i]
        q_dino_feat = query_dino_feats[i]           # ← dùng pre-computed
        query_idx   = pid_to_gallery_idx[qid]

        try:
            final_indices, _ = search_two_stage(
                query_img_feat  = q_img_feat,
                query_txt_feat  = q_txt_feat,
                query_dino_feat = q_dino_feat,      # ← truyền vào
                alpha           = alpha,
                query_idx       = query_idx,
                retrieval_k     = retrieval_k,
                final_k         = final_k,
                beta            = beta,
            )
        except Exception:
            ap_list.append(0.0); p1_list.append(0.0); r5_list.append(0.0)
            continue

        retrieved_pids = [gallery_pids[idx] for idx in final_indices]

        hits, ap = 0, 0.0
        for rank, pid in enumerate(retrieved_pids, 1):
            if pid in relevant:
                hits += 1
                ap   += hits / rank
        ap_list.append(ap / min(len(relevant), final_k))
        p1_list.append(1.0 if (retrieved_pids and retrieved_pids[0] in relevant) else 0.0)
        r5_list.append(len(set(retrieved_pids) & relevant) / len(relevant))

    return {
        'mAP@5'      : float(np.mean(ap_list)),
        'Precision@1': float(np.mean(p1_list)),
        'Recall@5'   : float(np.mean(r5_list)),
    }
```

**Bước 2.4 — Cập nhật các lời gọi tuning (Bước 3) để truyền `query_dino_feats`:**

```python
# Tuning retrieval_k — thêm query_dino_feats=val_dino_feats
val_m = evaluate_map_two_stage(
    query_df        = df_val,
    alpha           = best_alpha_2,
    retrieval_k     = k,
    final_k         = 5,
    query_img_feats = val_img_feats,
    query_txt_feats = val_txt_feats,
    query_dino_feats= val_dino_feats,   # ← thêm dòng này
)

# Tuning beta — thêm query_dino_feats=val_dino_feats
val_m = evaluate_map_two_stage(
    query_df        = df_val,
    alpha           = best_alpha_2,
    retrieval_k     = best_k_dino,
    final_k         = 5,
    query_img_feats = val_img_feats,
    query_txt_feats = val_txt_feats,
    query_dino_feats= val_dino_feats,   # ← thêm dòng này
    beta            = beta,
)

# Đánh giá TEST — thêm query_dino_feats=test_dino_feats
test_metrics_dino = evaluate_map_two_stage(
    query_df        = df_test,
    alpha           = best_alpha_2,
    retrieval_k     = best_k_dino,
    final_k         = 5,
    query_img_feats = test_img_feats,
    query_txt_feats = test_txt_feats,
    query_dino_feats= test_dino_feats,  # ← thêm dòng này
    beta            = best_beta,
)
```

---

## Fix 3 — Upgrade DINOv2 Small → Base (Tăng metric ~+0.02–0.04 mAP@5)

**Cell:** `Bước 1 — Load DINOv2`  
**Vấn đề:** DINOv2-Small (`dinov2_vits14`) cho vector 384 chiều. DINOv2-Base (`dinov2_vitb14`) cho 768 chiều, biểu diễn chi tiết hơn đáng kể, đặc biệt với ảnh sản phẩm e-commerce có texture và logo phức tạp.  
**VRAM:** DINOv2-Base cần ~2.5 GB VRAM → vừa đủ T4 sau khi đã unload MobileCLIP.

**Thay tất cả chỗ load DINOv2 (có 2 chỗ trong nhánh `if` và `else` của Cell Bước 1):**

```python
# CŨ:
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

# MỚI:
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
```

**Cập nhật comment trong `get_dinov2_embedding` (cả 2 chỗ định nghĩa):**

```python
@torch.no_grad()
def get_dinov2_embedding(img_path):
    """Trả về vector 768 chiều đã chuẩn hóa L2."""  # ← 384 → 768
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception:
        img = Image.new('RGB', (224, 224), (128, 128, 128))
    img_tensor = transform_dino(img).unsqueeze(0).to(DEVICE)
    feats = dinov2.forward_features(img_tensor)['x_norm_clstoken']
    feats = F.normalize(feats, dim=-1)
    return feats.cpu().numpy().flatten()
```

**Xóa cache cũ (nếu đã chạy trước đó với vits14):**

```python
# Chạy cell này 1 lần để force recompute với vitb14
import os
for cache_path in [
    os.path.join(FEAT_DIR, 'dinov2_gallery.npy'),
    os.path.join(FEAT_DIR, 'dinov2_val.npy'),
    os.path.join(FEAT_DIR, 'dinov2_test.npy'),
]:
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print(f'🗑️  Đã xóa cache cũ: {cache_path}')
print('✅ Sẵn sàng recompute với DINOv2-Base')
```

---

## Fix 4 — Sửa các lỗi nhỏ (Cosmetic / Safety)

### 4a — Unreachable `print` trong `search_two_stage`

**Cell:** `Bước 2 — Hàm Tìm Kiếm 2 Giai Đoạn`

```python
# CŨ (print nằm sau return, không bao giờ chạy):
    return final_indices, final_scores
    print('✅ Hàm search_two_stage sẵn sàng!')

# MỚI (di chuyển print ra ngoài hàm):
    return final_indices, final_scores

print('✅ Hàm search_two_stage sẵn sàng!')
```

### 4b — Duplicate import `tqdm`

**Cell:** `Bước 3 — Hàm Đánh Giá`

```python
# CŨ:
from tqdm.notebook import tqdm
from tqdm.notebook import tqdm   # dòng thừa

# MỚI:
from tqdm.notebook import tqdm
```

### 4c — `dir()` → `globals()` để check model tồn tại

**Cell:** `Bước 1` — nhánh `else` (khi đã có cache)

```python
# CŨ:
if 'dinov2' not in dir() or dinov2 is None:

# MỚI:
if 'dinov2' not in globals() or globals()['dinov2'] is None:
```

**Tương tự tại Cell 10 (Bước 4):**

```python
# CŨ:
if 'test_metrics_2' in dir():

# MỚI:
if 'test_metrics_2' in globals():
```

---

## Thứ tự thực hiện

```
Fix 4 (nhỏ, không phụ thuộc)
  → Fix 1 (normalize) + chạy lại Cell 8
  → Fix 3 (đổi sang vitb14) + xóa cache DINOv2
  → Fix 2 (pre-compute query features)
  → Re-run toàn bộ tuning và test eval
```

Sau khi áp dụng đầy đủ, thứ tự chạy notebook:

| Bước | Cell       | Ghi chú                                       |
| ---- | ---------- | --------------------------------------------- |
| 1    | Cell 0–3   | Setup, không đổi                              |
| 2    | Cell 4     | Data split, không đổi                         |
| 3    | Cell 5     | Load MobileCLIP, không đổi                    |
| 4    | **Cell 6** | **Đã fix normalize**                          |
| 5    | Cell 7     | Fusion/FAISS utils, không đổi                 |
| 6    | **Cell 8** | **Chạy lại** để có features đã normalize      |
| 7    | Cell 9     | Grid search alpha trên val                    |
| 8    | Cell 10    | Test eval baseline MobileCLIP                 |
| 9    | Bước 0     | Build FAISS Stage 1, không đổi                |
| 10   | **Bước 1** | **DINOv2-Base** + pre-compute val/test DINOv2 |
| 11   | **Bước 2** | **search_two_stage** đã vectorize             |
| 12   | **Bước 3** | Tuning k + beta (nhanh hơn nhiều)             |
| 13   | Bước 4     | Test eval cuối, chạy 1 lần                    |
| 14   | Bước 5     | Export CSV                                    |
