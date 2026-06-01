# HƯỚNG DẪN: Thêm YOLO + SAHI + Crop Box + pHash vào file Tuan4_GiaVy_Pipeline.ipynb

## BỐI CẢNH

File hiện tại đã có:

- Cell 0: Kiểm tra GPU
- Cell 1: Cài thư viện + MobileCLIP
- Cell 2: Mount Drive + tìm đường dẫn
- Cell 3: Import + cấu hình
- Cell 4: Đọc dữ liệu + chia val/test (STRICT SPLIT)
- Cell 5: Tải MobileCLIP
- Cell 6: Hàm trích xuất image + text features
- Cell 7: Fusion + FAISS + Metrics utils
- Cell 8: Trích xuất tất cả features
- Cell 9+: Grid search alpha + đánh giá

**Pipeline hiện tại:**

```
MobileCLIP (GĐ1) → FAISS top-100 → DINOv2 rerank (GĐ2) → top-5
```

**Pipeline mới sau khi thêm:**

```
YOLO + SAHI detect → Crop bbox → MobileCLIP (GĐ1) → FAISS top-100
    → DINOv2 rerank (GĐ2) + pHash boost → top-5
```

---

## THAY ĐỔI 1: Cell 1 — Thêm cài YOLO + SAHI

**Tìm dòng này trong Cell 1:**

```python
!pip install -q faiss-gpu timm
```

**Thêm ngay sau dòng đó:**

```python
# Cài YOLO + SAHI để detect và crop sản phẩm
!pip install -q ultralytics
!pip install -q sahi
print('✅ YOLO + SAHI đã cài xong!')
```

---

## THAY ĐỔI 2: Cell 3 — Thêm config YOLO + pHash

**Tìm phần cấu hình trong Cell 3 (phần có MOBILECLIP_VARIANT, BATCH_SIZE...):**

**Thêm vào cuối phần cấu hình:**

```python
# ─── Cấu hình YOLO + SAHI ────────────────────────────────────────────────────
YOLO_MODEL        = 'yolov8s.pt'     # yolov8n (nhẹ nhất) / yolov8s / yolov8m
YOLO_CONF         = 0.25             # ngưỡng confidence detect
SAHI_SLICE_SIZE   = 512              # kích thước tile SAHI
SAHI_OVERLAP      = 0.2              # độ overlap giữa các tile
USE_SAHI          = True             # True = dùng SAHI cho ảnh nhỏ

# ─── Cấu hình pHash Boost ────────────────────────────────────────────────────
PHASH_THRESHOLD   = 5                # Hamming distance ≤ 5 → boost
PHASH_BOOST       = 0.3              # điểm cộng thêm khi pHash gần giống

# ─── Đường dẫn cache ─────────────────────────────────────────────────────────
YOLO_CROP_DIR     = '/content/drive/MyDrive/DuLieuPython/yolo_crops'
PHASH_CACHE_PATH  = '/content/drive/MyDrive/DuLieuPython/phash_array.npy'

import os
os.makedirs(YOLO_CROP_DIR, exist_ok=True)

print(f'✅ YOLO model     : {YOLO_MODEL}')
print(f'✅ SAHI           : {USE_SAHI}')
print(f'✅ pHash threshold: {PHASH_THRESHOLD}')
print(f'✅ pHash boost    : {PHASH_BOOST}')
```

---

## THÊM MỚI: Cell 5.5 — YOLO + SAHI Detect và Crop

**Thêm cell mới GIỮA Cell 5 (Load MobileCLIP) và Cell 6 (Extract features).**

Nội dung cell mới hoàn chỉnh:

```python
# ============================================================
# Cell 5.5: YOLO + SAHI — Detect và Crop sản phẩm
# ============================================================
# Mục đích: Thay vì dùng toàn bộ ảnh, detect bbox sản phẩm chính
# rồi crop ra → MobileCLIP chỉ nhìn vào sản phẩm, không bị nhiễu nền
#
# Fallback: Nếu YOLO không detect được → dùng toàn bộ ảnh
# Cache: Lưu crop ra Drive để không cần chạy lại khi Colab reset

from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from PIL import Image
import os

# ── Load YOLO model ──────────────────────────────────────────
print('⏳ Loading YOLO model...')
yolo_model = YOLO(YOLO_MODEL)
print(f'✅ YOLO {YOLO_MODEL} loaded!')

# ── Load SAHI wrapper ─────────────────────────────────────────
if USE_SAHI:
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=YOLO_MODEL,
        confidence_threshold=YOLO_CONF,
        device=DEVICE
    )
    print('✅ SAHI model loaded!')


def detect_and_crop_yolo(img_path: str,
                          yolo_model,
                          sahi_model=None,
                          use_sahi: bool = True,
                          conf: float = 0.25,
                          crop_dir: str = None) -> Image.Image:
    """
    Detect sản phẩm bằng YOLO (+SAHI nếu use_sahi=True).
    Trả về crop của bbox lớn nhất.
    Fallback: trả về toàn bộ ảnh nếu không detect được.
    Cache: lưu crop ra crop_dir để dùng lại.
    """
    img_name = os.path.splitext(os.path.basename(img_path))[0]

    # Kiểm tra cache
    if crop_dir:
        cache_path = os.path.join(crop_dir, img_name + '.jpg')
        if os.path.exists(cache_path):
            return Image.open(cache_path).convert('RGB')

    try:
        pil_img = Image.open(img_path).convert('RGB')
        w, h    = pil_img.size

        boxes = []

        if use_sahi and sahi_model is not None:
            # SAHI: chia ảnh thành tile nhỏ → detect ảnh nhỏ tốt hơn
            result = get_sliced_prediction(
                img_path,
                sahi_model,
                slice_height=SAHI_SLICE_SIZE,
                slice_width=SAHI_SLICE_SIZE,
                overlap_height_ratio=SAHI_OVERLAP,
                overlap_width_ratio=SAHI_OVERLAP,
                verbose=0
            )
            for obj in result.object_prediction_list:
                b = obj.bbox
                boxes.append((b.minx, b.miny, b.maxx, b.maxy,
                               obj.score.value))
        else:
            # YOLO thường (không SAHI)
            results = yolo_model(img_path, conf=conf, verbose=False)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    boxes.append((x1, y1, x2, y2, float(box.conf)))

        if not boxes:
            # Fallback: không detect được → dùng toàn bộ ảnh
            if crop_dir:
                pil_img.save(cache_path, quality=90)
            return pil_img

        # Chọn bbox lớn nhất (diện tích lớn nhất = sản phẩm chính)
        best_box = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
        x1, y1, x2, y2, _ = best_box

        # Thêm padding 5% để không bị cắt sát
        pad_x = (x2 - x1) * 0.05
        pad_y = (y2 - y1) * 0.05
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        crop = pil_img.crop((int(x1), int(y1), int(x2), int(y2)))

        # Lưu cache
        if crop_dir:
            crop.save(cache_path, quality=90)

        return crop

    except Exception:
        # Bất kỳ lỗi nào → fallback toàn ảnh
        return Image.open(img_path).convert('RGB')


# ── Test thử 1 ảnh để kiểm tra pipeline ─────────────────────
test_path = os.path.join(IMG_DIR, df['image'].iloc[0])
test_crop = detect_and_crop_yolo(
    test_path, yolo_model, sahi_model if USE_SAHI else None,
    use_sahi=USE_SAHI, conf=YOLO_CONF, crop_dir=YOLO_CROP_DIR
)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(Image.open(test_path).convert('RGB'))
axes[0].set_title('Ảnh gốc'); axes[0].axis('off')
axes[1].imshow(test_crop)
axes[1].set_title(f'YOLO Crop ({test_crop.size[0]}×{test_crop.size[1]})')
axes[1].axis('off')
plt.suptitle('YOLO + SAHI Detection Test')
plt.tight_layout(); plt.show()

# ── Thống kê cache hiện có ───────────────────────────────────
n_cached = len([f for f in os.listdir(YOLO_CROP_DIR) if f.endswith('.jpg')])
print(f'\n📁 Cache hiện có: {n_cached:,}/{len(df):,} ảnh')
print('✅ YOLO + SAHI sẵn sàng!')
```

---

## THAY ĐỔI 3: Cell 6 — Sửa hàm trích xuất để dùng YOLO crop

**Tìm class `ShopeeImageDataset` trong Cell 6:**

```python
class ShopeeImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, return_tensor=True):
        self.df           = df.reset_index(drop=True)
        self.img_dir      = img_dir
        self.transform    = transform
        self.return_tensor = return_tensor
```

**Thay toàn bộ class đó bằng:**

```python
class ShopeeImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, return_tensor=True,
                 use_yolo_crop=True, crop_dir=None):
        self.df            = df.reset_index(drop=True)
        self.img_dir       = img_dir
        self.transform     = transform
        self.return_tensor = return_tensor
        self.use_yolo_crop = use_yolo_crop
        self.crop_dir      = crop_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.df.at[idx, 'image'])
        try:
            if self.use_yolo_crop and self.crop_dir:
                # Dùng YOLO crop nếu có
                img = detect_and_crop_yolo(
                    img_path, yolo_model,
                    sahi_model if USE_SAHI else None,
                    use_sahi=USE_SAHI,
                    conf=YOLO_CONF,
                    crop_dir=self.crop_dir
                )
            else:
                img = Image.open(img_path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (256, 256), (128, 128, 128))

        if self.return_tensor and self.transform:
            return self.transform(img)
        return img
```

**Tìm hàm `extract_image_features_clip` trong Cell 6:**

```python
dataset = ShopeeImageDataset(df_input, img_dir,
                             transform=preprocess, return_tensor=True)
```

**Thay bằng:**

```python
dataset = ShopeeImageDataset(
    df_input, img_dir,
    transform=preprocess, return_tensor=True,
    use_yolo_crop=True,        # ← thêm 2 dòng này
    crop_dir=YOLO_CROP_DIR     # ← dùng YOLO crop
)
```

---

## THÊM MỚI: Cell 7.5 — pHash Array + Hàm Boost

**Thêm cell mới GIỮA Cell 7 (Fusion/FAISS utils) và Cell 8 (Trích xuất features).**

Nội dung cell mới hoàn chỉnh:

```python
# ============================================================
# Cell 7.5: pHash Array + Hàm Boost cho Reranking
# ============================================================
# pHash boost: ảnh có Hamming distance nhỏ → cùng ảnh/gần trùng
# → cộng thêm điểm để đẩy lên top
# Cache: lưu phash_array.npy để không tính lại

import numpy as np

def compute_phash_array(df_input: pd.DataFrame,
                         save_path: str = None) -> np.ndarray:
    """
    Chuyển cột image_phash (hex string) thành mảng uint64.
    Dùng để tính Hamming distance nhanh bằng bit-twiddling.
    """
    phash_ints = np.array(
        [int(h, 16) for h in df_input['image_phash'].values],
        dtype=np.uint64
    )
    if save_path:
        np.save(save_path, phash_ints)
        print(f'✅ Đã lưu pHash array → {save_path}')
    return phash_ints


def hamming_distance_batch(query_hash: np.uint64,
                            gallery_hashes: np.ndarray) -> np.ndarray:
    """
    Tính Hamming distance giữa 1 query hash và toàn bộ gallery.
    Dùng thuật toán bit-twiddling (không cần vòng lặp Python).
    Trả về mảng uint8 (0–64).
    """
    x = query_hash ^ gallery_hashes
    x = (x & np.uint64(0x5555555555555555)) + \
        ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + \
        ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x & np.uint64(0x0F0F0F0F0F0F0F0F)) + \
        ((x >> np.uint64(4)) & np.uint64(0x0F0F0F0F0F0F0F0F))
    x = (x & np.uint64(0x00FF00FF00FF00FF)) + \
        ((x >> np.uint64(8)) & np.uint64(0x00FF00FF00FF00FF))
    x = (x & np.uint64(0x0000FFFF0000FFFF)) + \
        ((x >> np.uint64(16)) & np.uint64(0x0000FFFF0000FFFF))
    return ((x & np.uint64(0x00000000FFFFFFFF)) + \
            (x >> np.uint64(32))).astype(np.uint8)


def phash_boost_scores(query_phash: np.uint64,
                        candidate_indices: np.ndarray,
                        gallery_phashes: np.ndarray,
                        base_scores: np.ndarray,
                        threshold: int = PHASH_THRESHOLD,
                        boost: float = PHASH_BOOST) -> np.ndarray:
    """
    Cộng thêm điểm boost cho các candidate có pHash gần giống query.
    - Hamming distance = 0  → boost × 2 (ảnh giống hệt)
    - Hamming distance ≤ threshold → boost × 1
    - Hamming distance > threshold → không boost
    """
    cand_hashes = gallery_phashes[candidate_indices]
    ham         = hamming_distance_batch(query_phash, cand_hashes)

    boosted = base_scores.copy()
    boosted[ham <= threshold] += boost
    boosted[ham == 0]         += boost  # double boost nếu giống hệt
    return boosted


# ── Load hoặc tạo pHash array ────────────────────────────────
if os.path.exists(PHASH_CACHE_PATH):
    gallery_phashes = np.load(PHASH_CACHE_PATH)
    print(f'✅ Load pHash array từ cache: {gallery_phashes.shape}')
else:
    print('⏳ Tạo pHash array...')
    gallery_phashes = compute_phash_array(df, save_path=PHASH_CACHE_PATH)

print(f'✅ pHash array: {gallery_phashes.shape}, dtype={gallery_phashes.dtype}')

# ── Test thử pHash boost ──────────────────────────────────────
# Lấy 2 ảnh cùng nhóm, kiểm tra Hamming distance
sample_row = df.iloc[0]
same_group = df[df['label_group'] == sample_row['label_group']]
if len(same_group) > 1:
    h1 = int(sample_row['image_phash'], 16)
    h2 = int(same_group.iloc[1]['image_phash'], 16)
    dist = hamming_distance_batch(
        np.uint64(h1), np.array([h2], dtype=np.uint64)
    )[0]
    print(f'\n📊 Test pHash: 2 ảnh cùng nhóm → Hamming distance = {dist}')
    print(f'   (distance ≤ {PHASH_THRESHOLD} → sẽ được boost +{PHASH_BOOST})')

print('✅ pHash functions sẵn sàng!')
```

---

## THAY ĐỔI 4: Sửa hàm `evaluate_retrieval_clip` — Thêm pHash boost

**Tìm hàm `evaluate_retrieval_clip` trong Cell 7:**

```python
def evaluate_retrieval_clip(query_df, gallery_df, query_img, query_txt,
                             gallery_img, gallery_txt, alpha, K=5):
```

**Thêm tham số `use_phash_boost=True` và logic boost vào hàm:**

```python
def evaluate_retrieval_clip(query_df, gallery_df, query_img, query_txt,
                             gallery_img, gallery_txt, alpha, K=5,
                             use_phash_boost=True):
    q_fused      = fuse_and_normalize_clip(query_img, query_txt, alpha)
    g_fused      = fuse_and_normalize_clip(gallery_img, gallery_txt, alpha)
    gt_dict      = get_ground_truth_dict(gallery_df)
    index        = build_faiss_index(g_fused)

    # Lấy top-(K*10) để có đủ candidate cho pHash rerank
    RERANK_K     = K * 20
    _, indices   = index.search(q_fused, RERANK_K + 1)
    gallery_pids = gallery_df['posting_id'].tolist()
    gallery_idx  = {pid: i for i, pid in enumerate(gallery_pids)}

    ap_list, p1_list, r5_list = [], [], []

    for i, row in enumerate(query_df.itertuples()):
        qid      = row.posting_id
        relevant = gt_dict.get(qid, set()) - {qid}
        if not relevant:
            continue

        # Lấy candidate indices (loại self)
        cand_idx    = []
        cand_scores = []
        for idx in indices[i]:
            pid = gallery_pids[idx]
            if pid != qid:
                cand_idx.append(idx)
                # Tính base score = dot product với gallery
                cand_scores.append(
                    float(np.dot(q_fused[i], g_fused[idx]))
                )
            if len(cand_idx) == RERANK_K:
                break

        if not cand_idx:
            continue

        cand_idx_arr    = np.array(cand_idx, dtype=np.int64)
        cand_scores_arr = np.array(cand_scores, dtype=np.float32)

        # pHash boost (nếu bật)
        if use_phash_boost and 'gallery_phashes' in globals():
            q_phash      = gallery_phashes[gallery_idx.get(qid, 0)]
            cand_scores_arr = phash_boost_scores(
                q_phash, cand_idx_arr,
                gallery_phashes, cand_scores_arr
            )

        # Rerank theo final score
        order    = np.argsort(-cand_scores_arr)
        retrieved = [gallery_pids[cand_idx_arr[o]] for o in order[:K]]

        hits, ap = 0, 0.0
        for rank, pid in enumerate(retrieved, 1):
            if pid in relevant:
                hits += 1
                ap   += hits / rank
        ap_list.append(ap / min(len(relevant), K))
        p1_list.append(1.0 if (retrieved and retrieved[0] in relevant) else 0.0)
        r5_list.append(len(set(retrieved) & relevant) / len(relevant))

    return {
        'mAP@5'      : float(np.mean(ap_list)),
        'Precision@1': float(np.mean(p1_list)),
        'Recall@5'   : float(np.mean(r5_list)),
    }
```

---

## THỨ TỰ CHẠY SAU KHI SỬA

```
Cell 0: Kiểm tra GPU
Cell 1: Cài thư viện (+ YOLO + SAHI)
Cell 2: Mount Drive
Cell 3: Import + config (+ YOLO/pHash config)
Cell 4: Đọc data + chia val/test
Cell 5: Load MobileCLIP
Cell 5.5: [MỚI] Load YOLO + SAHI + test detect
Cell 6: Trích xuất features (dùng YOLO crop)
Cell 7: Fusion/FAISS/Metric utils (+ evaluate có pHash boost)
Cell 7.5: [MỚI] pHash array + boost functions
Cell 8+: Trích xuất all features → Grid search → Đánh giá test
```

---

## LƯU Ý QUAN TRỌNG

1. **Cell 5.5 phải chạy TRƯỚC Cell 6** vì Cell 6 dùng `yolo_model`, `sahi_model` từ Cell 5.5

2. **Cell 7.5 phải chạy TRƯỚC Grid Search** vì hàm evaluate mới dùng `gallery_phashes`

3. **Cache YOLO crops** lưu vào `YOLO_CROP_DIR` trên Drive — lần đầu chạy chậm (~30-40 phút), lần sau load cache ngay

4. **Cache pHash array** lưu vào `PHASH_CACHE_PATH` — chỉ tính 1 lần, ~vài giây

5. **SAHI có thể chậm** với ảnh lớn — nếu cần nhanh hơn, đặt `USE_SAHI = False` để chỉ dùng YOLO thường

6. **Không thay đổi** phần grid search alpha, val/test split, và cấu trúc các cell còn lại

---

## KẾT QUẢ MONG ĐỢI

| Metric      | Trước (không YOLO/pHash) | Sau (có YOLO+SAHI+pHash) |
| ----------- | ------------------------ | ------------------------ |
| mAP@5       | ~0.75–0.80               | ~0.82–0.88               |
| Precision@1 | ~0.75                    | ~0.80–0.87               |
| Recall@5    | ~0.72                    | ~0.76–0.83               |

Tăng nhờ:

- YOLO crop loại nhiễu nền → MobileCLIP vector chất lượng hơn
- SAHI bắt được sản phẩm nhỏ/góc xa
- pHash boost đẩy ảnh gần trùng lên top
