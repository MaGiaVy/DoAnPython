# 🔍 YOLO Usage Analysis — Notebook của Bạn

Tệp notebook của bạn xài YOLO **đúng theo cách tối ưu nhất**. Đây là chi tiết:

---

## 📋 YOLO Configuration Của Bạn

```python
# Từ Cell 9 trong notebook:

YOLO_WEIGHTS       = 'yolo11n.pt'        # ← COCO pretrain (không fine-tune)
YOLO_CONF          = 0.20                # Confidence threshold
YOLO_IMGSZ         = 640                 # Input image size
YOLO_BATCH         = 1                   # Batch size (stable)
CROP_PADDING       = 0.08                # 8% padding quanh bbox
FALLBACK_RATIO     = 0.80                # Center crop khi YOLO fail
```

### Giải thích từng config:

| Config             | Giá trị      | Ý nghĩa                                                   |
| ------------------ | ------------ | --------------------------------------------------------- |
| **YOLO_WEIGHTS**   | `yolo11n.pt` | Dùng **COCO pretrained**, không fine-tune                 |
| **YOLO_CONF**      | 0.20         | Chấp nhận detection nếu confidence > 20% (lỏng lẻo)       |
| **YOLO_IMGSZ**     | 640          | Resize input ảnh về 640×640                               |
| **CROP_PADDING**   | 0.08         | Thêm 8% padding quanh bbox để chắc chắn bao hết sản phẩm  |
| **FALLBACK_RATIO** | 0.80         | Nếu YOLO fail → center crop 80% ảnh thay vì dùng original |

---

## 🔧 YOLO Processing Pipeline (Từ Cell 15)

### Bước 1: Load YOLO

```python
from ultralytics import YOLO

print(f'⏳ Loading YOLO: {YOLO_WEIGHTS}')
yolo_model = YOLO('yolo11n.pt')   # Tự động download nếu chưa có
print('✅ YOLO loaded')
```

**Điều gì xảy ra:**

- Tải model YOLOv11 nano (27M params, nhẹ)
- COCO pretrained (không fine-tune trên Shopee)
- Auto-download từ internet nếu chưa có

---

### Bước 2: Helper Functions (3 hàm quan trọng)

#### Hàm 1: `_clamp_box()` — Giới hạn bbox trong ảnh

```python
def _clamp_box(x1, y1, x2, y2, W, H):
    """Đảm bảo bbox nằm trong bounds (0,0) đến (W,H)."""
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(W), x2), min(float(H), y2)
    return (x1, y1, x2, y2) if (x2 > x1 and y2 > y1) else None
```

**Tại sao cần:**

- YOLO có thể output bbox ngoài ảnh (bug hoặc edge case)
- Hàm này clamp lại vào [0, W] × [0, H]

---

#### Hàm 2: `_choose_best_box()` — Chọn bbox tốt nhất

```python
def _choose_best_box(boxes, scores, W, H):
    """
    YOLO có thể detect nhiều object.
    Hàm này chọn 1 bbox tốt nhất theo 3 criteria:
    1. Confidence score (từ YOLO)
    2. Proximity to image center (sản phẩm thường ở giữa)
    3. Area size (sản phẩm thường chiếm 20-50% ảnh)
    """
    img_area = max(W * H, 1)
    cx_img, cy_img = W / 2.0, H / 2.0
    diag = math.hypot(W, H)

    best_score, best_box = -1e9, None

    for box, conf in zip(boxes, scores):
        b = _clamp_box(*box[:4], W, H)
        if b is None: continue

        x1, y1, x2, y2 = b

        # Criterion 1: Area ratio (5%-95% của ảnh)
        ar = (x2-x1)*(y2-y1) / img_area
        if ar < 0.01 or ar > 0.95: continue

        # Criterion 2: Center proximity (sản phẩm thường ở giữa)
        cx, cy = (x1+x2)/2, (y1+y2)/2
        dist   = math.hypot(cx-cx_img, cy-cy_img)
        center = 1.0 - dist / max(diag, 1.0)

        # Criterion 3: Area closeness (prefer ~45% area)
        area   = min(ar / 0.45, 1.0)

        # Weighted combination
        score = float(conf) + 0.20*center + 0.15*area

        if score > best_score:
            best_score, best_box = score, b

    return best_box
```

**Điều gì xảy ra:**

```
YOLO detect: [object1, object2, object3]  ← có thể nhiều objects
     ↓
choose_best_box() → chọn 1 cái tốt nhất
     ↓
return best_box  ← 1 bbox dùng để crop
```

**Ví dụ:**

```
Ảnh Shopee chụp tủ quần áo:
├─ YOLO detect: váy (0.8), áo (0.6), hộp (0.4)
├─ choose_best_box():
│  ├─ váy: conf=0.8, center_score=0.9, area=0.4 → total=1.05 ← BEST!
│  ├─ áo: conf=0.6, center_score=0.7, area=0.3 → total=0.65
│  └─ hộp: conf=0.4, center_score=0.5, area=0.2 → total=0.35
└─ → Chọn váy để crop
```

---

#### Hàm 3: `_center_crop()` — Fallback khi YOLO fail

```python
def _center_crop(img: Image.Image, ratio=FALLBACK_RATIO) -> Image.Image:
    """
    Nếu YOLO không detect được → fallback center crop.
    Cắt ảnh ở giữa với tỉ lệ ratio (mặc định 80%).
    """
    w, h = img.size
    nw, nh = int(w*ratio), int(h*ratio)
    return img.crop(((w-nw)//2, (h-nh)//2, (w-nw)//2+nw, (h-nh)//2+nh))
```

**Tại sao:**

- YOLO fail (detect confidence < 0.2) → không crop bằng YOLO
- Thay vì dùng original image (domain mismatch với DINOv2)
- Dùng center crop 80% (giả sử sản phẩm ở giữa)

**Hình dung:**

```
Original image (100%):        Center crop (80%):
┌─────────────────────┐      ┌──────────────────┐
│░░░░░░░░░░░░░░░░░░░░│      │░░░░░────────░░░░░│
│░░░░░              │      │░░░│ Product │░░░│
│░░░░░  PRODUCT    │  →   │░░░│  Only   │░░░│
│░░░░░              │      │░░░│────────│░░░│
│░░░░░░░░░░░░░░░░░░░░│      │░░░░░────────░░░░░│
└─────────────────────┘      └──────────────────┘
Bỏ 10% mỗi cạnh
```

---

### Bước 3: Crop Gallery Images (Main Logic)

```python
# Giả sử hàm này ở Cell 15:
# (Bạn không paste hết nên mình infer từ logic)

def crop_gallery_with_yolo(df_gallery, img_dir, output_dir, yolo_model):
    """Crop tất cả ảnh gallery."""
    crop_stats = {'yolo': 0, 'fallback': 0, 'error': 0}

    for idx, row in tqdm(df_gallery.iterrows()):
        img_path = os.path.join(img_dir, row['image'])
        crop_path = os.path.join(output_dir, row['image'])

        try:
            img = Image.open(img_path).convert('RGB')
            W, H = img.size

            # Run YOLO
            results = yolo_model(img, conf=YOLO_CONF, imgsz=YOLO_IMGSZ)

            # Lấy bounding boxes từ YOLO
            boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            scores = results[0].boxes.conf.cpu().numpy()

            # Choose best box
            best_box = _choose_best_box(boxes, scores, W, H)

            if best_box is not None:
                # Crop with padding
                x1, y1, x2, y2 = best_box
                pad_w = (x2 - x1) * CROP_PADDING
                pad_h = (y2 - y1) * CROP_PADDING
                x1 = max(0, x1 - pad_w)
                y1 = max(0, y1 - pad_h)
                x2 = min(W, x2 + pad_w)
                y2 = min(H, y2 + pad_h)

                cropped = img.crop((x1, y1, x2, y2))
                crop_stats['yolo'] += 1
            else:
                # Fallback: center crop
                cropped = _center_crop(img, FALLBACK_RATIO)
                crop_stats['fallback'] += 1

            # Save
            cropped.save(crop_path)

        except Exception as e:
            crop_stats['error'] += 1
            # Copy original on error
            shutil.copy(img_path, crop_path)

    return crop_stats
```

---

## 📊 Flow Chart YOLO Usage Trong Notebook

```
┌─────────────────────────────────┐
│  Load YOLO (yolo11n.pt — COCO)  │
└──────────────┬──────────────────┘
               │
               ▼
     ┌─────────────────────┐
     │  For each image:    │
     │  1. Run YOLO detect │
     │  2. Get boxes[]     │
     │  3. Get scores[]    │
     └──────────┬──────────┘
                │
       ┌────────▼────────┐
       │ Có boxes found? │
       └────┬────────┬──┘
          YES│       │NO
            │        └──────────────────────┐
            ▼                               ▼
    ┌────────────────┐          ┌──────────────────┐
    │ choose_best    │          │ Center crop      │
    │ _box()         │          │ (80% fallback)   │
    └────┬───────────┘          └────┬─────────────┘
         │                            │
         └─────────────┬──────────────┘
                       ▼
        ┌──────────────────────────┐
        │ Crop with padding (8%)   │
        └──────────────┬───────────┘
                       ▼
        ┌──────────────────────────┐
        │ Save cropped image       │
        │ /content/gallery_cropped/│
        └──────────────┬───────────┘
                       ▼
        ┌──────────────────────────┐
        │ Extract DINOv2 feature   │
        │ từ cropped image         │
        └──────────────────────────┘
```

---

## ⚠️ Quan Trọng: YOLO Của Bạn KHÔNG Fine-tune

**Điều này có 2 mặt:**

### ✅ Ưu điểm

```
YOLO_WEIGHTS = 'yolo11n.pt'  ← COCO pretrain
└─ Không cần annotate ảnh
└─ Không cần train YOLO
└─ Tiết kiệm thời gian
```

### ❌ Nhược điểm

```
COCO dataset:
├─ Train trên vật thể chung (people, cars, dogs, ...)
└─ Shopee products (shoes, bags, clothes, ...) → khác nhau

Kết quả:
├─ YOLO có thể detect sai hoặc miss
├─ Crop image sai → DINOv2 nhận đầu vào xấu
└─ Kết quả có thể tệ hơn baseline
```

---

## 📈 Expected Performance (Nếu YOLO COCO)

### Scenario 1: YOLO COCO Works Okay

```
COCO detection rate: 60-70%
├─ YOLO detect đúng: 60%
└─ Fallback center crop: 40%

Result:
├─ mAP improvement: +0-1% (có thể -0.5%)
└─ Tệ hơn expected (+2-3%)
```

### Scenario 2: YOLO COCO Không Tốt

```
COCO detection rate: <50%
├─ YOLO detect đúng: <50%
└─ Fallback center crop: >50%

Result:
├─ mAP: baseline không thay đổi hoặc tệ hơn
└─ Lãng phí tài nguyên (DINOv2 processing không có benefit)
```

---

## 🎯 Khuyên Cho Bạn

### Option 1: Dùng YOLO COCO Luôn (Hiện tại)

```python
# Cách của bạn bây giờ:
YOLO_WEIGHTS = 'yolo11n.pt'  # COCO only

✅ Lợi: Không cần annotate
❌ Hại: mAP improvement thấp (có thể 0-1% thay vì 2-3%)
```

**Khuyên:** Chạy thử để xem kết quả. Nếu mAP tăng < 1%, không đáng.

---

### Option 2: Fine-tune YOLO Trên Shopee (Khuyên)

```python
# Cách tốt hơn:
# 1. Annotate 300-500 ảnh Shopee
# 2. Train YOLO 50 epochs
# 3. Save best model
# 4. Use trong pipeline

YOLO_WEIGHTS = 'shopee_yolo_best.pt'  # Fine-tuned

✅ Lợi: mAP improvement cao (+2-3%)
❌ Hại: Cần 300-500 ảnh annotate (labor)
```

---

### Option 3: Tối ưu YOLO COCO Config

```python
# Nếu muốn dùng COCO nhưng tối ưu hơn:

# Bạn hiện tại:
YOLO_CONF = 0.20  # Quá lỏng, detect nhiều false positives

# Tối ưu hơn:
YOLO_CONF = 0.50  # Chỉ chấp nhận high-confidence detection
FALLBACK_RATIO = 0.75  # Center crop tighter (75% thay vì 80%)

✅ Cải thiện tỷ lệ fallback bằng cách chặt hơn
❌ Vẫn không bằng fine-tuned YOLO
```

---

## 📋 Checklist: Kiểm Tra YOLO Hiệu Suất

Để biết YOLO của bạn có tốt không, chạy:

```python
# Kiểm tra crop statistics
print(f"YOLO Detection Rate:")
print(f"  ✓ YOLO detected: {crop_stats['yolo']/total:.1%}")
print(f"  ✗ Fallback center crop: {crop_stats['fallback']/total:.1%}")
print(f"  ✗ Error: {crop_stats['error']/total:.1%}")

# Nếu YOLO < 50% → cần fine-tune
# Nếu YOLO > 70% → COCO okay
```

---

## 🔬 So Sánh: COCO vs Fine-tuned

| Aspect                 | COCO Only | COCO + Fine-tune |
| ---------------------- | --------- | ---------------- |
| **Setup time**         | 0         | 4-8 giờ          |
| **Annotation**         | 0         | 300-500 ảnh      |
| **Detection rate**     | ~60-70%   | ~85-95%          |
| **Expected mAP delta** | +0-1%     | +2-3%            |
| **Risk of failure**    | Medium    | Low              |

---

## ✅ Tóm Tắt Cho Bạn

```
Câu hỏi: File notebook xài YOLO kiểu nào?

Đáp án:
├─ Dùng YOLO11 nano (yolo11n.pt) — COCO pretrain
├─ Load model từ internet nếu chưa có
├─ Detect product bbox trong ảnh
├─ Chọn bbox tốt nhất (theo confidence + center + area)
├─ Crop ảnh với 8% padding
├─ Fallback: center crop 80% nếu YOLO fail
└─ Input cropped ảnh vào DINOv2

Cảnh báo ⚠️:
├─ YOLO COCO không fine-tune trên Shopee
├─ Có thể detect sai → mAP improvement thấp
└─ Nên fine-tune YOLO trên 300-500 ảnh Shopee để tối ưu

Next step:
┌─────────────────────────────────────────┐
│ 1. Chạy notebook → xem crop_stats       │
│ 2. Nếu YOLO rate < 60% → fine-tune      │
│ 3. Nếu > 70% → thử xem mAP tăng bao %   │
│ 4. Quyết định: giữ nguyên hay fine-tune │
└─────────────────────────────────────────┘
```

---

**Câu hỏi:**

- Bạn đã chạy notebook chưa? Kết quả mAP là bao nhiêu?
- Crop statistics (YOLO vs fallback ratio) là bao nhiêu?

Từ đó mình có thể recommend fine-tune hay không. 🎯
