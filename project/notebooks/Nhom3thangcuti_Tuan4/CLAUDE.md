# 🚀 Improved 2-Stage Visual Search Pipeline
## MobileCLIP + YOLO-Cropped DINOv2 Re-ranking

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Stage 1: MobileCLIP Retrieval](#stage-1-mobileclip-retrieval)
4. [Stage 2: DINOv2-Crop Re-ranking](#stage-2-dinov2-crop-re-ranking)
5. [Data Preparation & YOLO Training](#data-preparation--yolo-training)
6. [Implementation Steps](#implementation-steps)
7. [Hyperparameter Optimization](#hyperparameter-optimization)
8. [Inference Pipeline](#inference-pipeline)
9. [Code Examples](#code-examples)
10. [Risk Analysis & Recommendations](#risk-analysis--recommendations)
11. [Performance Comparison](#performance-comparison)

---

## Executive Summary

This improved pipeline enhances the baseline MobileCLIP system (mAP@5 ≈ 0.77) by adding a **second stage re-ranking using DINOv2** trained on **YOLO-cropped product images**.

### Key Innovation
- **Stage 1:** Retrieval via MobileCLIP (image + text fusion) on original images
- **Stage 2:** Re-ranking via DINOv2-Small on cropped product regions detected by YOLO
- **Result:** Expected improvement of **+2-3% mAP** (if YOLO detection is accurate)

### Expected Metrics
| Metric | Baseline | Improved | Gain |
|---|---|---|---|
| mAP@5 | 0.7708 | **~0.79-0.80** | +2-3% |
| Precision@1 | 0.7937 | **~0.81-0.82** | +1-2% |
| Recall@5 | 0.7430 | **~0.76-0.77** | +1-2% |
| Latency | ~500ms | **~2000ms** | ⚠️ 4× slower |

---

## Pipeline Architecture

### High-Level Flow

```
Query Image + Title
     │
     ├─→ [YOLO Crop] → Cropped product region
     │
     ├─→ [Stage 1: MobileCLIP]
     │   ├─ Extract: img_feat (from original) + txt_feat
     │   ├─ Fuse: alpha * img_feat + (1-alpha) * txt_feat
     │   └─ FAISS Search → top-K candidates (K=100-200)
     │
     └─→ [Stage 2: DINOv2-Crop]
         ├─ Extract: dino_feat from cropped query image
         ├─ For each candidate idx:
         │   ├─ dino_score = cosine_sim(q_dino, gallery_dino_crop[idx])
         │   ├─ clip_score = MobileCLIP score from Stage 1
         │   ├─ final_score = beta * dino_score + (1-beta) * clip_score
         │
         └─ Sort & return top-5
```

### Data Flow Diagram

```
34,250 Gallery Images
├─ Original Images (for MobileCLIP)
│  ├─ Image Features (512-dim, cached)
│  └─ Text Features (512-dim, cached)
│
├─ YOLO-Cropped Images (for DINOv2)
│  └─ DINOv2 Features (384-dim, cached)
│
Validation Set (20%)
├─ MobileCLIP features (for alpha tuning)
└─ DINOv2-Crop features (for k, beta tuning)

Test Set (80%)
└─ Final evaluation (run once only!)
```

---

## Stage 1: MobileCLIP Retrieval

### Purpose
**Fast retrieval from 34,250 items using cross-modal embeddings (image + text).**

### How It Works

1. **Input:** Query image + product title
2. **Feature Extraction:**
   ```python
   q_img_feat = MobileCLIP.encode_image(original_query_image)   # 512-dim
   q_txt_feat = MobileCLIP.encode_text(query_title)             # 512-dim
   ```

3. **Fusion (Weighted combination):**
   ```python
   q_fused = normalize(alpha * q_img_feat + (1-alpha) * q_txt_feat)
   ```
   - `alpha = 0.5` means equal weight to image and text
   - Grid search finds optimal alpha on validation set

4. **Retrieval via FAISS:**
   ```python
   distances, indices = faiss_index.search(q_fused, retrieval_k)
   # retrieval_k = number of candidates to retrieve (typically 100-200)
   ```

5. **Output:** Top-K candidate indices from gallery

### Why Original Images (Not Cropped)?
- ✅ MobileCLIP is trained on images **with context and background**
- ✅ Background helps generate better text embeddings (e.g., "shoe on shelf" vs "shoe")
- ✅ Consistent with pre-training data distribution
- ❌ Cropped images might hurt MobileCLIP performance

### Hyperparameter: Alpha
- **Range:** 0.0 to 1.0
- **0.0:** Text only
- **0.5:** Image + Text equally
- **1.0:** Image only
- **Typical best:** 0.4-0.6 (slight image bias)

---

## Stage 2: DINOv2-Crop Re-ranking

### Purpose
**Fine-grained visual similarity on product details (removes background noise).**

### How It Works

1. **Input:** Top-K candidates from Stage 1 + cropped query image

2. **DINOv2 Feature Extraction (on cropped images):**
   ```python
   q_dino_feat = DINOv2.encode(cropped_query_image)  # 384-dim, L2-normalized
   ```

3. **For Each Candidate (idx in top-K):**
   ```python
   gallery_dino_feat = gallery_dino_crop[idx]  # pre-computed, 384-dim
   dino_score = cosine_similarity(q_dino_feat, gallery_dino_feat)
   ```

4. **Score Fusion (Weighted combination):**
   ```python
   clip_score = clip_scores[idx]  # from Stage 1
   final_score = beta * dino_score + (1-beta) * clip_score
   ```
   - `beta = 0.3` means 30% trust DINOv2, 70% trust MobileCLIP
   - Grid search finds optimal beta on validation set

5. **Sorting & Output:**
   ```python
   sorted_indices = argsort(final_scores, descending=True)
   return sorted_indices[:5]  # top-5 results
   ```

### Why Cropped Images for DINOv2?
- ✅ **Focus on product details:** No background clutter, watermarks, text
- ✅ **Better visual discrimination:** DINOv2 sees only the actual product
- ✅ **Reduces false positives:** Less likely to match based on background
- ❌ **Domain shift risk:** If YOLO detection fails, DINOv2 sees out-of-distribution images

### Hyperparameter: Beta
- **Range:** 0.0 to 1.0
- **0.0:** Use only MobileCLIP (no re-ranking benefit)
- **0.3:** Slight DINOv2 boost (recommended for safety)
- **0.5:** Equal weight to both models
- **1.0:** Use only DINOv2 (risky if YOLO fails)

---

## Data Preparation & YOLO Training

### Step 1: Annotation Strategy

**Goal:** Label bounding boxes for products in Shopee images

#### Option A: Manual Annotation (Rigorous but Time-Consuming)
```
Tools: LabelImg, CVAT, Roboflow
Process:
  1. Select 500-1000 images from Shopee dataset (NOT from Gallery!)
  2. Annotate product bounding boxes (rectangle around product)
  3. Refine labels (ensure boxes are tight, centered)
  4. Split: 80% train, 10% val, 10% test for YOLO
```

#### Option B: Auto-Generate + Manual Refinement (Faster)
```
Tools: Pre-trained YOLO (from COCO) + manual review
Process:
  1. Run pre-trained YOLOv8s on Shopee images
  2. Filter predictions with confidence > 0.5
  3. Manually review & correct (fix misses, remove false positives)
  4. Same split: 80/10/10 for YOLO training
```

#### Option C: Strong Supervision + Transfer Learning (Hybrid)
```
Use existing e-commerce datasets (Amazon, eBay) for pre-training
  → Fine-tune on 100-200 Shopee images (cheaper annotation)
```

### Critical: Data Isolation

```python
# ❌ WRONG: Data leakage
all_images = 34,250
yolo_train = random sample from all_images
gallery = all_images  # overlap!

# ✅ CORRECT: Strict separation
yolo_train = 500-1000 images (from external source or reserved)
gallery = remaining 34,250 images (NO overlap with YOLO train)
val = 20% of gallery (6,850 images)
test = 80% of gallery (27,400 images)
```

### Step 2: YOLO Training

```python
from ultralytics import YOLO

# Load a pretrained model (YOLOv8s or YOLOv11n)
model = YOLO('yolov8s.pt')

# Train on Shopee dataset
results = model.train(
    data='shopee_dataset.yaml',  # Dataset config (train/val paths)
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,  # GPU
    patience=10,  # Early stopping
    augment=True,  # Data augmentation
    conf=0.5,  # Confidence threshold
)

# Evaluate on YOLO validation set
metrics = model.val()
print(f"mAP@0.5: {metrics.box.map50}")  # Should be > 0.80
```

### Step 3: Box Selection Strategy (choose_best_box)

```python
def choose_best_box(yolo_output, img_shape, confidence_threshold=0.5):
    """
    Select the best bounding box from YOLO output.
    
    Criteria:
    1. Confidence > threshold
    2. Box area not too small (min_area) and not too large (max_area)
    3. Box centered (proximity to image center)
    
    Fallback: If no good box found, return None → use crop fallback
    """
    h, w = img_shape[:2]
    img_area = h * w
    
    candidates = []
    
    for box in yolo_output:
        conf = box.conf
        x1, y1, x2, y2 = box.xyxy
        box_area = (x2 - x1) * (y2 - y1)
        
        # Confidence check
        if conf < confidence_threshold:
            continue
        
        # Area check: box should be 5%-95% of image
        area_ratio = box_area / img_area
        if area_ratio < 0.05 or area_ratio > 0.95:
            continue
        
        # Center proximity: box should be within center 70% of image
        box_center_x = (x1 + x2) / 2
        box_center_y = (y1 + y2) / 2
        img_center_x, img_center_y = w / 2, h / 2
        
        dist_to_center = ((box_center_x - img_center_x)**2 + 
                         (box_center_y - img_center_y)**2)**0.5
        max_dist = (w**2 + h**2)**0.5 / 3  # 30% tolerance
        
        if dist_to_center > max_dist:
            continue
        
        # Score: high confidence + centered
        score = conf - 0.1 * (dist_to_center / max_dist)
        candidates.append((score, box))
    
    if not candidates:
        return None  # Fallback
    
    # Return best box
    return max(candidates, key=lambda x: x[0])[1]
```

### Step 4: Image Cropping

```python
def crop_image(img, yolo_box, padding=10):
    """
    Crop image to bounding box with optional padding.
    """
    x1, y1, x2, y2 = yolo_box.xyxy
    
    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(img.shape[1], x2 + padding)
    y2 = min(img.shape[0], y2 + padding)
    
    cropped = img[int(y1):int(y2), int(x1):int(x2)]
    return cropped

def get_crop_or_fallback(img, yolo_model):
    """
    Main function: detect product box and crop, with fallback strategies.
    """
    # Run YOLO
    results = yolo_model(img)
    yolo_box = choose_best_box(results[0], img.shape)
    
    if yolo_box is not None:
        # Crop to bounding box
        cropped = crop_image(img, yolo_box, padding=10)
        return cropped, "yolo"
    else:
        # Fallback: Center crop (not original image!)
        h, w = img.shape[:2]
        h_crop = int(h * 0.8)
        w_crop = int(w * 0.8)
        y_start = (h - h_crop) // 2
        x_start = (w - w_crop) // 2
        cropped = img[y_start:y_start+h_crop, x_start:x_start+w_crop]
        return cropped, "fallback_center_crop"
```

---

## Implementation Steps

### Phase 1: Setup & Data Preparation (Week 1)

#### 1.1 Collect & Annotate YOLO Training Data
```bash
# Allocate 500-1000 images for annotation
# Use LabelImg or CVAT
# Output: annotations in YOLO format (txt files)
```

#### 1.2 Prepare YOLO Dataset Structure
```
yolo_dataset/
├─ images/
│  ├─ train/       # 400 images
│  ├─ val/         # 50 images
│  └─ test/        # 50 images
├─ labels/
│  ├─ train/       # YOLO format .txt
│  ├─ val/
│  └─ test/
└─ dataset.yaml    # Dataset config
```

#### 1.3 Train YOLO Model
```python
from ultralytics import YOLO

model = YOLO('yolov8s.pt')  # or yolov11n.pt
results = model.train(
    data='path/to/dataset.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device=0
)

# Save best model
model.save('shopee_yolo_best.pt')

# Verify on test set
metrics = model.val()
print(f"✅ YOLO mAP@0.5: {metrics.box.map50:.4f}")
if metrics.box.map50 < 0.80:
    print("⚠️ WARNING: YOLO mAP < 0.80, consider re-training")
```

### Phase 2: Feature Extraction (Week 1-2)

#### 2.1 Load YOLO Model
```python
import cv2
import torch
from ultralytics import YOLO

yolo_model = YOLO('shopee_yolo_best.pt')
yolo_model.to('cuda:0')
```

#### 2.2 Crop All Gallery Images
```python
import os
import numpy as np
from tqdm import tqdm
from PIL import Image

CROPPED_DIR = '/content/gallery_cropped'
os.makedirs(CROPPED_DIR, exist_ok=True)

crop_stats = {'yolo': 0, 'fallback': 0, 'error': 0}

for idx, row in tqdm(df_gallery.iterrows(), total=len(df_gallery)):
    img_path = os.path.join(IMG_DIR, row['image'])
    crop_path = os.path.join(CROPPED_DIR, row['image'])
    
    try:
        img = cv2.imread(img_path)
        cropped, crop_type = get_crop_or_fallback(img, yolo_model)
        cv2.imwrite(crop_path, cropped)
        crop_stats[crop_type] += 1
    except Exception as e:
        crop_stats['error'] += 1
        # Fallback: copy original
        cv2.imwrite(crop_path, img)

print(f"Crop Stats: {crop_stats}")
print(f"✅ Success rate: {(crop_stats['yolo'] + crop_stats['fallback']) / len(df_gallery):.2%}")
```

#### 2.3 Extract DINOv2 Features on Cropped Images
```python
import torch
import torch.nn.functional as F
from torchvision import transforms

dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2 = dinov2.cuda().eval()

transform_dino = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@torch.no_grad()
def extract_dinov2_batch(crop_paths, batch_size=32):
    all_feats = []
    for i in range(0, len(crop_paths), batch_size):
        batch_paths = crop_paths[i:i+batch_size]
        imgs = torch.stack([
            transform_dino(Image.open(p).convert('RGB')) 
            for p in batch_paths
        ]).cuda()
        
        feats = dinov2.forward_features(imgs)['x_norm_clstoken']
        feats = F.normalize(feats, dim=-1)  # L2 normalization
        all_feats.append(feats.cpu().numpy())
    
    return np.vstack(all_feats)

gallery_dino_crop = extract_dinov2_batch(
    [os.path.join(CROPPED_DIR, f) for f in df_gallery['image']]
)
np.save('/content/features/gallery_dino_crop.npy', gallery_dino_crop)
print(f"✅ Gallery DINOv2-crop shape: {gallery_dino_crop.shape}")
```

### Phase 3: Hyperparameter Tuning (Week 2-3)

See next section: [Hyperparameter Optimization](#hyperparameter-optimization)

### Phase 4: Evaluation (Week 3)

See: [Inference Pipeline](#inference-pipeline)

---

## Hyperparameter Optimization

### Strategy: Sequential Grid Search

**Why sequential instead of joint?**
- Joint 3D grid: 10 × 5 × 8 = 400 evaluations (~8 hours)
- Sequential: 10 + 5 + 8 = 23 evaluations (~1 hour)

### Step 1: Grid Search Alpha (MobileCLIP only)

**Why on MobileCLIP alone?**
- DINOv2 is not affected by alpha
- This step is identical to baseline pipeline

```python
best_alpha = 0.5
best_map5 = 0.0

for alpha in np.arange(0.0, 1.05, 0.1):
    val_metrics = evaluate_stage1_only(
        query_df=df_val,
        gallery_df=df_gallery,
        alpha=alpha,
        K=5
    )
    print(f"alpha={alpha:.1f} → mAP@5={val_metrics['mAP@5']:.4f}")
    
    if val_metrics['mAP@5'] > best_map5:
        best_map5 = val_metrics['mAP@5']
        best_alpha = alpha

print(f"\n✅ Best alpha: {best_alpha:.1f} (mAP@5={best_map5:.4f})")
```

### Step 2: Grid Search Retrieval_K (with DINOv2 only, beta=1.0)

**Why beta=1.0?**
- Isolate the effect of retrieval_k
- Later we'll tune beta to blend MobileCLIP scores

```python
best_k = 100
best_map5_k = 0.0

for k in [50, 100, 150, 200]:
    val_metrics = evaluate_two_stage(
        query_df=df_val,
        gallery_df=df_gallery,
        alpha=best_alpha,
        retrieval_k=k,
        beta=1.0,  # DINOv2 only
        K=5
    )
    print(f"retrieval_k={k:3d} → mAP@5={val_metrics['mAP@5']:.4f}")
    
    if val_metrics['mAP@5'] > best_map5_k:
        best_map5_k = val_metrics['mAP@5']
        best_k = k

print(f"\n✅ Best retrieval_k: {best_k} (mAP@5={best_map5_k:.4f})")
```

### Step 3: Grid Search Beta (full pipeline)

**Why now?**
- Both alpha and k are fixed
- Only tune the blend between DINOv2 and MobileCLIP

```python
best_beta = 0.3
best_map5_beta = 0.0

for beta in np.arange(0.2, 0.6, 0.05):
    val_metrics = evaluate_two_stage(
        query_df=df_val,
        gallery_df=df_gallery,
        alpha=best_alpha,
        retrieval_k=best_k,
        beta=beta,
        K=5
    )
    print(f"beta={beta:.2f} → mAP@5={val_metrics['mAP@5']:.4f}")
    
    if val_metrics['mAP@5'] > best_map5_beta:
        best_map5_beta = val_metrics['mAP@5']
        best_beta = beta

print(f"\n✅ Best beta: {best_beta:.2f} (mAP@5={best_map5_beta:.4f})")
```

### Results Storage

```python
# Save hyperparameters
hyperparam_config = {
    'alpha': float(best_alpha),
    'retrieval_k': int(best_k),
    'beta': float(best_beta),
    'val_mAP@5': float(best_map5_beta),
}

import json
with open('/content/features/best_hyperparams.json', 'w') as f:
    json.dump(hyperparam_config, f, indent=2)

print("✅ Hyperparameters saved!")
```

---

## Inference Pipeline

### Single Query Inference Function

```python
@torch.no_grad()
def search_two_stage_improved(
    query_image_path,
    query_title,
    alpha,
    retrieval_k,
    beta,
    final_k=5
):
    """
    End-to-end 2-stage search.
    
    Args:
        query_image_path: Path to query image (original, not cropped)
        query_title: Product title
        alpha, retrieval_k, beta: Hyperparameters
        final_k: Number of final results to return
    
    Returns:
        top_indices: Indices of top-k results in gallery
        top_scores: Final scores
    """
    
    # ──── PREPROCESSING ────────────────────────────────────────────────────
    query_img = cv2.imread(query_image_path)
    query_img_rgb = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    
    # Crop query image using YOLO
    query_cropped, crop_type = get_crop_or_fallback(query_img, yolo_model)
    
    # ──── STAGE 1: MobileCLIP RETRIEVAL ────────────────────────────────────
    
    # Feature extraction (original image)
    query_img_tensor = preprocess(Image.fromarray(query_img_rgb))
    with torch.no_grad():
        q_img_feat = clip_model.encode_image(query_img_tensor.unsqueeze(0).cuda())
    
    # Text feature
    q_txt_feat = clip_model.encode_text(tokenizer([query_title]).cuda())
    q_txt_feat = q_txt_feat / q_txt_feat.norm(dim=-1, keepdim=True)
    
    # Fusion
    q_img_feat = q_img_feat.cpu().numpy().flatten()
    q_txt_feat = q_txt_feat.cpu().numpy().flatten()
    q_fused = normalize(alpha * q_img_feat + (1-alpha) * q_txt_feat)
    
    # FAISS search
    distances, indices = faiss_index.search(
        q_fused.reshape(1, -1), 
        retrieval_k
    )
    
    clip_scores = distances[0]  # Save for later fusion
    
    # ──── STAGE 2: DINOv2-CROP RE-RANKING ─────────────────────────────────
    
    # Extract DINOv2 feature from cropped query
    q_crop_tensor = transform_dino(Image.fromarray(query_cropped)).unsqueeze(0).cuda()
    q_dino_feat = dinov2.forward_features(q_crop_tensor)['x_norm_clstoken']
    q_dino_feat = F.normalize(q_dino_feat, dim=-1)
    q_dino_feat = q_dino_feat.cpu().numpy().flatten()
    
    # Rerank top-K candidates
    final_scores = []
    for idx in indices[0]:
        dino_feat = gallery_dino_crop[idx]
        dino_score = np.dot(q_dino_feat, dino_feat)  # cosine similarity
        
        # Normalize scores before fusion (IMPORTANT!)
        clip_score_norm = (clip_scores[idx] - clip_scores.min()) / (clip_scores.max() - clip_scores.min() + 1e-8)
        dino_score_norm = (dino_score - gallery_dino_crop_scores.min()) / (gallery_dino_crop_scores.max() - gallery_dino_crop_scores.min() + 1e-8)
        
        final_score = beta * dino_score_norm + (1-beta) * clip_score_norm
        final_scores.append(final_score)
    
    # Sort and return top-5
    final_scores = np.array(final_scores)
    top_5_local_indices = np.argsort(-final_scores)[:final_k]
    top_indices = indices[0][top_5_local_indices]
    top_scores = final_scores[top_5_local_indices]
    
    return top_indices, top_scores
```

### Batch Inference (for evaluation)

```python
def evaluate_two_stage(query_df, gallery_df, alpha, retrieval_k, beta, K=5):
    """
    Evaluate 2-stage pipeline on multiple queries.
    """
    from sklearn.metrics import average_precision_score
    
    gallery_pids = gallery_df['posting_id'].tolist()
    gt_dict = get_ground_truth_dict(gallery_df)
    
    ap_list, p1_list, r5_list = [], [], []
    
    for _, row in tqdm(query_df.iterrows(), total=len(query_df)):
        query_id = row['posting_id']
        query_img_path = os.path.join(IMG_DIR, row['image'])
        query_title = row['title']
        
        # Get ground truth
        relevant = gt_dict.get(query_id, set()) - {query_id}
        if not relevant:
            continue
        
        try:
            # Inference
            top_indices, top_scores = search_two_stage_improved(
                query_img_path, query_title,
                alpha=alpha, retrieval_k=retrieval_k, beta=beta,
                final_k=K
            )
            
            # Convert indices to posting_ids
            retrieved_pids = [gallery_pids[idx] for idx in top_indices]
            
            # Compute metrics
            hits, ap = 0, 0.0
            for rank, pid in enumerate(retrieved_pids, 1):
                if pid in relevant:
                    hits += 1
                    ap += hits / rank
            
            ap_list.append(ap / min(len(relevant), K))
            p1_list.append(1.0 if retrieved_pids and retrieved_pids[0] in relevant else 0.0)
            r5_list.append(len(set(retrieved_pids) & relevant) / len(relevant))
            
        except Exception as e:
            print(f"Error on query {query_id}: {e}")
            continue
    
    return {
        'mAP@5': float(np.mean(ap_list)),
        'Precision@1': float(np.mean(p1_list)),
        'Recall@5': float(np.mean(r5_list)),
    }
```

---

## Code Examples

### Complete Training Pipeline

```python
# ============================================================================
# COMPLETE PIPELINE: YOLO TRAINING + FEATURE EXTRACTION + TUNING + EVAL
# ============================================================================

import os
import json
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from torchvision import transforms
from tqdm import tqdm

# ──── 1. TRAIN YOLO ────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: YOLO TRAINING")
print("=" * 60)

yolo_model = YOLO('yolov8s.pt')
results = yolo_model.train(
    data='yolo_dataset.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    patience=10
)

yolo_best = YOLO('runs/detect/train/weights/best.pt')
metrics = yolo_best.val()
print(f"✅ YOLO Training complete. mAP@0.5: {metrics.box.map50:.4f}")

if metrics.box.map50 < 0.80:
    print("⚠️ WARNING: YOLO accuracy low (<0.80). Results may be suboptimal.")

# ──── 2. CROP ALL GALLERY IMAGES ───────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: CROP GALLERY IMAGES WITH YOLO")
print("=" * 60)

CROPPED_DIR = '/content/gallery_cropped'
os.makedirs(CROPPED_DIR, exist_ok=True)

crop_stats = {'yolo': 0, 'fallback': 0, 'error': 0}

for idx, row in tqdm(df_gallery.iterrows(), total=len(df_gallery)):
    img_path = os.path.join(IMG_DIR, row['image'])
    crop_path = os.path.join(CROPPED_DIR, row['image'])
    
    try:
        img = cv2.imread(img_path)
        cropped, crop_type = get_crop_or_fallback(img, yolo_best)
        cv2.imwrite(crop_path, cropped)
        crop_stats[crop_type] += 1
    except Exception as e:
        crop_stats['error'] += 1

print(f"Crop statistics: {crop_stats}")
print(f"Success rate: {(crop_stats['yolo'] + crop_stats['fallback']) / len(df_gallery):.2%}")

# ──── 3. EXTRACT DINOV2 FEATURES ───────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: EXTRACT DINOV2 FEATURES")
print("=" * 60)

# Clear VRAM
del clip_model
torch.cuda.empty_cache()

dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2 = dinov2.cuda().eval()

transform_dino = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@torch.no_grad()
def extract_dinov2_batch(crop_paths, batch_size=32):
    all_feats = []
    for i in range(0, len(crop_paths), batch_size):
        batch_paths = crop_paths[i:i+batch_size]
        imgs = torch.stack([
            transform_dino(Image.open(p).convert('RGB')) 
            for p in batch_paths
        ]).cuda()
        
        feats = dinov2.forward_features(imgs)['x_norm_clstoken']
        feats = F.normalize(feats, dim=-1)
        all_feats.append(feats.cpu().numpy())
    
    return np.vstack(all_feats)

crop_paths = [os.path.join(CROPPED_DIR, f) for f in df_gallery['image']]
gallery_dino_crop = extract_dinov2_batch(crop_paths)
np.save('/content/features/gallery_dino_crop.npy', gallery_dino_crop)

print(f"✅ Gallery DINOv2-crop shape: {gallery_dino_crop.shape}")

# ──── 4. GRID SEARCH (Alpha, K, Beta) ──────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: HYPERPARAMETER TUNING")
print("=" * 60)

# Load MobileCLIP back
del dinov2
torch.cuda.empty_cache()
clip_model = ... # reload

best_alpha = 0.5
for alpha in np.arange(0.0, 1.05, 0.1):
    metrics = evaluate_stage1_only(df_val, df_gallery, alpha=alpha, K=5)
    if metrics['mAP@5'] > best_map5_alpha:
        best_alpha = alpha
        best_map5_alpha = metrics['mAP@5']
    print(f"alpha={alpha:.1f} → mAP@5={metrics['mAP@5']:.4f}")

print(f"\n✅ Best alpha: {best_alpha:.1f}\n")

# Reload DINOv2
del clip_model
torch.cuda.empty_cache()
dinov2 = ...

best_k = 100
for k in [50, 100, 150, 200]:
    metrics = evaluate_two_stage(df_val, df_gallery, 
                                  alpha=best_alpha, retrieval_k=k, beta=1.0, K=5)
    if metrics['mAP@5'] > best_map5_k:
        best_k = k
        best_map5_k = metrics['mAP@5']
    print(f"retrieval_k={k:3d} → mAP@5={metrics['mAP@5']:.4f}")

print(f"\n✅ Best retrieval_k: {best_k}\n")

best_beta = 0.3
for beta in np.arange(0.2, 0.6, 0.05):
    metrics = evaluate_two_stage(df_val, df_gallery,
                                  alpha=best_alpha, retrieval_k=best_k, beta=beta, K=5)
    if metrics['mAP@5'] > best_map5_beta:
        best_beta = beta
        best_map5_beta = metrics['mAP@5']
    print(f"beta={beta:.2f} → mAP@5={metrics['mAP@5']:.4f}")

print(f"\n✅ Best beta: {best_beta:.2f}\n")

# ──── 5. TEST SET EVALUATION (RUN ONCE!) ───────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: FINAL TEST SET EVALUATION")
print("=" * 60)

test_metrics = evaluate_two_stage(df_test, df_gallery,
                                   alpha=best_alpha, retrieval_k=best_k, 
                                   beta=best_beta, K=5)

print(f"\n📊 FINAL RESULTS:")
print(f"   mAP@5        = {test_metrics['mAP@5']:.4f}")
print(f"   Precision@1  = {test_metrics['Precision@1']:.4f}")
print(f"   Recall@5     = {test_metrics['Recall@5']:.4f}")

# Save results
results = {
    'model': 'MobileCLIP + DINOv2-Crop',
    'hyperparams': {
        'alpha': float(best_alpha),
        'retrieval_k': int(best_k),
        'beta': float(best_beta)
    },
    'test_metrics': {
        'mAP@5': float(test_metrics['mAP@5']),
        'Precision@1': float(test_metrics['Precision@1']),
        'Recall@5': float(test_metrics['Recall@5']),
    }
}

with open('/content/results_improved_pipeline.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✅ PIPELINE COMPLETE!")
```

---

## Risk Analysis & Recommendations

### ⚠️ Risk 1: YOLO Detection Failure

**Symptom:** YOLO has low mAP (<0.80) on validation set.

**Impact:**
- Cropped images are incorrect → DINOv2 features are wrong
- Final results worse than baseline
- Expected performance gain lost

**Mitigation:**
```python
# 1. Check YOLO accuracy BEFORE using it
if yolo_val_map < 0.80:
    print("⚠️ YOLO accuracy too low. Consider:")
    print("   - Re-annotate training data")
    print("   - Use larger YOLO model (s/m instead of n)")
    print("   - Train for more epochs")
    print("   - Use better fallback strategy")
```

**Solution:** If YOLO fails, fall back to baseline (MobileCLIP only).

---

### ⚠️ Risk 2: Domain Mismatch on Fallback

**Symptom:** YOLO can't detect some products → fallback to center crop or original image.

**Problem:** DINOv2 is trained on YOLO crops, not fallback images → inconsistent input distribution.

**Mitigation:**
```python
def choose_crop_strategy(yolo_box, img_shape):
    """Always return a crop, never original image."""
    h, w = img_shape[:2]
    
    if yolo_box is not None:
        # Option 1: Use YOLO box
        return crop_to_box(img, yolo_box)
    else:
        # Option 2: Center crop (70-80% of image)
        h_crop = int(h * 0.75)
        w_crop = int(w * 0.75)
        y_start = (h - h_crop) // 2
        x_start = (w - w_crop) // 2
        return img[y_start:y_start+h_crop, x_start:x_start+w_crop]
```

---

### ⚠️ Risk 3: Score Fusion Imbalance

**Symptom:** final_score = beta * dino_score + (1-beta) * clip_score is imbalanced.

**Problem:** Scores from different models may have different ranges (e.g., [0,1] vs [-1,1]).

**Mitigation:**
```python
# Normalize scores before fusion
def fuse_scores_safe(dino_score, clip_score, beta, gallery_stats):
    """Normalize both scores to [0,1] before fusion."""
    
    # Normalize DINOv2
    dino_min, dino_max = gallery_stats['dino_min'], gallery_stats['dino_max']
    dino_norm = (dino_score - dino_min) / (dino_max - dino_min + 1e-8)
    
    # Normalize MobileCLIP
    clip_min, clip_max = gallery_stats['clip_min'], gallery_stats['clip_max']
    clip_norm = (clip_score - clip_min) / (clip_max - clip_min + 1e-8)
    
    # Clamp to [0,1]
    dino_norm = np.clip(dino_norm, 0, 1)
    clip_norm = np.clip(clip_norm, 0, 1)
    
    return beta * dino_norm + (1-beta) * clip_norm

# Compute gallery statistics once (during training)
def compute_gallery_stats():
    all_dino_scores = []
    all_clip_scores = []
    
    for idx in range(len(gallery_dino_crop)):
        # Compute all pairwise scores
        dino_scores = np.dot(gallery_dino_crop, gallery_dino_crop[idx])
        all_dino_scores.extend(dino_scores)
        # Similar for clip_scores...
    
    return {
        'dino_min': np.min(all_dino_scores),
        'dino_max': np.max(all_dino_scores),
        'clip_min': np.min(all_clip_scores),
        'clip_max': np.max(all_clip_scores),
    }
```

---

### ⚠️ Risk 4: Inference Latency

**Symptom:** Pipeline takes 2-3 seconds per query (vs 500ms for baseline).

**Impact:**
- Not suitable for real-time applications
- High computational cost for production
- Mobile/edge deployment not feasible

**Mitigation:**
```python
# Measure latency
import time

def benchmark_pipeline():
    times = {}
    
    # YOLO inference
    t0 = time.time()
    yolo_box = get_crop_or_fallback(img, yolo_model)
    times['yolo'] = time.time() - t0
    
    # MobileCLIP
    t0 = time.time()
    q_img_feat = clip_model.encode_image(img)
    times['mobileclip'] = time.time() - t0
    
    # FAISS search
    t0 = time.time()
    _, indices = faiss_index.search(q_fused, 100)
    times['faiss'] = time.time() - t0
    
    # DINOv2 + rerank
    t0 = time.time()
    q_dino = dinov2(cropped_img)
    for idx in indices:
        dino_score = cosine_sim(q_dino, gallery_dino[idx])
    times['dinov2'] = time.time() - t0
    
    print(f"Latency breakdown:")
    for name, duration in times.items():
        print(f"  {name:15s}: {duration*1000:6.1f}ms")
    print(f"  {'Total':15s}: {sum(times.values())*1000:6.1f}ms")

benchmark_pipeline()
```

**Solution Options:**
1. **Reduce K:** Use K=50 instead of K=100 → faster reranking
2. **Quantization:** Quantize models to int8 (PyTorch QINT8)
3. **Distillation:** Train smaller DINOv2 on cropped images
4. **Batch inference:** Process multiple queries at once

---

### 📋 Checklist Before Deployment

```
Pre-Deployment Checklist:
─────────────────────────────────────

□ YOLO Training
  □ Collected 500-1000 annotated images (separate from gallery)
  □ YOLO mAP@0.5 > 0.85 on validation set
  □ Tested on edge cases (small objects, cluttered scenes)

□ Feature Extraction
  □ Gallery DINOv2-crop features cached (gallery_dino_crop.npy)
  □ Verified feature dimensions (N, 384) match gallery size
  □ No NaN/Inf values in features

□ Hyperparameter Tuning
  □ Grid search completed on validation set only
  □ Test set run exactly once (never re-run!)
  □ Results saved to JSON file

□ Inference Pipeline
  □ Single-query function works end-to-end
  □ Batch evaluation function works
  □ Latency benchmarked (target < 3 seconds)
  □ Error handling for edge cases (corrupt images, YOLO fail)

□ Validation
  □ Compare results vs baseline MobileCLIP
  □ mAP improvement > 1% to justify added complexity
  □ No data leakage between YOLO train/test sets
```

---

## Performance Comparison

### Expected Improvements

| Metric | Baseline (MobileCLIP) | Improved (+ DINOv2-Crop) | Improvement |
|---|---|---|---|
| **mAP@5** | 0.7708 | ~0.79-0.80 | +2-3% |
| **Precision@1** | 0.7937 | ~0.81-0.82 | +1-2% |
| **Recall@5** | 0.7430 | ~0.76-0.77 | +1-2% |
| **Latency** | ~500ms | ~2000ms | ⚠️ 4× slower |
| **Model Size** | ~150MB | ~400MB | 2.6× larger |
| **VRAM (Inference)** | ~2GB | ~4GB | 2× more |

### When Pipeline Fails (YOLO mAP < 0.80)

| Scenario | Expected Degradation |
|---|---|
| YOLO mAP = 0.70 | MobileCLIP performance ≈ baseline (-0-1%) |
| YOLO mAP = 0.60 | DINOv2 on wrong crops → performance -1-3% |
| YOLO mAP = 0.50 | Serious degradation → better use baseline |

**Recommendation:** Only deploy if YOLO mAP > 0.85.

---

## Summary

### ✅ When to Use This Pipeline
- ✅ You have accurate YOLO model (mAP > 0.85)
- ✅ Inference latency budget allows 2-3 seconds
- ✅ Extra 2-3% mAP improvement is critical for your use case
- ✅ Computational resources available (GPU for inference)

### ❌ When to Stick with Baseline
- ❌ YOLO mAP < 0.80
- ❌ Real-time constraints (<500ms required)
- ❌ Limited computational resources
- ❌ Baseline already meets performance requirements

### 🎯 Recommended Path Forward

```
Week 1-2: Annotation + YOLO Training
  └─ Goal: Achieve YOLO mAP > 0.85

Week 2-3: Feature Extraction + Hyperparameter Tuning
  └─ Goal: Find best (alpha, k, beta)

Week 3: Evaluation + Benchmarking
  └─ Goal: Verify mAP improvement > 1%

Week 4: Decision
  ├─ If mAP > 1%: Deploy improved pipeline
  └─ If mAP < 1%: Keep baseline, optimize elsewhere
```

---

## References

- [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
- [MobileCLIP: Fast Image-Text Models through Text-Efficient Training](https://arxiv.org/abs/2311.17049)
- [YOLOv8: A Fast SOTA Object Detector](https://github.com/ultralytics/ultralytics)
- [Shopee Product Matching Dataset](https://www.kaggle.com/competitions/shopee-product-matching)

---

**Document Version:** 1.0  
**Last Updated:** 2026-06-03  
**Status:** Ready for Implementation