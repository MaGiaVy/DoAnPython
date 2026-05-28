# HƯỚNG DẪN: Pipeline Tuần 5 – rembg + MobileCLIP + EfficientNetB0 + MiniLM

**Người thực hiện:** Mã Gia Vỹ  
**Môi trường:** Google Colab (GPU T4)  
**File output:** `Tuan5_GiaVy_Pipeline.ipynb`

---

## BỐI CẢNH

Pipeline tuần 4: ResNet50 + TF-IDF + pHash + Reranking → mAP = 0.7635

**Cải tiến tuần 5:**

- Thêm **rembg** xóa nền trước khi trích vector → vector không bị nhiễu nền/chữ quảng cáo
- **Giai đoạn 1:** MobileCLIP (s1) encode cả ảnh + title → FAISS top-100
- **Giai đoạn 2:** EfficientNetB0 (image) + paraphrase-multilingual-MiniLM-L12-v2 (text) rerank top-100 → top-K
- **pHash và alpha chưa dùng** — để tuần sau

---

## PIPELINE

```
Input image + title
    ↓
rembg → xóa nền, chỉ giữ sản phẩm
    ↓
━━━━━━━━ GIAI ĐOẠN 1 ━━━━━━━━
MobileCLIP s1:
  - encode ảnh đã xóa nền → image vector 512 chiều
  - encode title → text vector 512 chiều
  - concat + L2-normalize → fusion vector 1024 chiều
FAISS IndexFlatIP → top-100 candidate
    ↓
━━━━━━━━ GIAI ĐOẠN 2 ━━━━━━━━
EfficientNetB0: tính image similarity query vs top-100
MiniLM multilingual: tính text similarity query_title vs top-100 titles
final_score = 0.5 × effnet_score + 0.5 × minilm_score
Rerank top-100 → top-K
    ↓
Kết quả cuối
```

---

## YÊU CẦU NOTEBOOK

Tạo file `Tuan5_GiaVy_Pipeline.ipynb` với **11 cell** theo thứ tự:

---

### CELL 0 – Cài thư viện

```python
!pip install -q rembg onnxruntime
!pip install -q git+https://github.com/apple/ml-mobileclip.git
!pip install -q sentence-transformers
!pip install -q timm
print('Cài xong!')
```

---

### CELL 1 – Import thư viện

```python
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# rembg
from rembg import remove, new_session

# MobileCLIP
import mobileclip

# EfficientNetB0
import timm
from torchvision import transforms

# MiniLM
from sentence_transformers import SentenceTransformer

# FAISS
import faiss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
print('Import thành công!')
```

---

### CELL 2 – Mount Drive + Đường dẫn

```python
from google.colab import drive
drive.mount('/content/drive')

# TODO: Chỉnh lại đường dẫn cho đúng với Drive của nhóm
BASE_DIR  = '/content/drive/MyDrive/DuLieuPython'
CSV_PATH  = os.path.join(BASE_DIR, 'train.csv')
IMAGE_DIR = os.path.join(BASE_DIR, 'train_images')
PROCESSED = os.path.join(BASE_DIR, 'processed_tuan5')
RESULTS   = os.path.join(BASE_DIR, 'results_tuan5')

os.makedirs(PROCESSED, exist_ok=True)
os.makedirs(RESULTS,   exist_ok=True)

for p in [CSV_PATH, IMAGE_DIR]:
    status = '✅' if os.path.exists(p) else '❌ KHÔNG TÌM THẤY'
    print(f'{status}  {p}')
```

---

### CELL 3 – Load dataset + Chia val/test

```python
df_full      = pd.read_csv(CSV_PATH)
label_counts = df_full['label_group'].value_counts()

# Lọc bỏ nhóm chỉ có 1 ảnh
nhom_du      = label_counts[label_counts >= 2].index
candidate_df = df_full[df_full['label_group'].isin(nhom_du)].reset_index(drop=True)

print(f'Tổng ảnh sau lọc: {len(candidate_df):,}')
print(f'Số nhóm          : {candidate_df["label_group"].nunique():,}')

# Chia val (20%) / test (80%) — không stratify vì nhiều nhóm chỉ có 2 ảnh
val_idx, test_idx = train_test_split(
    candidate_df.index.tolist(),
    test_size=0.8,
    random_state=42
)

# Lưu split để dùng lại
with open(os.path.join(PROCESSED, 'split_indices.json'), 'w') as f:
    json.dump({'val_idx': val_idx, 'test_idx': test_idx}, f)

print(f'\nGallery   : {len(candidate_df):,} ảnh (100%)')
print(f'Validation: {len(val_idx):,} ảnh (20%) → tuning')
print(f'Test      : {len(test_idx):,} ảnh (80%) → báo kết quả cuối')
```

---

### CELL 4 – rembg xóa nền + Cache

```python
# rembg xóa nền trắng/đơn sắc, giữ lại sản phẩm chính
# Lần đầu chạy mất ~2-3 tiếng (34k ảnh × 0.3s/ảnh)
# Lần sau load cache PNG, không cần chạy lại

REMBG_DIR = os.path.join(PROCESSED, 'rembg_crops')
os.makedirs(REMBG_DIR, exist_ok=True)

# Khởi tạo rembg session (dùng u2net model)
rembg_session = new_session('u2net')

def remove_bg(img_path, out_dir, session):
    """
    Xóa nền ảnh, lưu ra file PNG.
    Nếu đã có file cache thì load lại, không xử lý lại.
    Trả về PIL Image RGB (nền trắng thay cho nền trong suốt).
    """
    img_name  = os.path.splitext(os.path.basename(img_path))[0]
    out_path  = os.path.join(out_dir, img_name + '.png')

    if os.path.exists(out_path):
        # Load từ cache
        img = Image.open(out_path).convert('RGBA')
    else:
        try:
            img = Image.open(img_path).convert('RGB')
            img = remove(img, session=session)  # trả về RGBA
            img.save(out_path)
        except:
            # Fallback: dùng ảnh gốc nếu rembg lỗi
            img = Image.open(img_path).convert('RGBA')

    # Đặt nền trắng thay cho trong suốt
    background = Image.new('RGB', img.size, (255, 255, 255))
    background.paste(img, mask=img.split()[3])  # dùng alpha channel làm mask
    return background

# Chạy rembg cho toàn bộ dataset
print(f'Đang xóa nền {len(candidate_df):,} ảnh...')
print('LƯU Ý: Lần đầu mất ~2-3 tiếng. Lần sau load cache ngay!')

n_cached = sum(
    1 for img in candidate_df['image']
    if os.path.exists(os.path.join(REMBG_DIR,
       os.path.splitext(img)[0] + '.png'))
)
print(f'Đã có cache: {n_cached:,}/{len(candidate_df):,} ảnh')

if n_cached < len(candidate_df):
    for _, row in tqdm(candidate_df.iterrows(),
                       total=len(candidate_df),
                       desc='rembg'):
        img_path = os.path.join(IMAGE_DIR, row['image'])
        remove_bg(img_path, REMBG_DIR, rembg_session)

print('rembg xong!')

# Test thử 1 ảnh
test_path  = os.path.join(IMAGE_DIR, candidate_df['image'].iloc[0])
test_clean = remove_bg(test_path, REMBG_DIR, rembg_session)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(Image.open(test_path).convert('RGB'))
axes[0].set_title('Ảnh gốc'); axes[0].axis('off')
axes[1].imshow(test_clean)
axes[1].set_title('Sau rembg'); axes[1].axis('off')
plt.tight_layout(); plt.show()
```

---

### CELL 5 – Giai đoạn 1: MobileCLIP trích vector

```python
# MobileCLIP s1: encode ảnh đã xóa nền + title
# Vector: image 512 + text 512 → concat → 1024 chiều → L2-normalize
# Cache vào .npy để không chạy lại

MOBILECLIP_PATH = os.path.join(PROCESSED, 'mobileclip_fusion.npy')

if os.path.exists(MOBILECLIP_PATH):
    fusion_feats = np.load(MOBILECLIP_PATH).astype('float32')
    print(f'Load MobileCLIP features: {fusion_feats.shape}')
else:
    print('Trích xuất MobileCLIP features (~20-30 phút)...')

    # Load MobileCLIP s1
    mc_model, _, mc_preprocess = mobileclip.create_model_and_transforms(
        'mobileclip_s1',
        pretrained='datacompdr'  # checkpoint tốt nhất
    )
    mc_model = mc_model.to(device).eval()
    mc_tokenizer = mobileclip.get_tokenizer('mobileclip_s1')

    all_img_vecs = []
    all_txt_vecs = []

    BATCH = 64
    titles = candidate_df['title'].fillna('').astype(str).tolist()

    with torch.inference_mode():
        # Trích image features
        for start in tqdm(range(0, len(candidate_df), BATCH),
                          desc='MobileCLIP image'):
            batch_imgs = []
            for i in range(start, min(start+BATCH, len(candidate_df))):
                img_name = candidate_df['image'].iloc[i]
                img_path = os.path.join(IMAGE_DIR, img_name)
                img = remove_bg(img_path, REMBG_DIR, rembg_session)
                batch_imgs.append(mc_preprocess(img))
            batch_tensor = torch.stack(batch_imgs).to(device)
            vecs = mc_model.encode_image(batch_tensor)
            all_img_vecs.append(vecs.cpu().float().numpy())

        # Trích text features
        for start in tqdm(range(0, len(titles), BATCH),
                          desc='MobileCLIP text'):
            batch_titles = titles[start:start+BATCH]
            tokens = mc_tokenizer(batch_titles).to(device)
            vecs   = mc_model.encode_text(tokens)
            all_txt_vecs.append(vecs.cpu().float().numpy())

    img_feats = np.vstack(all_img_vecs).astype('float32')  # (N, 512)
    txt_feats = np.vstack(all_txt_vecs).astype('float32')  # (N, 512)

    # L2-normalize từng loại
    img_norm = img_feats / (np.linalg.norm(img_feats, axis=1, keepdims=True) + 1e-8)
    txt_norm = txt_feats / (np.linalg.norm(txt_feats, axis=1, keepdims=True) + 1e-8)

    # Concat → fusion 1024 chiều → L2-normalize lại
    fusion = np.concatenate([img_norm, txt_norm], axis=1)  # (N, 1024)
    fusion_feats = fusion / (np.linalg.norm(fusion, axis=1, keepdims=True) + 1e-8)
    fusion_feats = fusion_feats.astype('float32')

    np.save(MOBILECLIP_PATH, fusion_feats)
    print(f'Đã lưu MobileCLIP features: {fusion_feats.shape}')

# Tạo FAISS index
print('Tạo FAISS index...')
dim   = fusion_feats.shape[1]  # 1024
index = faiss.IndexFlatIP(dim)
if torch.cuda.is_available():
    res   = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)
index.add(fusion_feats)
print(f'FAISS index: {index.ntotal:,} vectors, dim={dim}')
```

---

### CELL 6 – Giai đoạn 2A: EfficientNetB0 image features

```python
# EfficientNetB0 trích vector ảnh đã xóa nền
# Dùng để rerank top-100 từ giai đoạn 1
EFFNET_PATH = os.path.join(PROCESSED, 'effnet_features.npy')

if os.path.exists(EFFNET_PATH):
    effnet_feats = np.load(EFFNET_PATH).astype('float32')
    print(f'Load EfficientNetB0 features: {effnet_feats.shape}')
else:
    print('Trích xuất EfficientNetB0 features (~15-20 phút)...')

    # Load EfficientNetB0, bỏ lớp classification
    effnet = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
    effnet = effnet.to(device).eval()

    # Lấy transform chuẩn của EfficientNetB0
    data_config = timm.data.resolve_model_data_config(effnet)
    effnet_transform = timm.data.create_transform(**data_config, is_training=False)

    all_vecs = []
    BATCH    = 128

    with torch.inference_mode():
        for start in tqdm(range(0, len(candidate_df), BATCH),
                          desc='EfficientNetB0'):
            batch_imgs = []
            for i in range(start, min(start+BATCH, len(candidate_df))):
                img_name = candidate_df['image'].iloc[i]
                img_path = os.path.join(IMAGE_DIR, img_name)
                img = remove_bg(img_path, REMBG_DIR, rembg_session)
                batch_imgs.append(effnet_transform(img))
            batch_tensor = torch.stack(batch_imgs).to(device)
            vecs = effnet(batch_tensor)
            all_vecs.append(vecs.cpu().float().numpy())

    effnet_feats = np.vstack(all_vecs).astype('float32')
    effnet_feats = effnet_feats / (
        np.linalg.norm(effnet_feats, axis=1, keepdims=True) + 1e-8)

    np.save(EFFNET_PATH, effnet_feats)
    print(f'Đã lưu EfficientNetB0 features: {effnet_feats.shape}')
```

---

### CELL 7 – Giai đoạn 2B: MiniLM text features

```python
# paraphrase-multilingual-MiniLM-L12-v2
# Hỗ trợ 50+ ngôn ngữ, phù hợp Shopee Đông Nam Á
MINILM_PATH = os.path.join(PROCESSED, 'minilm_features.npy')

if os.path.exists(MINILM_PATH):
    minilm_feats = np.load(MINILM_PATH).astype('float32')
    print(f'Load MiniLM features: {minilm_feats.shape}')
else:
    print('Trích xuất MiniLM features (~10-15 phút)...')

    minilm_model = SentenceTransformer(
        'paraphrase-multilingual-MiniLM-L12-v2',
        device=str(device)
    )

    titles = candidate_df['title'].fillna('').astype(str).tolist()
    minilm_feats = minilm_model.encode(
        titles,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True  # L2-normalize luôn
    ).astype('float32')

    np.save(MINILM_PATH, minilm_feats)
    print(f'Đã lưu MiniLM features: {minilm_feats.shape}')
```

---

### CELL 8 – Hàm rerank + Tính metric

```python
def rerank_top100(query_idx, top100_indices, effnet_feats, minilm_feats, labels):
    """
    Rerank top-100 từ giai đoạn 1 bằng EfficientNetB0 + MiniLM.
    Trả về top-K sau khi rerank.
    """
    # Loại chính nó
    cands = [j for j in top100_indices if j != query_idx][:100]
    if not cands:
        return []

    cands = np.array(cands)

    # Image score (EfficientNetB0)
    q_img  = effnet_feats[query_idx]
    c_imgs = effnet_feats[cands]
    img_scores = c_imgs @ q_img  # cosine similarity (đã normalize)

    # Text score (MiniLM)
    q_txt  = minilm_feats[query_idx]
    c_txts = minilm_feats[cands]
    txt_scores = c_txts @ q_txt

    # Kết hợp: 50% image + 50% text
    # pHash và alpha sẽ thêm sau
    final_scores = 0.5 * img_scores + 0.5 * txt_scores

    # Sắp xếp theo điểm giảm dần
    order = np.argsort(-final_scores)
    return cands[order].tolist()


def compute_metrics(query_indices, candidate_df, all_top100_I,
                    effnet_feats, minilm_feats, labels, label_counts,
                    K_LIST=[1, 3, 5, 10]):
    """Tính Precision@K, Recall@K, mAP@5 trên tập query."""
    rows   = []
    all_ap = []

    for q_pos, i in enumerate(tqdm(query_indices, desc='Tính metric')):
        q_label    = labels[i]
        n_relevant = label_counts.get(q_label, 0) - 1
        if n_relevant <= 0:
            continue

        # Rerank top-100 từ giai đoạn 1
        top100 = all_top100_I[q_pos]
        ranked = rerank_top100(i, top100, effnet_feats, minilm_feats, labels)
        if not ranked:
            continue

        # AP@5
        hits5 = [1 if labels[j] == q_label else 0 for j in ranked[:5]]
        ap, h = 0.0, 0
        for rank, hit in enumerate(hits5, 1):
            if hit:
                h  += 1
                ap += h / rank
        ap /= min(5, n_relevant)
        all_ap.append(ap)

        row = {'image': candidate_df['image'].iloc[i], 'label_group': q_label}
        for k in K_LIST:
            h_k = sum(1 for j in ranked[:k] if labels[j] == q_label)
            row[f'Precision@{k}'] = round(h_k / k, 4)
            row[f'Recall@{k}']    = round(h_k / min(n_relevant, k), 4)
        rows.append(row)

    detail_df = pd.DataFrame(rows)
    mAP       = round(float(np.mean(all_ap)), 4) if all_ap else 0.0
    return detail_df, mAP
```

---

### CELL 9 – FAISS Search + Đánh giá Test Set

```python
labels       = candidate_df['label_group'].values
label_counts = candidate_df['label_group'].value_counts().to_dict()
BATCH        = 512
TOP_100      = 101  # lấy 101 để loại chính nó còn 100

# FAISS batch search toàn bộ test set
print(f'FAISS search trên test set ({len(test_idx):,} query)...')
test_feats  = fusion_feats[test_idx]
all_I_test  = []

for s in tqdm(range(0, len(test_idx), BATCH), desc='FAISS batch'):
    batch    = test_feats[s:s+BATCH]
    _, I     = index.search(batch.astype('float32'), TOP_100)
    all_I_test.extend(I.tolist())

# Tính metric với reranking giai đoạn 2
print('Reranking + tính metric...')
detail_df, final_mAP = compute_metrics(
    test_idx, candidate_df, all_I_test,
    effnet_feats, minilm_feats, labels, label_counts
)

# In kết quả
K_LIST = [1, 3, 5, 10]
print('\n=== KẾT QUẢ CUỐI — rembg + MobileCLIP + EfficientNetB0 + MiniLM ===')
print(f'Tập đánh giá : Test set ({len(test_idx):,} query, 80%)')
print(f'Giai đoạn 1  : MobileCLIP s1 (ảnh đã xóa nền + title)')
print(f'Giai đoạn 2  : EfficientNetB0 (50%) + MiniLM multilingual (50%)')
print(f'pHash/alpha  : Chưa áp dụng (tuần sau)')

summary = []
for k in K_LIST:
    p = round(detail_df[f'Precision@{k}'].mean(), 4)
    r = round(detail_df[f'Recall@{k}'].mean(), 4)
    summary.append({'K': k, 'Precision@K': p, 'Recall@K': r, 'mAP': final_mAP})
    print(f'  K={k:2d}  Precision@K={p:.4f}  Recall@K={r:.4f}')
print(f'  mAP@5 = {final_mAP:.4f}')

# Lưu CSV
metrics_df = pd.DataFrame(summary)
metrics_df['method'] = 'rembg+MobileCLIP_s1+EffNetB0+MiniLM'
metrics_df.to_csv(os.path.join(RESULTS, 'metrics_tuan5.csv'), index=False)
detail_df.to_csv(os.path.join(RESULTS, 'detail_tuan5.csv'), index=False)
print('\nĐã lưu metrics_tuan5.csv và detail_tuan5.csv!')
```

---

### CELL 10 – Visualize kết quả

```python
def visualize(query_idx, candidate_df, image_dir, rembg_dir,
              fusion_feats, index, effnet_feats, minilm_feats, labels, k=5):
    """Hiển thị: ảnh gốc | ảnh xóa nền | top-K kết quả."""
    img_path   = os.path.join(image_dir, candidate_df['image'].iloc[query_idx])
    clean_img  = remove_bg(img_path, rembg_dir, rembg_session)
    query_label = labels[query_idx]

    # FAISS search
    q_feat = fusion_feats[query_idx:query_idx+1]
    _, I   = index.search(q_feat, 101)
    top100 = I[0].tolist()

    # Rerank
    ranked = rerank_top100(query_idx, top100,
                           effnet_feats, minilm_feats, labels)[:k]

    fig, axes = plt.subplots(1, k+2, figsize=(3*(k+2), 4))

    axes[0].imshow(Image.open(img_path).convert('RGB'))
    axes[0].set_title('Gốc', fontsize=9); axes[0].axis('off')

    axes[1].imshow(clean_img)
    axes[1].set_title('rembg', fontsize=9); axes[1].axis('off')

    for rank, j in enumerate(ranked):
        r_path  = os.path.join(image_dir, candidate_df['image'].iloc[j])
        correct = labels[j] == query_label
        axes[rank+2].imshow(Image.open(r_path).convert('RGB'))
        axes[rank+2].set_title(
            f'Top-{rank+1}\n{"✓" if correct else "✗"}',
            color='green' if correct else 'red', fontsize=9)
        axes[rank+2].axis('off')

    plt.suptitle(f'Pipeline T5 | nhóm: {query_label}', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, f'query_{query_idx}.png'),
                dpi=150, bbox_inches='tight')
    plt.show()

# Demo 3 query
for qi in [0, 100, 500]:
    visualize(qi, candidate_df, IMAGE_DIR, REMBG_DIR,
              fusion_feats, index, effnet_feats, minilm_feats, labels)
```

---

### CELL 11 – Ghi chú AI + Kế hoạch tuần 6

```markdown
## GHI CHÚ AI HỖ TRỢ

| Phần                | AI hỗ trợ                                | Người kiểm tra |
| ------------------- | ---------------------------------------- | -------------- |
| Pipeline design     | Claude gợi ý 2 giai đoạn                 | Mã Gia Vỹ      |
| rembg + cache       | Claude gợi ý cơ chế cache PNG            | Mã Gia Vỹ      |
| MobileCLIP fusion   | Claude gợi ý concat img+txt              | Mã Gia Vỹ      |
| EfficientNetB0      | Claude gợi ý dùng timm + transform chuẩn | Mã Gia Vỹ      |
| MiniLM multilingual | Claude gợi ý model phù hợp Shopee        | Mã Gia Vỹ      |
| Rerank function     | Claude gợi ý 50/50 image+text            | Mã Gia Vỹ      |

## KẾ HOẠCH TUẦN 6

| Nội dung                                  | Mục tiêu              |
| ----------------------------------------- | --------------------- |
| Thêm pHash boost vào reranking            | Bắt ảnh gần trùng lặp |
| Grid search tỷ lệ image/text (hiện 50/50) | Tìm tỷ lệ tối ưu      |
| Thêm TF-IDF vào fusion giai đoạn 1        | So sánh với MiniLM    |
| Fine-tune EfficientNetB0 trên Shopee      | Tăng mAP đáng kể      |
```

---

## LƯU Ý KHI CHẠY TRÊN COLAB

1. **Cell 4 (rembg) chậm nhất** — lần đầu ~2-3 tiếng, có cache thì chạy ngay
2. **Thứ tự chạy bắt buộc:** 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11
3. **Nếu Colab reset:** Chạy lại 0→3, sau đó 4→7 đều load cache ngay (không mất thời gian)
4. **Tên file metric:** `metrics_tuan5.csv` — file tổng hợp đọc đúng tên này

---

## KẾT QUẢ MONG ĐỢI

| Metric      | Tuần 4 | Tuần 5 (kỳ vọng) |
| ----------- | ------ | ---------------- |
| Precision@1 | ~0.79  | ~0.82–0.87       |
| Recall@5    | ~0.73  | ~0.76–0.82       |
| mAP@5       | 0.7635 | **0.80–0.87**    |

Tăng nhờ: rembg loại nhiễu nền + MobileCLIP multimodal + reranking 2 giai đoạn.
