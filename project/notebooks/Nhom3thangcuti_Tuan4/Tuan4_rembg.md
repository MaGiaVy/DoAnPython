# 🏆 Baseline 2: MobileCLIP (Zero-Shot Multimodal) + RemBG + pHash

## E-commerce Visual Search — Shopee Dataset (34,250 items) | Google Colab GPU

**Pipeline:**

- 🧹 **Pre-processing:** `rembg` — Xóa phông nền ảnh trước khi trích xuất features
- 🔢 **Dedup/Filter:** `pHash` (Perceptual Hash) — Lọc ảnh trùng lặp từ gallery
- 🍎 **Model:** `MobileCLIP` (Apple) — Image + Text cùng embedding space
- 🔀 **Fusion:** `L2_Norm(α × img_feat + (1-α) × txt_feat)`
- 🔍 **Search:** FAISS `IndexFlatIP` (Cosine Similarity)
- 📊 **Metrics:** mAP@5, Precision@1, Recall@5

**Fallback:** Nếu MobileCLIP không cài được → tự động dùng `openai/clip-vit-base-patch32` (HuggingFace).

**Dataset Split (STRICT — NO DATA LEAKAGE):**

- Gallery : toàn bộ 34,250 ảnh
- Val queries (20%) : ~6,850 → grid search `alpha`
- Test queries (80%) : ~27,400 → đánh giá cuối, chạy **1 lần duy nhất**

---

## ⚙️ Cell 0: Kiểm tra GPU & Runtime

```python
import subprocess
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
if result.returncode == 0:
    print('✅ GPU khả dụng:')
    print(result.stdout)
else:
    print('❌ Không tìm thấy GPU!')
    print('👉 Vào Runtime > Change runtime type > chọn T4 GPU rồi thử lại!')

import torch
print(f'PyTorch version : {torch.__version__}')
print(f'CUDA available  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU name        : {torch.cuda.get_device_name(0)}')
    print(f'VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
```

---

## 📦 Cell 1: Cài đặt thư viện & MobileCLIP

```python
import sys, subprocess

# Thư viện cơ bản + rembg
!pip install -q faiss-cpu timm rembg onnxruntime pillow imagehash

# ─── Thử cài MobileCLIP từ Apple ─────────────────────────────────────────────
USE_MOBILECLIP = False
print('⏳ Đang thử cài MobileCLIP (Apple)...')
try:
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-q',
         'git+https://github.com/apple/ml-mobileclip.git'],
        capture_output=True, text=True, timeout=180
    )
    import mobileclip
    USE_MOBILECLIP = True
    print('✅ MobileCLIP (Apple) cài thành công!')
except Exception as e:
    print(f'⚠️  Không cài được MobileCLIP: {e}')
    print('🔄 Fallback → openai/clip-vit-base-patch32 (HuggingFace)')
    !pip install -q transformers

print(f'\n🔧 Chế độ: {"MobileCLIP (Apple)" if USE_MOBILECLIP else "CLIP HuggingFace Fallback"}')
```

---

## 📂 Cell 2: Kết nối Google Drive & Tự động dò tìm đường dẫn Dataset

```python
from google.colab import drive
import os

drive.mount('/content/drive')

POSSIBLE_PATHS = [
    '/content/drive/MyDrive/DoAnPython/DuLieuPython',
    '/content/drive/MyDrive/DuLieuPython',
    '/content/drive/My Drive/DoAnPython/DuLieuPython',
    '/content/drive/My Drive/DuLieuPython'
]

DATA_DIR = None
for path in POSSIBLE_PATHS:
    if os.path.exists(os.path.join(path, 'train.csv')):
        DATA_DIR = path
        break

if DATA_DIR is None:
    DATA_DIR = '/content/drive/MyDrive/DuLieuPython/DuLieuPython'
    print(f'⚠️ Không tìm thấy train.csv. Dùng mặc định: {DATA_DIR}')
else:
    print(f'✅ Đã tự động phát hiện thư mục dữ liệu tại: {DATA_DIR}')

CSV_PATH        = os.path.join(DATA_DIR, 'train.csv')
IMAGE_ZIP_PATH  = os.path.join(DATA_DIR, 'train_images')
EXTRACTED_DIR   = '/content/train_images_extracted'
IMG_DIR         = os.path.join(EXTRACTED_DIR, 'train_images')

if not os.path.exists(IMG_DIR):
    if os.path.exists(IMAGE_ZIP_PATH):
        print('⏳ Đang giải nén train_images.zip...')
        !unzip -q {IMAGE_ZIP_PATH} -d {EXTRACTED_DIR}
        print('✅ Giải nén thành công!')
    else:
        print(f'❌ Không tìm thấy file zip tại {IMAGE_ZIP_PATH}!')
else:
    print('✅ Đã có thư mục ảnh giải nén cục bộ!')

for name, p in [('File CSV', CSV_PATH), ('Thư mục ảnh giải nén', IMG_DIR)]:
    status = 'Đã sẵn sàng' if os.path.exists(p) else 'KHÔNG tìm thấy – kiểm tra lại!'
    print(f'   {name}: {status} ({p})')
```

---

## 🧹 Cell 2.5: RemBG — Xóa Phông Nền Ảnh (với Save Point mỗi 500 ảnh)

> **Mục tiêu:** Loại bỏ background rác trước khi trích xuất embedding, giúp model tập trung vào vật thể sản phẩm.
>
> **Save point:** Cứ mỗi 500 ảnh xử lý xong, kết quả được lưu tự động để tránh mất dữ liệu khi Colab timeout.

```python
import os
import json
import shutil
import numpy as np
from PIL import Image
from tqdm.notebook import tqdm
from rembg import remove, new_session

# ─── Cấu hình đường dẫn ──────────────────────────────────────────────────────
REMBG_OUT_DIR    = '/content/train_images_rembg'          # Ảnh sau khi xóa phông
REMBG_CACHE_PATH = '/content/features/rembg_done.json'    # Checkpoint: list ảnh đã xử lý
SAVE_INTERVAL    = 500                                     # Save mỗi 500 ảnh

os.makedirs(REMBG_OUT_DIR, exist_ok=True)
os.makedirs('/content/features', exist_ok=True)

# ─── Load checkpoint nếu đã chạy dở ─────────────────────────────────────────
if os.path.exists(REMBG_CACHE_PATH):
    with open(REMBG_CACHE_PATH, 'r') as f:
        done_set = set(json.load(f))
    print(f'📂 Load checkpoint: đã xử lý {len(done_set):,} ảnh trước đó.')
else:
    done_set = set()
    print('🆕 Bắt đầu mới, chưa có checkpoint.')

# ─── Tạo rembg session (u2netp là mô hình nhẹ, cân bằng tốc độ/chất lượng) ──
rembg_session = new_session('u2netp')

# ─── Lấy danh sách tất cả ảnh cần xử lý ─────────────────────────────────────
all_images   = df_gallery['image'].tolist()
todo_images  = [f for f in all_images if f not in done_set]
print(f'📋 Tổng ảnh cần xử lý: {len(todo_images):,} / {len(all_images):,}')

# ─── Hàm xử lý 1 ảnh ─────────────────────────────────────────────────────────
def remove_bg_and_save(fname, src_dir, dst_dir, session):
    """
    Xóa phông 1 ảnh, lưu ra dst_dir với nền trắng (RGB).
    Trả về True nếu thành công, False nếu lỗi.
    """
    src_path = os.path.join(src_dir, fname)
    dst_path = os.path.join(dst_dir, fname)

    if os.path.exists(dst_path):       # đã xử lý rồi, bỏ qua
        return True
    try:
        img = Image.open(src_path).convert('RGB')
        result = remove(img, session=session)             # RGBA output
        # Paste lên nền trắng → RGB
        background = Image.new('RGB', result.size, (255, 255, 255))
        background.paste(result, mask=result.split()[3])  # alpha channel làm mask
        background.save(dst_path)
        return True
    except Exception as e:
        # Ảnh lỗi: copy ảnh gốc để không bị thiếu
        try:
            shutil.copy(src_path, dst_path)
        except Exception:
            pass
        return False

# ─── Vòng lặp chính với save point mỗi 500 ảnh ───────────────────────────────
error_count = 0

for i, fname in enumerate(tqdm(todo_images, desc='🧹 RemBG xóa phông')):
    ok = remove_bg_and_save(fname, IMG_DIR, REMBG_OUT_DIR, rembg_session)
    if ok:
        done_set.add(fname)
    else:
        error_count += 1

    # ── SAVE POINT mỗi 500 ảnh ──────────────────────────────────────────────
    if (i + 1) % SAVE_INTERVAL == 0:
        with open(REMBG_CACHE_PATH, 'w') as f:
            json.dump(list(done_set), f)
        # Đồng bộ lên Drive để bảo toàn khi Colab disconnect
        drive_cache = os.path.join(DATA_DIR, 'rembg_done.json')
        shutil.copy(REMBG_CACHE_PATH, drive_cache)
        print(f'\n💾 [Save point] Đã lưu checkpoint tại bước {i + 1:,} '
              f'({len(done_set):,} ảnh hoàn thành, {error_count} lỗi)')

# ─── Save cuối cùng ───────────────────────────────────────────────────────────
with open(REMBG_CACHE_PATH, 'w') as f:
    json.dump(list(done_set), f)
shutil.copy(REMBG_CACHE_PATH, os.path.join(DATA_DIR, 'rembg_done.json'))

print(f'\n✅ RemBG hoàn tất!')
print(f'   Tổng xử lý thành công : {len(done_set):,} ảnh')
print(f'   Số ảnh lỗi (giữ gốc)  : {error_count}')
print(f'   Thư mục output         : {REMBG_OUT_DIR}')

# ─── Cập nhật IMG_DIR trỏ sang ảnh đã xóa phông ─────────────────────────────
IMG_DIR_REMBG = REMBG_OUT_DIR
print(f'\n📁 IMG_DIR_REMBG = {IMG_DIR_REMBG}  ← dùng cho các bước tiếp theo')
```

> **Lưu ý về thời gian:**
>
> - `u2netp` xử lý ~2–3 ảnh/giây trên T4 GPU.
> - 34,250 ảnh ≈ 3–5 giờ. Nên chạy qua đêm hoặc chia session.
> - Nếu Colab timeout, chạy lại cell này — checkpoint sẽ tự tiếp tục từ chỗ dừng.

---

## 🔢 Cell 2.6: pHash — Phát hiện & Lọc Ảnh Trùng Lặp

> **Mục tiêu:** Dùng Perceptual Hash để phát hiện các ảnh gần giống nhau (ví dụ: ảnh sản phẩm chụp lại nhiều lần với background khác), giúp làm sạch gallery.

```python
import imagehash
import pandas as pd
from PIL import Image
from tqdm.notebook import tqdm
from collections import defaultdict

PHASH_THRESHOLD = 8   # Hamming distance ≤ 8 → coi là ảnh trùng (0=identical, 10=khá giống)
                       # Điều chỉnh: thấp hơn = chặt hơn, cao hơn = lỏng hơn

# ─── Tính pHash cho toàn bộ gallery (dùng ảnh đã rembg) ─────────────────────
phash_list  = []
valid_flags = []   # True = ảnh hợp lệ (không lỗi)

for fname in tqdm(df_gallery['image'], desc='🔢 Tính pHash'):
    img_path = os.path.join(IMG_DIR_REMBG, fname)
    try:
        img  = Image.open(img_path).convert('RGB')
        phash_val = str(imagehash.phash(img))
        phash_list.append(phash_val)
        valid_flags.append(True)
    except Exception:
        phash_list.append(None)
        valid_flags.append(False)

df_gallery['computed_phash'] = phash_list
df_gallery['img_valid']      = valid_flags

print(f'✅ Tính pHash xong: {sum(valid_flags):,} ảnh hợp lệ / {len(df_gallery):,}')

# ─── So sánh với image_phash sẵn có trong dataset ────────────────────────────
# Dataset Shopee đã có cột image_phash — kiểm tra xem khớp không
match_count = (df_gallery['computed_phash'] == df_gallery['image_phash']).sum()
print(f'📊 pHash khớp với cột gốc: {match_count:,} / {sum(valid_flags):,} ảnh')

# ─── Phát hiện nhóm ảnh trùng lặp dựa trên pHash ────────────────────────────
# Dùng image_phash từ dataset (đã có sẵn, nhanh hơn tính lại)
phash_groups = defaultdict(list)
for idx, row in df_gallery.iterrows():
    phash_groups[row['image_phash']].append(row['posting_id'])

# Thống kê
exact_dup_groups = {k: v for k, v in phash_groups.items() if len(v) > 1}
total_dup_imgs   = sum(len(v) - 1 for v in exact_dup_groups.values())

print(f'\n📊 Thống kê pHash (exact match):')
print(f'   Nhóm có ảnh trùng: {len(exact_dup_groups):,} nhóm')
print(f'   Tổng ảnh trùng lặp: {total_dup_imgs:,} ảnh')

# ─── Ví dụ: Hiển thị 3 nhóm trùng đầu tiên ──────────────────────────────────
print('\n🔍 Ví dụ 3 nhóm ảnh trùng pHash:')
for i, (phash_val, pids) in enumerate(list(exact_dup_groups.items())[:3]):
    print(f'   pHash={phash_val}: {len(pids)} ảnh → {pids[:4]}{"..." if len(pids)>4 else ""}')

# ─── Lưu mapping pHash → danh sách posting_id ───────────────────────────────
PHASH_MAP_PATH = '/content/features/phash_groups.json'
with open(PHASH_MAP_PATH, 'w') as f:
    json.dump(exact_dup_groups, f)
print(f'\n✅ Đã lưu pHash groups: {PHASH_MAP_PATH}')

# ─── Tùy chọn: Lọc gallery bỏ ảnh trùng (giữ 1 đại diện mỗi nhóm) ──────────
# Uncomment nếu muốn dùng gallery đã dedup
#
# keep_pids = set()
# for pids in phash_groups.values():
#     keep_pids.add(pids[0])   # giữ ảnh đầu tiên mỗi nhóm
# df_gallery_dedup = df_gallery[df_gallery['posting_id'].isin(keep_pids)].reset_index(drop=True)
# print(f'✂️  Gallery sau dedup: {len(df_gallery_dedup):,} ảnh (bỏ {len(df_gallery)-len(df_gallery_dedup):,} ảnh)')
```

---

## 🔧 Cell 3: Import & Cấu hình

```python
!pip install -q faiss-cpu

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.notebook import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import faiss

# ─── Cấu hình ────────────────────────────────────────────────────────────────
MOBILECLIP_VARIANT = 'mobileclip_s0'
MOBILECLIP_CKPT    = '/tmp/mobileclip_s0.pt'

BATCH_SIZE   = 128
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_SEED  = 42
NUM_WORKERS  = 2

# ── Dùng ảnh đã xóa phông cho toàn bộ pipeline ──
ACTIVE_IMG_DIR = IMG_DIR_REMBG   # ← thay vì IMG_DIR gốc

print(f'✅ Thiết bị        : {DEVICE}')
print(f'✅ Batch size      : {BATCH_SIZE}')
print(f'✅ Thư mục ảnh     : {ACTIVE_IMG_DIR}')
```

---

## 📊 Cell 4: Đọc dữ liệu & Chia tập (STRICT SPLIT)

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(CSV_PATH)
print(f'📊 Tổng số mẫu : {len(df):,}')
print(f'📋 Các cột     : {list(df.columns)}')
print(df.head(3))

# ─── KHÔNG stratify vì số lớp (11,014) > kích thước validation (6,850) ───
df_gallery = df.copy()

val_idx, test_idx = train_test_split(
    df.index.tolist(),
    test_size    = 0.8,
    random_state = RANDOM_SEED
)

df_val  = df.loc[val_idx].reset_index(drop=True)
df_test = df.loc[test_idx].reset_index(drop=True)

print(f'\n🗂️  Gallery size  : {len(df_gallery):,} ảnh (toàn bộ dataset)')
print(f'✅ Val queries   : {len(df_val):,} ảnh  → grid search alpha')
print(f'✅ Test queries  : {len(df_test):,} ảnh  → đánh giá cuối (1 lần!)')
```

---

## 🍎 Cell 5: Tải MobileCLIP (hoặc Fallback CLIP)

```python
if USE_MOBILECLIP:
    import mobileclip, urllib.request

    CKPT_URLS = {
        'mobileclip_s0': 'https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s0.pt',
        'mobileclip_s1': 'https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s1.pt',
        'mobileclip_s2': 'https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s2.pt',
        'mobileclip_b' : 'https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_b.pt',
    }

    if not os.path.exists(MOBILECLIP_CKPT):
        print(f'⏬ Đang download checkpoint {MOBILECLIP_VARIANT}...')
        urllib.request.urlretrieve(CKPT_URLS[MOBILECLIP_VARIANT], MOBILECLIP_CKPT)
        print(f'✅ Đã lưu: {MOBILECLIP_CKPT}')

    print(f'⏳ Đang tải {MOBILECLIP_VARIANT}...')
    clip_model, _, preprocess = mobileclip.create_model_and_transforms(
        MOBILECLIP_VARIANT, pretrained=MOBILECLIP_CKPT
    )
    tokenizer  = mobileclip.get_tokenizer(MOBILECLIP_VARIANT)
    clip_model = clip_model.to(DEVICE).eval()

    with torch.no_grad():
        _dummy   = torch.randn(1, 3, 256, 256).to(DEVICE)
        embed_dim = clip_model.encode_image(_dummy).shape[-1]

    MODEL_LABEL = f'MobileCLIP ({MOBILECLIP_VARIANT})'

else:
    from transformers import CLIPModel, CLIPProcessor
    HF_MODEL   = 'openai/clip-vit-base-patch32'
    print(f'⏳ Đang tải {HF_MODEL}...')
    clip_model = CLIPModel.from_pretrained(HF_MODEL).to(DEVICE).eval()
    preprocess = CLIPProcessor.from_pretrained(HF_MODEL)
    tokenizer  = None
    embed_dim  = clip_model.config.projection_dim
    MODEL_LABEL = f'CLIP ({HF_MODEL})'

print(f'\n✅ {MODEL_LABEL} đã sẵn sàng')
print(f'📐 Embedding dim: {embed_dim}')
```

---

## 🖼️📝 Cell 6: Hàm trích xuất Image & Text Features (dùng ảnh RemBG)

> Lưu ý: `ShopeeImageDataset` được cập nhật để đọc từ `ACTIVE_IMG_DIR` (ảnh đã xóa phông).

```python
class ShopeeImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, return_tensor=True):
        self.df            = df
        self.img_dir       = img_dir
        self.transform     = transform
        self.return_tensor = return_tensor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fname = self.df.iloc[idx]['image']
        try:
            img = Image.open(os.path.join(self.img_dir, fname)).convert('RGB')
        except Exception:
            img = Image.new('RGB', (256, 256), (255, 255, 255))  # nền trắng khi lỗi
        if self.return_tensor and self.transform:
            return self.transform(img)
        return img


@torch.no_grad()
def extract_image_features_clip(df_input, img_dir, batch_size=128, num_workers=2):
    all_feats = []
    if USE_MOBILECLIP:
        dataset = ShopeeImageDataset(df_input, img_dir,
                                     transform=preprocess, return_tensor=True)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
        for imgs in tqdm(loader, desc='🖼️ MobileCLIP image features (RemBG)'):
            feats = clip_model.encode_image(imgs.to(DEVICE))
            all_feats.append(feats.cpu().float().numpy())
    else:
        dataset = ShopeeImageDataset(df_input, img_dir, return_tensor=False)
        for i in tqdm(range(0, len(dataset), batch_size),
                      desc='🖼️ CLIP HuggingFace image features (RemBG)'):
            batch  = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
            inputs = preprocess(images=batch, return_tensors='pt',
                                padding=True).to(DEVICE)
            feats  = clip_model.get_image_features(**inputs)
            all_feats.append(feats.cpu().float().numpy())
    return np.vstack(all_feats)


@torch.no_grad()
def extract_text_features_clip(df_input, batch_size=256):
    titles    = df_input['title'].fillna('').tolist()
    all_feats = []
    if USE_MOBILECLIP:
        for i in tqdm(range(0, len(titles), batch_size),
                      desc='📝 MobileCLIP text features'):
            tokens = tokenizer(titles[i:i + batch_size]).to(DEVICE)
            feats  = clip_model.encode_text(tokens)
            feats  = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu().float().numpy())
    else:
        for i in tqdm(range(0, len(titles), batch_size),
                      desc='📝 CLIP HuggingFace text features'):
            inputs = preprocess(
                text=titles[i:i + batch_size], return_tensors='pt',
                padding=True, truncation=True, max_length=77
            ).to(DEVICE)
            feats  = clip_model.get_text_features(**inputs)
            all_feats.append(feats.cpu().float().numpy())
    return np.vstack(all_feats)


print('✅ Hàm trích xuất features đã sẵn sàng (dùng ảnh RemBG)')
```

---

## 🔀 Cell 7: Fusion, FAISS & Metrics Utils

```python
def fuse_and_normalize_clip(img_feats, txt_feats, alpha):
    fused = alpha * img_feats + (1 - alpha) * txt_feats
    norms = np.linalg.norm(fused, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    return (fused / norms).astype('float32')


def build_faiss_index(feats):
    dim   = feats.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(feats)
    index.add(feats)
    return index


def get_ground_truth_dict(df):
    gt = {}
    for _, grp in df.groupby('label_group'):
        pids = set(grp['posting_id'].tolist())
        for pid in pids:
            gt[pid] = pids
    return gt


def evaluate_retrieval_clip(query_df, gallery_df,
                             query_img, query_txt,
                             gallery_img, gallery_txt,
                             alpha, K=5):
    gallery_fused = fuse_and_normalize_clip(gallery_img, gallery_txt, alpha)
    query_fused   = fuse_and_normalize_clip(query_img,   query_txt,   alpha)
    index         = build_faiss_index(gallery_fused.copy())
    faiss.normalize_L2(query_fused)

    _, I = index.search(query_fused, K + 1)

    gallery_pids = gallery_df['posting_id'].tolist()
    gt_dict      = get_ground_truth_dict(gallery_df)

    ap_list, p1_list, r5_list = [], [], []
    for i, row in enumerate(query_df.itertuples()):
        qid      = row.posting_id
        relevant = gt_dict.get(qid, set()) - {qid}
        if not relevant:
            continue
        retrieved = [gallery_pids[idx] for idx in I[i] if gallery_pids[idx] != qid][:K]
        hits, ap  = 0, 0.0
        for rank, pid in enumerate(retrieved, 1):
            if pid in relevant:
                hits += 1
                ap   += hits / rank
        ap_list.append(ap / min(len(relevant), K))
        p1_list.append(1.0 if retrieved and retrieved[0] in relevant else 0.0)
        r5_list.append(len(set(retrieved) & relevant) / len(relevant))

    return {
        'mAP@5':       float(np.mean(ap_list)),
        'Precision@1': float(np.mean(p1_list)),
        'Recall@5':    float(np.mean(r5_list)),
    }


print('✅ Hàm fusion / FAISS / evaluate đã sẵn sàng')
```

---

## 🏗️ Cell 8: Trích xuất Gallery & Val/Test Features (từ ảnh RemBG)

```python
import os

FEAT_DIR = '/content/features'
os.makedirs(FEAT_DIR, exist_ok=True)

# ── Cache paths ────────────────────────────────────────────────────────────────
GALLERY_IMG_CACHE = os.path.join(FEAT_DIR, 'rembg_gallery_img.npy')
GALLERY_TXT_CACHE = os.path.join(FEAT_DIR, 'rembg_gallery_txt.npy')
VAL_IMG_CACHE     = os.path.join(FEAT_DIR, 'rembg_val_img.npy')
VAL_TXT_CACHE     = os.path.join(FEAT_DIR, 'rembg_val_txt.npy')
TEST_IMG_CACHE    = os.path.join(FEAT_DIR, 'rembg_test_img.npy')
TEST_TXT_CACHE    = os.path.join(FEAT_DIR, 'rembg_test_txt.npy')

# ── Gallery ───────────────────────────────────────────────────────────────────
if os.path.exists(GALLERY_IMG_CACHE):
    print('📂 Load gallery features từ cache...')
    gallery_img_feats = np.load(GALLERY_IMG_CACHE)
    gallery_txt_feats = np.load(GALLERY_TXT_CACHE)
else:
    print('🔄 Trích xuất gallery features (ảnh RemBG)...')
    gallery_img_feats = extract_image_features_clip(df_gallery, ACTIVE_IMG_DIR)
    gallery_txt_feats = extract_text_features_clip(df_gallery)
    np.save(GALLERY_IMG_CACHE, gallery_img_feats)
    np.save(GALLERY_TXT_CACHE, gallery_txt_feats)
    print(f'✅ Đã lưu cache gallery')

# ── Val ───────────────────────────────────────────────────────────────────────
if os.path.exists(VAL_IMG_CACHE):
    print('📂 Load val features từ cache...')
    val_img_feats = np.load(VAL_IMG_CACHE)
    val_txt_feats = np.load(VAL_TXT_CACHE)
else:
    print('🔄 Trích xuất val features...')
    val_img_feats = extract_image_features_clip(df_val, ACTIVE_IMG_DIR)
    val_txt_feats = extract_text_features_clip(df_val)
    np.save(VAL_IMG_CACHE, val_img_feats)
    np.save(VAL_TXT_CACHE, val_txt_feats)

# ── Test ──────────────────────────────────────────────────────────────────────
if os.path.exists(TEST_IMG_CACHE):
    print('📂 Load test features từ cache...')
    test_img_feats = np.load(TEST_IMG_CACHE)
    test_txt_feats = np.load(TEST_TXT_CACHE)
else:
    print('🔄 Trích xuất test features...')
    test_img_feats = extract_image_features_clip(df_test, ACTIVE_IMG_DIR)
    test_txt_feats = extract_text_features_clip(df_test)
    np.save(TEST_IMG_CACHE, test_img_feats)
    np.save(TEST_TXT_CACHE, test_txt_feats)

print(f'\n✅ Shape gallery img : {gallery_img_feats.shape}')
print(f'✅ Shape gallery txt : {gallery_txt_feats.shape}')
print(f'✅ Shape val img     : {val_img_feats.shape}')
print(f'✅ Shape test img    : {test_img_feats.shape}')
```

---

## 🎛️ Cell 9: Grid Search Alpha trên VAL SET

```python
import numpy as np

best_alpha_2    = 0.5
best_map5_2     = 0.0
alpha_results_2 = []

print('🎛️  Grid Search alpha trên VAL SET (ảnh RemBG)')
print('─' * 55)

for alpha in np.arange(0.0, 1.05, 0.1):
    val_m = evaluate_retrieval_clip(
        query_df    = df_val,
        gallery_df  = df_gallery,
        query_img   = val_img_feats,
        query_txt   = val_txt_feats,
        gallery_img = gallery_img_feats,
        gallery_txt = gallery_txt_feats,
        alpha       = alpha, K=5
    )
    alpha_results_2.append({'alpha': round(alpha, 1), **val_m})
    marker = ' ← BEST' if val_m['mAP@5'] > best_map5_2 else ''
    print(
        f'  alpha={alpha:.1f} | '
        f'mAP@5={val_m["mAP@5"]:.4f} | '
        f'P@1={val_m["Precision@1"]:.4f} | '
        f'R@5={val_m["Recall@5"]:.4f}{marker}'
    )
    if val_m['mAP@5'] > best_map5_2:
        best_map5_2  = val_m['mAP@5']
        best_alpha_2 = alpha

print('─' * 55)
print(f'\n✅ Best alpha = {best_alpha_2:.1f} | Val mAP@5 = {best_map5_2:.4f}')
```

---

## 🧪 Cell 10: Đánh giá TEST SET (Chạy 1 lần duy nhất!)

```python
print(f'🧪 Đánh giá TEST SET với BEST_ALPHA_2 = {best_alpha_2:.1f}')
print('⚠️  Đây là lần chạy DUY NHẤT trên test set!\n')

test_metrics_2 = evaluate_retrieval_clip(
    query_df    = df_test,
    gallery_df  = df_gallery,
    query_img   = test_img_feats,
    query_txt   = test_txt_feats,
    gallery_img = gallery_img_feats,
    gallery_txt = gallery_txt_feats,
    alpha       = best_alpha_2, K=5
)

print(f'📊 KẾT QUẢ — Baseline 2 ({MODEL_LABEL} + RemBG) trên TEST SET:')
print(f'   mAP@5        = {test_metrics_2["mAP@5"]:.4f}')
print(f'   Precision@1  = {test_metrics_2["Precision@1"]:.4f}')
print(f'   Recall@5     = {test_metrics_2["Recall@5"]:.4f}')
```

---

## 📋 Cell 11: Xuất kết quả

```python
import pandas as pd, os

metrics_data = [
    {
        "Model (Phương pháp chính)": f"MobileCLIP + RemBG (Zero-Shot Fusion)",
        "Pre-processing":            "rembg u2netp + pHash dedup",
        "Kích thước Vector (Dim)":   str(embed_dim),
        "Alpha tối ưu (Validation)": round(float(best_alpha_2), 1),
        "Test mAP@5":                round(test_metrics_2["mAP@5"], 4),
        "Test Precision@1":          round(test_metrics_2["Precision@1"], 4),
        "Test Recall@5":             round(test_metrics_2["Recall@5"], 4),
    }
]

df_metrics = pd.DataFrame(metrics_data)
print('📊 Final Metrics:')
print(df_metrics.to_string(index=False))

LOCAL_CSV = "/content/final_metric_rembg.csv"
df_metrics.to_csv(LOCAL_CSV, index=False, encoding='utf-8-sig')
print(f"\n✅ Đã lưu local : {LOCAL_CSV}")

DRIVE_CSV = os.path.join(DATA_DIR, "final_metric_rembg.csv")
df_metrics.to_csv(DRIVE_CSV, index=False, encoding='utf-8-sig')
print(f"✅ Đã lưu Drive  : {DRIVE_CSV}")

print('\n🏁 Hoàn tất xuất kết quả!')
```

---

## 📝 Ghi chú kỹ thuật

### RemBG — Lựa chọn model

| Model               | Tốc độ (T4) | Chất lượng | Ghi chú                     |
| ------------------- | ----------- | ---------- | --------------------------- |
| `u2netp`            | ~2–3 ảnh/s  | Tốt        | ✅ Khuyến nghị cho Colab    |
| `u2net`             | ~1–2 ảnh/s  | Tốt hơn    | Nặng hơn 2x                 |
| `isnet-general-use` | ~1 ảnh/s    | Tốt nhất   | Nặng, dùng khi có thời gian |

Đổi model: thay `new_session('u2netp')` bằng `new_session('u2net')`.

### pHash — Ý nghĩa ngưỡng

| Hamming Distance | Ý nghĩa                       |
| ---------------- | ----------------------------- |
| 0                | Ảnh giống hệt                 |
| 1–5              | Rất giống (crop nhẹ, resize)  |
| 6–10             | Khá giống (khác góc chụp nhỏ) |
| > 10             | Ảnh khác nhau                 |

Dataset Shopee đã có sẵn cột `image_phash` — có thể dùng trực tiếp thay vì tính lại.

### Save Point Strategy

```
Mỗi 500 ảnh RemBG xong:
  ├── Lưu checkpoint JSON cục bộ (/content/features/rembg_done.json)
  └── Đồng bộ lên Google Drive (DATA_DIR/rembg_done.json)

Khi Colab timeout → chạy lại cell RemBG:
  └── Tự động đọc checkpoint → bỏ qua ảnh đã xử lý → tiếp tục từ chỗ dừng
```
