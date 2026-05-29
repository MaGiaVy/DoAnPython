# 🚀 HƯỚNG DẪN TRIỂN KHAI PIPELINE 2 GIAI ĐOẠN

## MobileCLIP + DINOv2 Re‑ranking

### 🎯 Mục tiêu

- Nâng cấp baseline MobileCLIP + Alpha Tuning (mAP@5 ≈ 0.77) bằng cách thêm bước **re‑ranking với DINOv2‑Small**.
- Tận dụng tối đa GPU T4 (15.6 GB VRAM), quản lý bộ nhớ chặt chẽ để không bị `CUDA Out of Memory`.
- Toàn bộ code và log được viết bằng tiếng Việt, dễ hiểu cho nhóm.

### 🧱 Kiến trúc pipeline

Ảnh query
│
├──> MobileCLIP (GĐ1) ──> top‑K ứng viên (K = 100)
│
└──> DINOv2‑Small (GĐ2) ──> Tính lại similarity cho từng ứng viên
│
└──> Sắp xếp lại ──> Top‑5 cuối cùng

text

### 📦 Chuẩn bị môi trường

```python
!pip install faiss-cpu timm transformers torch torchvision pillow tqdm
🧠 Bước 0: Load lại các thành phần đã có
File best_alpha (từ grid search).

Ma trận đặc trưng MobileCLIP của gallery (mobileclip_gallery.npy).

FAISS index đã build cho MobileCLIP.

DataFrame df_gallery, df_val, df_test (đã chia tập).

🔁 Bước 1: Cache đặc trưng DINOv2 cho gallery
Mục đích: Chạy 1 lần duy nhất, tiết kiệm thời gian inference về sau.

python
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm

# 1.1. Load DINOv2‑Small
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dinov2 = dinov2.to(device).eval()

# 1.2. Transform ảnh chuẩn
transform_dino = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_dinov2_embedding(img_path):
    """Trả về vector 384 chiều đã chuẩn hóa L2."""
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform_dino(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = dinov2.forward_features(img_tensor)['x_norm_clstoken']
        feats = F.normalize(feats, dim=-1)
    return feats.cpu().numpy().flatten()

# 1.3. Tính toán hoặc load cache
gallery_paths = [os.path.join(IMAGE_DIR, fname) for fname in df_gallery['image']]
cache_path = 'features/dinov2_gallery.npy'

if not os.path.exists(cache_path):
    gallery_dino = []
    for path in tqdm(gallery_paths, desc='🦕 DINOv2 gallery'):
        gallery_dino.append(get_dinov2_embedding(path))
    gallery_dino = np.array(gallery_dino, dtype='float32')
    np.save(cache_path, gallery_dino)
else:
    gallery_dino = np.load(cache_path)

print(f'✅ Đặc trưng DINOv2 gallery: {gallery_dino.shape}')
🧹 Bước 1.5: Dọn dẹp VRAM trước khi load MobileCLIP
Quan trọng: Nếu bạn đã load MobileCLIP trước đó, hãy giải phóng hoàn toàn.

python
# Chạy cell này trước khi nạp MobileCLIP
import gc
del dinov2  # nếu không dùng nữa
gc.collect()
torch.cuda.empty_cache()
print("🧹 Đã dọn sạch VRAM, sẵn sàng nạp MobileCLIP!")
Lưu ý: Trong quá trình chạy thực tế, bạn sẽ load MobileCLIP trước, trích xuất đặc trưng (nếu chưa có), build FAISS, rồi mới chuyển sang DINOv2. Hãy luôn del model cũ và empty_cache() trước khi load model mới.

🔎 Bước 2: Hàm tìm kiếm 2 giai đoạn
python
def search_two_stage(query_img_path, alpha, retrieval_k=100, final_k=5):
    """
    GĐ1: MobileCLIP lấy top‑retrieval_k ứng viên.
    GĐ2: DINOv2 re‑ranking để chọn ra final_k kết quả.
    """
    # ----- Giai đoạn 1: MobileCLIP -----
    # (sử dụng code search có sẵn, trả về indices và scores)
    candidate_indices, candidate_scores = search_mobileclip(
        query_img_path, alpha, top_k=retrieval_k
    )

    # ----- Giai đoạn 2: DINOv2 re‑ranking -----
    q_dino = get_dinov2_embedding(query_img_path)
    rerank_scores = []
    for idx in candidate_indices:
        # gallery_dino đã được cache toàn bộ
        sim = np.dot(q_dino, gallery_dino[idx])
        rerank_scores.append(sim)

    # Sắp xếp lại theo độ tương đồng DINOv2 (giảm dần)
    new_order = np.argsort(rerank_scores)[::-1]
    final_indices = [candidate_indices[i] for i in new_order[:final_k]]
    final_scores = [rerank_scores[i] for i in new_order[:final_k]]
    return final_indices, final_scores
Chú ý: Hàm search_mobileclip cần trả về chỉ số (index) trong gallery. Nếu hiện tại hàm của bạn trả về đường dẫn, hãy ánh xạ ngược lại bằng cách tạo một dictionary path_to_idx.

🎛️ Bước 3: Tuning retrieval_k trên tập Validation
python
best_alpha = ...  # giá trị bạn đã tìm được trước đó (ví dụ 0.7)
best_k = 100
best_val_map = 0.0

for k in [50, 100, 150, 200]:
    # Bạn cần một hàm evaluate_map_simple sử dụng search_two_stage
    val_map = evaluate_map_two_stage(df_val, best_alpha, retrieval_k=k, final_k=5)
    print(f'🔍 retrieval_k={k:3d}  |  Val mAP@5 = {val_map:.4f}')
    if val_map > best_val_map:
        best_val_map = val_map
        best_k = k

print(f'\n✅ Chọn retrieval_k = {best_k} với Val mAP@5 = {best_val_map:.4f}')
Gợi ý hàm evaluate_map_two_stage (có thể viết nhanh):

python
def evaluate_map_two_stage(query_df, alpha, retrieval_k, final_k=5):
    ap_list = []
    for _, row in tqdm(query_df.iterrows(), total=len(query_df)):
        query_path = os.path.join(IMAGE_DIR, row['image'])
        true_label = row['label_group']
        try:
            pred_indices, _ = search_two_stage(query_path, alpha, retrieval_k, final_k)
        except:
            ap_list.append(0.0)
            continue
        pred_labels = [df_gallery.iloc[i]['label_group'] for i in pred_indices]
        # Tính AP đơn giản (coi thứ hạng đã là score)
        y_true = [1 if lbl == true_label else 0 for lbl in pred_labels]
        if sum(y_true) == 0:
            ap = 0.0
        else:
            # Dùng công thức AP thủ công (hoặc sklearn nếu muốn)
            ap = sum([(i+1) / (idx+1) for i, idx in enumerate([j for j, v in enumerate(y_true) if v == 1])]) / max(sum(y_true), 1)
        ap_list.append(ap)
    return np.mean(ap_list)
🧪 Bước 4: Đánh giá CUỐI CÙNG trên Test (chạy 1 lần)
python
print('⚠️  Chạy đánh giá TEST DUY NHẤT 1 LẦN')
test_map = evaluate_map_two_stage(df_test, best_alpha, retrieval_k=best_k, final_k=5)
print(f'🏆 KẾT QUẢ CUỐI CÙNG (Test): mAP@5 = {test_map:.4f}')

# Ghi vào file CSV
import pandas as pd
result = pd.DataFrame([{
    'Method': 'MobileCLIP + DINOv2 Re-rank',
    'mAP@5': test_map,
    'Best Alpha': best_alpha,
    'Retrieval K': best_k
}])
result.to_csv('results/final_metrics.csv', index=False)
print('✅ Đã lưu kết quả vào results/final_metrics.csv')
```
