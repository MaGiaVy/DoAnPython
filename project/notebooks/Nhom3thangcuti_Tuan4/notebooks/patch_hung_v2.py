"""
patch_hung_v2.py
Patch Tuan4_Hung_Dinov3+miniML.ipynb theo yêu cầu:
  Fix 1: gt_len = 0 -> continue (thay vì append 0)
  Fix 2: all-MiniLM-L6-v2 -> paraphrase-multilingual-MiniLM-L12-v2
  Fix 3: Grid search alpha + pHash threshold + val/test split
  Bổ sung: Lưu CSV metric, thêm markdown AI notes
"""

import json
import copy
import re

NB_PATH = r"d:\New folder (3)\project\notebooks\Nhom3thangcuti_Tuan4\notebooks\Tuan4_Hung_Dinov3+miniML.ipynb"
OUT_PATH = r"d:\New folder (3)\project\notebooks\Nhom3thangcuti_Tuan4\notebooks\Tuan4_Hung_Dinov3+miniML.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ─────────────────────────────────────────────────────────────────────────────
# Hàm tiện ích
# ─────────────────────────────────────────────────────────────────────────────
def make_markdown_cell(text, cell_id):
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": [text]
    }

def make_code_cell(source_lines, cell_id, exec_count=None):
    return {
        "cell_type": "code",
        "execution_count": exec_count,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source_lines
    }

def source_str(cell):
    return "".join(cell["source"])

# ─────────────────────────────────────────────────────────────────────────────
# FIX 2: Đổi model MiniLM → multilingual trong tất cả các cell
# (cell load model + cell trích xuất features)
# ─────────────────────────────────────────────────────────────────────────────
MINILM_OLD = "all-MiniLM-L6-v2"
MINILM_NEW = "paraphrase-multilingual-MiniLM-L12-v2"

replaced_minilm = 0
for cell in nb["cells"]:
    new_src = []
    changed = False
    for line in cell["source"]:
        if MINILM_OLD in line:
            new_src.append(line.replace(MINILM_OLD, MINILM_NEW))
            changed = True
        else:
            new_src.append(line)
    if changed:
        cell["source"] = new_src
        replaced_minilm += 1
print(f"[Fix 2] Đã đổi MiniLM model trong {replaced_minilm} cell(s)")

# ─────────────────────────────────────────────────────────────────────────────
# Sửa batch_size cho multilingual model (lớn hơn, cần giảm batch)
# ─────────────────────────────────────────────────────────────────────────────
for cell in nb["cells"]:
    new_src = []
    for line in cell["source"]:
        # Giảm batch_size cho text encode vì model mới lớn hơn
        if "batch_size=256" in line and "encode" in "".join(cell["source"]):
            new_src.append(line.replace("batch_size=256", "batch_size=128"))
        else:
            new_src.append(line)
    cell["source"] = new_src

# ─────────────────────────────────────────────────────────────────────────────
# Thêm markdown AI note cho Cell Load Model (cell id 5f5feb5b)
# ─────────────────────────────────────────────────────────────────────────────
ai_note_model = make_markdown_cell(
    "## 🤖 AI Note – Chọn Model\n\n"
    "**DINOv3** (`vit_base_patch16_dinov3.lvd1689m`): Model vision transformer tự giám sát của Meta, "
    "được train trên 142M ảnh. Tốt hơn DINOv2 nhờ dữ liệu lớn hơn và kỹ thuật distillation cải tiến.\n\n"
    "**paraphrase-multilingual-MiniLM-L12-v2**: Model NLP đa ngôn ngữ (50+ ngôn ngữ bao gồm Tiếng Việt, "
    "Indonesia, Thái...). Phù hợp hơn `all-MiniLM-L6-v2` (chỉ tiếng Anh) vì dữ liệu Shopee là đa ngôn ngữ. "
    "Kỳ vọng tăng mAP ~3–5%.",
    "ai_note_model_01"
)

# ─────────────────────────────────────────────────────────────────────────────
# XÂY DỰNG LẠI CÁC CELL CUỐI (Similarity + Metric) với tất cả fix
# ─────────────────────────────────────────────────────────────────────────────

# ── Cell: Chia val/test split + AI Note ──────────────────────────────────────
cell_split_note = make_markdown_cell(
    "## 🤖 AI Note – Chia Val/Test Split\n\n"
    "Chia dữ liệu thành 3 phần để đánh giá mô hình đúng chuẩn:\n"
    "- **Train (80%)**: Dùng để trích xuất features\n"
    "- **Val (10%)**: Dùng để grid search tham số tối ưu (alpha, pHash threshold)\n"
    "- **Test (10%)**: Dùng để báo cáo metric cuối cùng – **không được dùng để chọn tham số**\n\n"
    "Lý do: Nếu dùng toàn bộ data để cả chọn tham số lẫn đánh giá → metric bị overfit, không trung thực.",
    "ai_note_split_01"
)

cell_split_code_lines = [
    "import os\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import torch\n",
    "from tqdm import tqdm\n",
    "\n",
    "csv_path = \"/content/drive/MyDrive/DoAnPython/DuLieuPython/train.csv\"\n",
    "output_dir = \"/content/drive/MyDrive/DoAnPython/DuLieuPython\"\n",
    "\n",
    "candidate_df = pd.read_csv(csv_path)\n",
    "\n",
    "# ── Chia val/test split theo label_group (đảm bảo không data leak) ──────────\n",
    "# Lấy danh sách label_group duy nhất, shuffle ngẫu nhiên\n",
    "np.random.seed(42)\n",
    "unique_labels = candidate_df['label_group'].unique()\n",
    "np.random.shuffle(unique_labels)\n",
    "\n",
    "n = len(unique_labels)\n",
    "n_val  = int(n * 0.10)   # 10% label groups cho val\n",
    "n_test = int(n * 0.10)   # 10% label groups cho test\n",
    "\n",
    "val_labels  = set(unique_labels[:n_val])\n",
    "test_labels = set(unique_labels[n_val:n_val + n_test])\n",
    "\n",
    "val_mask  = candidate_df['label_group'].isin(val_labels)\n",
    "test_mask = candidate_df['label_group'].isin(test_labels)\n",
    "train_mask = ~val_mask & ~test_mask\n",
    "\n",
    "val_df   = candidate_df[val_mask].reset_index(drop=True)\n",
    "test_df  = candidate_df[test_mask].reset_index(drop=True)\n",
    "train_df = candidate_df[train_mask].reset_index(drop=True)\n",
    "\n",
    "print(f\"Train: {len(train_df):,} samples | Val: {len(val_df):,} samples | Test: {len(test_df):,} samples\")\n",
    "print(f\"Val labels: {len(val_labels)} | Test labels: {len(test_labels)}\")\n",
    "\n",
    "# Lưu index để tra cứu nhanh sau\n",
    "val_indices  = candidate_df[val_mask].index.values\n",
    "test_indices = candidate_df[test_mask].index.values\n",
]

cell_split_code = make_code_cell(cell_split_code_lines, "split_code_01")

# ── Cell: Grid Search Note ────────────────────────────────────────────────────
cell_gridsearch_note = make_markdown_cell(
    "## 🤖 AI Note – Grid Search Alpha + pHash Threshold\n\n"
    "Thay vì dùng `alpha = 0.7` cố định:\n"
    "- **Alpha**: Trọng số kết hợp image/text similarity. `sim = alpha * img_sim + (1-alpha) * txt_sim`\n"
    "- **pHash threshold**: Ngưỡng Hamming distance để coi 2 ảnh là \"giống nhau\". "
    "Threshold quá thấp (=2) bỏ sót nhiều ảnh tương đồng. Threshold 5–10 capture được nhiều hơn.\n\n"
    "Grid search tìm bộ tham số tốt nhất trên **val set** → tránh overfit.",
    "ai_note_gridsearch_01"
)

# ── Cell: Grid Search Code ────────────────────────────────────────────────────
cell_gridsearch_code_lines = [
    "import gc\n",
    "\n",
    "# Load features đã trích xuất\n",
    "dinov3_image_features = np.load(os.path.join(output_dir, 'dinov3_image_features.npy')).astype('float32')\n",
    "minilm_text_features  = np.load(os.path.join(output_dir, 'minilm_text_features.npy')).astype('float32')\n",
    "\n",
    "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
    "\n",
    "image_tensor = torch.tensor(dinov3_image_features).to(device)\n",
    "text_tensor  = torch.tensor(minilm_text_features).to(device)\n",
    "\n",
    "# Chuẩn hóa L2\n",
    "image_norm = image_tensor / image_tensor.norm(dim=-1, keepdim=True)\n",
    "text_norm  = text_tensor  / text_tensor.norm(dim=-1,  keepdim=True)\n",
    "\n",
    "# pHash của toàn bộ dataset\n",
    "phash_strings = candidate_df['image_phash'].values\n",
    "phash_ints = np.array([int(h, 16) for h in phash_strings], dtype=np.uint64)\n",
    "\n",
    "# ── Hàm tính Hamming distance batch ─────────────────────────────────────────\n",
    "def compute_hamming_batch(batch_phash, all_phash):\n",
    "    \"\"\"Tính số bit khác nhau giữa batch và toàn bộ dataset (bit-twiddling).\"\"\"\n",
    "    x = batch_phash[:, None] ^ all_phash[None, :]\n",
    "    x = (x & 0x5555555555555555) + ((x >> 1) & 0x5555555555555555)\n",
    "    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)\n",
    "    x = (x & 0x0F0F0F0F0F0F0F0F) + ((x >> 4) & 0x0F0F0F0F0F0F0F0F)\n",
    "    x = (x & 0x00FF00FF00FF00FF) + ((x >> 8) & 0x00FF00FF00FF00FF)\n",
    "    x = (x & 0x0000FFFF0000FFFF) + ((x >> 16) & 0x0000FFFF0000FFFF)\n",
    "    return ((x & 0x00000000FFFFFFFF) + (x >> 32)).astype(np.uint8)\n",
    "\n",
    "# ── Hàm tính mAP@5 trên một tập chỉ số cho trước ───────────────────────────\n",
    "def compute_map5_on_subset(subset_indices, alpha, phash_threshold, top_k=50):\n",
    "    \"\"\"Tính mAP@5 cho subset_indices với bộ tham số alpha và phash_threshold.\"\"\"\n",
    "    labels = candidate_df['label_group'].values\n",
    "    group_to_indices = {}\n",
    "    for idx, lb in enumerate(labels):\n",
    "        group_to_indices.setdefault(lb, []).append(idx)\n",
    "\n",
    "    ap5_scores = []\n",
    "    BATCH = 256\n",
    "\n",
    "    for b_start in range(0, len(subset_indices), BATCH):\n",
    "        b_end   = min(b_start + BATCH, len(subset_indices))\n",
    "        b_idx   = subset_indices[b_start:b_end]  # chỉ số trong candidate_df\n",
    "\n",
    "        img_sim = torch.matmul(image_norm[b_idx], image_norm.T).cpu().numpy().astype(np.float32)\n",
    "        txt_sim = torch.matmul(text_norm[b_idx],  text_norm.T).cpu().numpy().astype(np.float32)\n",
    "        sim = alpha * img_sim + (1 - alpha) * txt_sim\n",
    "\n",
    "        # pHash boost\n",
    "        hdist = compute_hamming_batch(phash_ints[b_idx], phash_ints)\n",
    "        sim[hdist <= phash_threshold] += 0.5\n",
    "        sim[hdist == 0] += 0.5\n",
    "\n",
    "        for local_i, global_i in enumerate(b_idx):\n",
    "            sim_row = sim[local_i]\n",
    "            sim_row[global_i] = -9999  # loại bỏ chính nó\n",
    "            top_idx = np.argsort(-sim_row)[:top_k]\n",
    "\n",
    "            query_label = labels[global_i]\n",
    "            gt_all = np.array(group_to_indices[query_label])\n",
    "            gt_without_self = gt_all[gt_all != global_i]\n",
    "            gt_len = len(gt_without_self)\n",
    "\n",
    "            # Fix 1: gt_len = 0 → bỏ qua (không append 0)\n",
    "            if gt_len == 0:\n",
    "                continue\n",
    "\n",
    "            is_relevant = np.isin(top_idx, gt_without_self)\n",
    "            is_rel_5    = is_relevant[:5]\n",
    "            hit_pos = np.where(is_rel_5)[0] + 1\n",
    "\n",
    "            if len(hit_pos) > 0:\n",
    "                ap_5 = np.sum(np.arange(1, len(hit_pos) + 1) / hit_pos) / min(5, gt_len)\n",
    "            else:\n",
    "                ap_5 = 0.0\n",
    "            ap5_scores.append(ap_5)\n",
    "\n",
    "        del img_sim, txt_sim, sim, hdist\n",
    "        gc.collect()\n",
    "\n",
    "    return float(np.mean(ap5_scores)) if ap5_scores else 0.0\n",
    "\n",
    "# ── Grid Search trên VAL SET ─────────────────────────────────────────────────\n",
    "# Chạy grid search trên val set để tìm alpha và threshold tốt nhất\n",
    "alpha_grid     = [0.5, 0.6, 0.7, 0.8]\n",
    "threshold_grid = [2, 5, 8, 10]\n",
    "\n",
    "best_alpha     = 0.7\n",
    "best_threshold = 5\n",
    "best_map_val   = 0.0\n",
    "grid_results   = []\n",
    "\n",
    "print(\"Grid search alpha × pHash threshold trên VAL set...\")\n",
    "print(f\"{'Alpha':>6} | {'Threshold':>9} | {'mAP@5':>7}\")\n",
    "print(\"-\" * 32)\n",
    "\n",
    "for alpha in alpha_grid:\n",
    "    for threshold in threshold_grid:\n",
    "        map_val = compute_map5_on_subset(val_indices, alpha, threshold)\n",
    "        grid_results.append({'alpha': alpha, 'threshold': threshold, 'map5_val': round(map_val, 4)})\n",
    "        print(f\"{alpha:>6.1f} | {threshold:>9d} | {map_val:>7.4f}\")\n",
    "        if map_val > best_map_val:\n",
    "            best_map_val   = map_val\n",
    "            best_alpha     = alpha\n",
    "            best_threshold = threshold\n",
    "\n",
    "print(f\"\\n✅ Best: alpha={best_alpha}, threshold={best_threshold}, val mAP@5={best_map_val:.4f}\")\n",
    "\n",
    "# Lưu grid results\n",
    "grid_df = pd.DataFrame(grid_results)\n",
    "grid_df.to_csv(os.path.join(output_dir, 'grid_search_results_Hung.csv'), index=False)\n",
    "print(f\"Đã lưu grid_search_results_Hung.csv\")\n",
]

cell_gridsearch_code = make_code_cell(cell_gridsearch_code_lines, "gridsearch_code_01")

# ── Cell: Tính similarity và lưu sorted indices với tham số tốt nhất ─────────
cell_sim_note = make_markdown_cell(
    "## TÍNH TOÁN MA TRẬN TƯƠNG ĐỒNG (với tham số tối ưu từ Grid Search)",
    "md_sim_optimized_01"
)

cell_sim_code_lines = [
    "import os\n",
    "import gc\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import torch\n",
    "from tqdm import tqdm\n",
    "\n",
    "csv_path   = \"/content/drive/MyDrive/DoAnPython/DuLieuPython/train.csv\"\n",
    "output_dir = \"/content/drive/MyDrive/DoAnPython/DuLieuPython\"\n",
    "\n",
    "candidate_df           = pd.read_csv(csv_path)\n",
    "dinov3_image_features  = np.load(os.path.join(output_dir, 'dinov3_image_features.npy')).astype('float32')\n",
    "minilm_text_features   = np.load(os.path.join(output_dir, 'minilm_text_features.npy')).astype('float32')\n",
    "\n",
    "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
    "\n",
    "image_tensor = torch.tensor(dinov3_image_features).to(device)\n",
    "text_tensor  = torch.tensor(minilm_text_features).to(device)\n",
    "\n",
    "image_norm = image_tensor / image_tensor.norm(dim=-1, keepdim=True)\n",
    "text_norm  = text_tensor  / text_tensor.norm(dim=-1,  keepdim=True)\n",
    "\n",
    "phash_strings = candidate_df['image_phash'].values\n",
    "phash_ints = np.array([int(h, 16) for h in phash_strings], dtype=np.uint64)\n",
    "\n",
    "# ── Dùng tham số tốt nhất từ grid search ──────────────────────────────────\n",
    "# Nếu chưa chạy grid search, dùng giá trị mặc định đã được tối ưu theo kinh nghiệm\n",
    "try:\n",
    "    ALPHA          = best_alpha      # từ cell grid search\n",
    "    PHASH_THRESH   = best_threshold\n",
    "except NameError:\n",
    "    ALPHA          = 0.7             # fallback\n",
    "    PHASH_THRESH   = 5\n",
    "\n",
    "TOP_K      = 50\n",
    "BATCH_SIZE = 1024\n",
    "num_samples = len(candidate_df)\n",
    "\n",
    "print(f\"Chạy similarity với alpha={ALPHA}, pHash threshold={PHASH_THRESH}\")\n",
    "\n",
    "all_sorted_indices = []\n",
    "torch.cuda.empty_cache()\n",
    "gc.collect()\n",
    "\n",
    "for start_idx in tqdm(range(0, num_samples, BATCH_SIZE)):\n",
    "    end_idx = min(start_idx + BATCH_SIZE, num_samples)\n",
    "\n",
    "    # 1. Tính similarity trên GPU\n",
    "    img_sim_batch = torch.matmul(image_norm[start_idx:end_idx], image_norm.T)\n",
    "    txt_sim_batch = torch.matmul(text_norm[start_idx:end_idx],  text_norm.T)\n",
    "\n",
    "    # Kết hợp với alpha tối ưu từ grid search\n",
    "    sim_batch = (ALPHA * img_sim_batch + (1 - ALPHA) * txt_sim_batch).cpu().numpy().astype(np.float32)\n",
    "\n",
    "    del img_sim_batch, txt_sim_batch\n",
    "    torch.cuda.empty_cache()\n",
    "\n",
    "    # 2. Hamming distance (bit-twiddling)\n",
    "    batch_phash = phash_ints[start_idx:end_idx]\n",
    "    phash_diff  = batch_phash[:, None] ^ phash_ints[None, :]\n",
    "\n",
    "    x = phash_diff\n",
    "    x = (x & 0x5555555555555555) + ((x >> 1) & 0x5555555555555555)\n",
    "    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)\n",
    "    x = (x & 0x0F0F0F0F0F0F0F0F) + ((x >> 4) & 0x0F0F0F0F0F0F0F0F)\n",
    "    x = (x & 0x00FF00FF00FF00FF) + ((x >> 8) & 0x00FF00FF00FF00FF)\n",
    "    x = (x & 0x0000FFFF0000FFFF) + ((x >> 16) & 0x0000FFFF0000FFFF)\n",
    "    hamming_dist = ((x & 0x00000000FFFFFFFF) + (x >> 32)).astype(np.uint8)\n",
    "\n",
    "    del phash_diff, x\n",
    "\n",
    "    # 3. pHash boost với threshold tối ưu từ grid search (thay vì cố định = 2)\n",
    "    sim_batch[hamming_dist <= PHASH_THRESH] += 0.5\n",
    "    sim_batch[hamming_dist == 0]            += 0.5\n",
    "\n",
    "    del hamming_dist\n",
    "\n",
    "    # 4. Lấy Top K\n",
    "    sorted_indices = np.argsort(-sim_batch, axis=-1)[:, :TOP_K].astype(np.int32)\n",
    "    all_sorted_indices.append(sorted_indices)\n",
    "\n",
    "    del sim_batch\n",
    "    gc.collect()\n",
    "\n",
    "all_sorted_indices = np.vstack(all_sorted_indices)\n",
    "np.save(os.path.join(output_dir, 'all_sorted_indices.npy'), all_sorted_indices)\n",
    "\n",
    "print(f\"Thành công! Ma trận đầu ra: {all_sorted_indices.shape}\")\n",
]

cell_sim_code = make_code_cell(cell_sim_code_lines, "sim_code_optimized_01")

# ── Cell: Metric Note ─────────────────────────────────────────────────────────
cell_metric_note = make_markdown_cell(
    "## TÍNH TOÁN METRIC\n\n"
    "🤖 **AI Note**: Metric được tính **tách biệt** trên val set và test set:\n"
    "- **Val set**: Đã dùng để chọn tham số → không dùng làm kết quả báo cáo\n"
    "- **Test set**: Chưa bao giờ được nhìn thấy → metric trung thực để báo cáo\n\n"
    "**Fix quan trọng**: `gt_len = 0` (items không có cặp matching) sẽ bị `continue` – "
    "không tính vào mAP vì chúng không có ground truth để đánh giá.",
    "ai_note_metric_01"
)

cell_metric_code_lines = [
    "import os\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from tqdm import tqdm\n",
    "\n",
    "csv_path   = \"/content/drive/MyDrive/DoAnPython/DuLieuPython/train.csv\"\n",
    "output_dir = \"/content/drive/MyDrive/DoAnPython/DuLieuPython\"\n",
    "\n",
    "candidate_df       = pd.read_csv(csv_path)\n",
    "all_sorted_indices = np.load(os.path.join(output_dir, 'all_sorted_indices.npy'))\n",
    "\n",
    "labels = candidate_df['label_group'].values\n",
    "\n",
    "# Tạo mapping label_group → danh sách index\n",
    "group_to_indices = {}\n",
    "for idx, lb in enumerate(labels):\n",
    "    group_to_indices.setdefault(lb, []).append(idx)\n",
    "\n",
    "# ── Hàm tính metrics trên một subset ─────────────────────────────────────────\n",
    "def evaluate_on_subset(subset_indices, split_name='Full'):\n",
    "    \"\"\"Tính Precision@K, Recall@K, mAP@5 cho subset_indices.\"\"\"\n",
    "    ap5_scores = []\n",
    "    p1, r1 = [], []\n",
    "    p3, r3 = [], []\n",
    "    p5, r5 = [], []\n",
    "    p10, r10 = [], []\n",
    "\n",
    "    for global_i in tqdm(subset_indices, desc=f'Evaluating {split_name}'):\n",
    "        retrieved_indices = all_sorted_indices[global_i]\n",
    "        query_label       = labels[global_i]\n",
    "        gt_all            = np.array(group_to_indices[query_label])\n",
    "\n",
    "        retrieved_indices = retrieved_indices[retrieved_indices != global_i]\n",
    "        gt_without_self   = gt_all[gt_all != global_i]\n",
    "        gt_len            = len(gt_without_self)\n",
    "\n",
    "        # ✅ Fix 1: gt_len = 0 → continue (không append 0 kéo mAP xuống giả tạo)\n",
    "        # Items chỉ có 1 posting duy nhất không có ground truth, không nên tính vào metric\n",
    "        if gt_len == 0:\n",
    "            continue\n",
    "\n",
    "        is_relevant = np.isin(retrieved_indices, gt_without_self)\n",
    "\n",
    "        h1 = is_relevant[:1].sum()\n",
    "        p1.append(h1 / 1);  r1.append(h1 / gt_len)\n",
    "\n",
    "        h3 = is_relevant[:3].sum()\n",
    "        p3.append(h3 / 3);  r3.append(h3 / gt_len)\n",
    "\n",
    "        h5 = is_relevant[:5].sum()\n",
    "        p5.append(h5 / 5);  r5.append(h5 / gt_len)\n",
    "\n",
    "        h10 = is_relevant[:10].sum()\n",
    "        p10.append(h10 / 10); r10.append(h10 / gt_len)\n",
    "\n",
    "        is_rel_5  = is_relevant[:5]\n",
    "        hit_pos   = np.where(is_rel_5)[0] + 1\n",
    "        if len(hit_pos) > 0:\n",
    "            ap_5 = np.sum(np.arange(1, len(hit_pos) + 1) / hit_pos) / min(5, gt_len)\n",
    "        else:\n",
    "            ap_5 = 0.0\n",
    "        ap5_scores.append(ap_5)\n",
    "\n",
    "    df_metrics = pd.DataFrame({\n",
    "        'K':         [1, 3, 5, 10],\n",
    "        'Precision': [np.mean(p1), np.mean(p3), np.mean(p5), np.mean(p10)],\n",
    "        'Recall':    [np.mean(r1), np.mean(r3), np.mean(r5), np.mean(r10)],\n",
    "    })\n",
    "    df_metrics[['Precision', 'Recall']] = df_metrics[['Precision', 'Recall']].round(4)\n",
    "    map_at_5 = round(np.mean(ap5_scores), 4) if ap5_scores else 0.0\n",
    "\n",
    "    return df_metrics, map_at_5\n",
    "\n",
    "# ── Tính metric trên VAL SET ─────────────────────────────────────────────────\n",
    "# Khôi phục val_indices và test_indices từ split ở cell trước\n",
    "try:\n",
    "    val_metrics_df, val_map5 = evaluate_on_subset(val_indices, 'Val')\n",
    "    print(\"\\n📊 VAL SET METRICS:\")\n",
    "    print(val_metrics_df.to_string(index=False))\n",
    "    print(f\"mAP@5 (val): {val_map5}\")\n",
    "except NameError:\n",
    "    print(\"(Bỏ qua val metrics – chưa có val_indices)\")\n",
    "\n",
    "# ── Tính metric trên TEST SET (metric báo cáo chính thức) ────────────────────\n",
    "try:\n",
    "    test_metrics_df, test_map5 = evaluate_on_subset(test_indices, 'Test')\n",
    "    print(\"\\n📊 TEST SET METRICS (Báo cáo chính thức):\")\n",
    "    print(test_metrics_df.to_string(index=False))\n",
    "    print(f\"mAP@5 (test): {test_map5}\")\n",
    "except NameError:\n",
    "    print(\"(Bỏ qua test metrics – chưa có test_indices)\")\n",
    "\n",
    "# ── Tính metric toàn bộ (giống notebook gốc – để so sánh) ───────────────────\n",
    "num_samples = len(candidate_df)\n",
    "all_indices = np.arange(num_samples)\n",
    "\n",
    "full_metrics_df, full_map5 = evaluate_on_subset(all_indices, 'Full')\n",
    "print(\"\\n📊 FULL DATASET METRICS:\")\n",
    "print(full_metrics_df.to_string(index=False))\n",
    "print(f\"mAP@5 (full): {full_map5}\")\n",
    "\n",
    "# ── Lưu CSV metric để dùng trong bảng so sánh Cell 9 ────────────────────────\n",
    "full_metrics_df['mAP@5'] = full_map5\n",
    "full_metrics_df['model'] = 'DINOv3 + MultilingualMiniLM'\n",
    "full_metrics_df['split'] = 'full'\n",
    "\n",
    "try:\n",
    "    test_metrics_df['mAP@5'] = test_map5\n",
    "    test_metrics_df['model'] = 'DINOv3 + MultilingualMiniLM'\n",
    "    test_metrics_df['split'] = 'test'\n",
    "    combined = pd.concat([full_metrics_df, test_metrics_df], ignore_index=True)\n",
    "except NameError:\n",
    "    combined = full_metrics_df\n",
    "\n",
    "csv_out = os.path.join(output_dir, 'metrics_Hung_DINOv3_MultilingualMiniLM.csv')\n",
    "combined.to_csv(csv_out, index=False)\n",
    "print(f\"\\n✅ Đã lưu metric CSV: {csv_out}\")\n",
]

cell_metric_code = make_code_cell(cell_metric_code_lines, "metric_code_v2_01")

# ─────────────────────────────────────────────────────────────────────────────
# Tái cấu trúc notebook: thay thế các cell cũ bằng cell mới đã fix
# ─────────────────────────────────────────────────────────────────────────────
# Tìm vị trí cell model load (id: 5f5feb5b)
# Tìm vị trí cell similarity (id: bb367483)
# Tìm vị trí cell metric (id: 476c1f06)

MODEL_CELL_ID   = "5f5feb5b"
SIM_CELL_ID     = "bb367483"
METRIC_CELL_ID  = "476c1f06"

# Tìm index các cell cần xử lý
idx_model  = next((i for i, c in enumerate(nb["cells"]) if c.get("id") == MODEL_CELL_ID),  None)
idx_sim    = next((i for i, c in enumerate(nb["cells"]) if c.get("id") == SIM_CELL_ID),    None)
idx_metric = next((i for i, c in enumerate(nb["cells"]) if c.get("id") == METRIC_CELL_ID), None)

print(f"Cell model  @ index {idx_model}")
print(f"Cell sim    @ index {idx_sim}")
print(f"Cell metric @ index {idx_metric}")

# ── Chèn AI note trước cell load model ──────────────────────────────────────
if idx_model is not None:
    nb["cells"].insert(idx_model, ai_note_model)
    # Re-calc indices sau khi insert
    idx_sim    = next((i for i, c in enumerate(nb["cells"]) if c.get("id") == SIM_CELL_ID),    None)
    idx_metric = next((i for i, c in enumerate(nb["cells"]) if c.get("id") == METRIC_CELL_ID), None)

# ── Xoá cell similarity cũ, chèn các cell mới vào vị trí đó ────────────────
if idx_sim is not None:
    nb["cells"].pop(idx_sim)

    # Tìm markdown "TÍNH TOÁN MA TRẬN TƯƠNG ĐỒNG" ngay trước → xoá nó luôn
    # (sẽ được thay bằng markdown mới có nội dung chuẩn hơn)
    md_before = idx_sim - 1
    if md_before >= 0 and nb["cells"][md_before].get("cell_type") == "markdown":
        nb["cells"].pop(md_before)
        idx_sim = md_before  # vị trí để chèn

    # Chèn theo thứ tự: split_note, split_code, gridsearch_note, gridsearch_code, sim_note, sim_code
    to_insert = [
        cell_split_note,
        cell_split_code,
        cell_gridsearch_note,
        cell_gridsearch_code,
        cell_sim_note,
        cell_sim_code,
    ]
    for offset, cell in enumerate(to_insert):
        nb["cells"].insert(idx_sim + offset, cell)

    # Re-calc metric index
    idx_metric = next((i for i, c in enumerate(nb["cells"]) if c.get("id") == METRIC_CELL_ID), None)
    print(f"Sau insert, cell metric @ index {idx_metric}")

# ── Xoá cell metric cũ, chèn cell metric mới đã fix ─────────────────────────
if idx_metric is not None:
    nb["cells"].pop(idx_metric)

    # Tìm markdown "TÍNH TOÁN METRIC" ngay trước → thay bằng AI note mới
    md_before = idx_metric - 1
    if md_before >= 0 and nb["cells"][md_before].get("cell_type") == "markdown":
        nb["cells"].pop(md_before)
        idx_metric = md_before

    nb["cells"].insert(idx_metric,     cell_metric_note)
    nb["cells"].insert(idx_metric + 1, cell_metric_code)

print(f"Tổng số cell sau patch: {len(nb['cells'])}")

# ─────────────────────────────────────────────────────────────────────────────
# Ghi file ra
# ─────────────────────────────────────────────────────────────────────────────
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\n✅ Patch hoàn tất! Notebook đã được lưu: {OUT_PATH}")
print("\nTóm tắt thay đổi:")
print("  [Fix 1] gt_len=0 → continue (không append 0)")
print("  [Fix 2] all-MiniLM-L6-v2 → paraphrase-multilingual-MiniLM-L12-v2")
print("  [Fix 3] pHash threshold: 2 → grid search [2,5,8,10]")
print("  [Fix 4] alpha: 0.7 cố định → grid search [0.5,0.6,0.7,0.8]")
print("  [NEW]   Chia val/test split (80/10/10)")
print("  [NEW]   Lưu CSV metric: metrics_Hung_DINOv3_MultilingualMiniLM.csv")
print("  [NEW]   Grid search results: grid_search_results_Hung.csv")
print("  [NEW]   Thêm AI notes/markdown giải thích từng bước")
