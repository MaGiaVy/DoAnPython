import json

# Load original notebook
with open('notebooks/Tuan4_GiaVy_Dinov3+miniML - Copy.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

old_cells = nb['cells']

# Create new cells list
new_cells = []

# Keep cells 0 to 10 (0-indexed)
# 0: Empty markdown
# 1: pip install faiss-cpu
# 2: import libs
# 3: markdown "DUNG GG COLAB..."
# 4: mount drive
# 5: DATA_DIR setup
# 6: markdown "TIEN XU LY..."
# 7: ShopeeDataset class definition (old version)
# 8: pd.read_csv + head
# 9: df.info()
# 10: candidate_df = pd.read_csv(CSV_PATH)
for i in range(11):
    new_cells.append(old_cells[i])

# Insert Step 1 (Phan chia Validation / Test Set)
step1_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "## 📊 Bước 1: Phân chia Validation / Test Set\n",
        "- **Gallery:** Toàn bộ 34,250 ảnh (không thay đổi)\n",
        "- **Val Set (20%):** Dùng để tune tham số α trong Grid Search\n",
        "- **Test Set (80%):** Chỉ dùng một lần duy nhất để báo cáo kết quả cuối\n",
        "- **Không dùng stratify** vì số lượng nhóm (11,014) nhiều hơn số lượng mẫu Val (6,850)"
    ]
}

step1_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================\n",
        "# 📊 BƯỚC 1: PHÂN CHIA DỮ LIỆU VALIDATION / TEST\n",
        "# ============================================================\n",
        "print('=' * 60)\n",
        "print('BƯỚC 1: PHÂN CHIA DỮ LIỆU VALIDATION / TEST')\n",
        "print('=' * 60)\n",
        "\n",
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "# --- 1.2: Phân chia Validation / Test ---\n",
        "print('\\n✂️  Đang phân chia dữ liệu...')\n",
        "print('   - Val Set  : 20% (dùng để tune tham số)')\n",
        "print('   - Test Set : 80% (CHỈ DÙNG ĐỂ ĐÁNH GIÁ CUỐI CÙNG)')\n",
        "\n",
        "val_query_df, test_query_df = train_test_split(\n",
        "    candidate_df,\n",
        "    test_size=0.8,\n",
        "    random_state=42,\n",
        ")\n",
        "\n",
        "# Reset index để tránh lỗi khi dùng .at[]\n",
        "val_query_df  = val_query_df.reset_index(drop=True)\n",
        "test_query_df = test_query_df.reset_index(drop=True)\n",
        "\n",
        "# --- 1.3: Lưu kết quả phân chia ---\n",
        "val_csv_path  = os.path.join(RESULTS, 'val_query.csv')\n",
        "test_csv_path = os.path.join(RESULTS, 'test_query.csv')\n",
        "val_query_df.to_csv(val_csv_path, index=False)\n",
        "test_query_df.to_csv(test_csv_path, index=False)\n",
        "\n",
        "# --- 1.4: In kết quả ---\n",
        "print(f'\\n✅ PHÂN CHIA HOÀN TẤT!')\n",
        "print(f'   📁 Gallery (toàn bộ)  : {len(candidate_df):,} ảnh  (100%)')\n",
        "print(f'   📁 Validation Set     : {len(val_query_df):,} ảnh  (20%)  → Lưu tại: val_query.csv')\n",
        "print(f'   📁 Test Set           : {len(test_query_df):,} ảnh  (80%)  → Lưu tại: test_query.csv')\n",
        "\n",
        "# Kiểm tra phân phối\n",
        "val_labels  = val_query_df['label_group'].nunique()\n",
        "test_labels = test_query_df['label_group'].nunique()\n",
        "print(f'\\n📊 Kiểm tra phân phối:')\n",
        "print(f'   Label groups trong Val Set  : {val_labels:,}')\n",
        "print(f'   Label groups trong Test Set : {test_labels:,}')\n",
        "print(f'   Label groups trong Gallery  : {candidate_df[\"label_group\"].nunique():,}')\n",
        "\n",
        "print(f'\\n⚠️  NHẮC NHỞ QUAN TRỌNG:')\n",
        "print(f'   ✅ Bước 4 (Grid Search α) → CHỈ DÙNG val_query.csv')\n",
        "print(f'   ✅ Bước 5 (Đánh giá cuối) → CHỈ DÙNG test_query.csv — TUYỆT ĐỐI KHÔNG TUNE!')"
    ]
}

new_cells.append(step1_md)
new_cells.append(step1_code)

# Keep cells 11 to 14 (load models and extract features)
# In the new list they will occupy indexes 13 to 16
for i in range(11, 15):
    new_cells.append(old_cells[i])

# Add Step 4 (Grid Search alpha on Val Set)
step4_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "## 🔍 Bước 4: Grid Search tham số α trên Validation Set\n",
        "- Chỉ dùng `val_query.csv` để tìm α tối ưu\n",
        "- Công thức fusion: `sim = α * img_sim + (1 - α) * txt_sim`\n",
        "- Sau đó áp dụng pHash boosting"
    ]
}

step4_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================\n",
        "# 🔍 BƯỚC 4: GRID SEARCH α TRÊN VALIDATION SET\n",
        "# ============================================================\n",
        "print('=' * 60)\n",
        "print('BƯỚC 4: GRID SEARCH α — WEIGHTED LATE FUSION (VAL SET)')\n",
        "print('=' * 60)\n",
        "\n",
        "import gc\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import torch\n",
        "import os\n",
        "from tqdm import tqdm\n",
        "\n",
        "csv_path = \"/content/drive/MyDrive/DoAnPython/DuLieuPython/train.csv\"\n",
        "output_dir = \"/content/drive/MyDrive/DoAnPython/DuLieuPython\"\n",
        "\n",
        "candidate_df = pd.read_csv(csv_path)\n",
        "dinov3_image_features = np.load(os.path.join(output_dir, \"dinov3_image_features.npy\")).astype('float32')\n",
        "minilm_text_features = np.load(os.path.join(output_dir, \"minilm_text_features.npy\")).astype('float32')\n",
        "\n",
        "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
        "\n",
        "image_tensor = torch.tensor(dinov3_image_features).to(device)\n",
        "text_tensor = torch.tensor(minilm_text_features).to(device)\n",
        "\n",
        "# Chuẩn hóa L2\n",
        "image_norm = image_tensor / image_tensor.norm(dim=-1, keepdim=True)\n",
        "text_norm = text_tensor / text_tensor.norm(dim=-1, keepdim=True)\n",
        "\n",
        "phash_strings = candidate_df[\"image_phash\"].values\n",
        "phash_ints = np.array([int(h, 16) for h in phash_strings], dtype=np.uint64)\n",
        "\n",
        "# Lấy chỉ số val queries trong gallery\n",
        "posting_id_to_gidx = {\n",
        "    pid: idx for idx, pid in enumerate(candidate_df['posting_id'])\n",
        "}\n",
        "val_gallery_indices = np.array(\n",
        "    [posting_id_to_gidx[pid] for pid in val_query_df['posting_id']],\n",
        "    dtype=np.int64\n",
        ")\n",
        "\n",
        "labels = candidate_df[\"label_group\"].values\n",
        "group_to_indices = candidate_df[\"label_group\"].groupby(candidate_df[\"label_group\"]).indices\n",
        "\n",
        "# Định nghĩa hàm đánh giá nhanh theo Batch trên GPU\n",
        "def evaluate_subset(query_indices, query_df, alpha, top_k=50):\n",
        "    num_queries = len(query_indices)\n",
        "    BATCH_SIZE = 1024\n",
        "    ap5_scores = []\n",
        "    \n",
        "    for start_idx in range(0, num_queries, BATCH_SIZE):\n",
        "        end_idx = min(start_idx + BATCH_SIZE, num_queries)\n",
        "        batch_q_indices = query_indices[start_idx:end_idx]\n",
        "        \n",
        "        # 1. Tính similarity trên GPU\n",
        "        img_sim_batch = torch.matmul(image_norm[batch_q_indices], image_norm.T)\n",
        "        txt_sim_batch = torch.matmul(text_norm[batch_q_indices], text_norm.T)\n",
        "        sim_batch = (alpha * img_sim_batch + (1.0 - alpha) * txt_sim_batch).cpu().numpy().astype(np.float32)\n",
        "        \n",
        "        del img_sim_batch, txt_sim_batch\n",
        "        torch.cuda.empty_cache()\n",
        "        \n",
        "        # 2. Hamming distance của pHash (Thuật toán Bit-twiddling gốc)\n",
        "        batch_phash = phash_ints[batch_q_indices]\n",
        "        phash_diff = batch_phash[:, None] ^ phash_ints[None, :]\n",
        "        \n",
        "        x = phash_diff\n",
        "        x = (x & 0x5555555555555555) + ((x >> 1) & 0x5555555555555555)\n",
        "        x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)\n",
        "        x = (x & 0x0F0F0F0F0F0F0F0F) + ((x >> 4) & 0x0F0F0F0F0F0F0F0F)\n",
        "        x = (x & 0x00FF00FF00FF00FF) + ((x >> 8) & 0x00FF00FF00FF00FF)\n",
        "        x = (x & 0x0000FFFF0000FFFF) + ((x >> 16) & 0x0000FFFF0000FFFF)\n",
        "        hamming_dist = ((x & 0x00000000FFFFFFFF) + (x >> 32)).astype(np.uint8)\n",
        "        \n",
        "        del phash_diff, x\n",
        "        \n",
        "        sim_batch[hamming_dist <= 2] += 0.5\n",
        "        sim_batch[hamming_dist == 0] += 0.5\n",
        "        \n",
        "        del hamming_dist\n",
        "        \n",
        "        # Loại self-match\n",
        "        for local_idx, global_q_idx in enumerate(batch_q_indices):\n",
        "            sim_batch[local_idx, global_q_idx] = -999\n",
        "            \n",
        "        # Lấy top_k\n",
        "        sorted_indices = np.argsort(-sim_batch, axis=-1)[:, :top_k].astype(np.int32)\n",
        "        \n",
        "        del sim_batch\n",
        "        gc.collect()\n",
        "        \n",
        "        # Tính AP@5\n",
        "        for local_idx, global_q_idx in enumerate(batch_q_indices):\n",
        "            retrieved_indices = sorted_indices[local_idx]\n",
        "            query_label = labels[global_q_idx]\n",
        "            ground_truth_indices = group_to_indices[query_label]\n",
        "            gt_without_self = ground_truth_indices[ground_truth_indices != global_q_idx]\n",
        "            gt_len = len(gt_without_self)\n",
        "            \n",
        "            if gt_len > 0:\n",
        "                is_relevant = np.isin(retrieved_indices, gt_without_self)\n",
        "                is_relevant_5 = is_relevant[:5]\n",
        "                hit_positions = np.where(is_relevant_5)[0] + 1\n",
        "                if len(hit_positions) > 0:\n",
        "                    hits_cum = np.arange(1, len(hit_positions) + 1)\n",
        "                    ap_5 = np.sum(hits_cum / hit_positions) / min(5, gt_len)\n",
        "                    ap5_scores.append(ap_5)\n",
        "                else:\n",
        "                    ap5_scores.append(0.0)\n",
        "            \n",
        "    return np.mean(ap5_scores) if ap5_scores else 0.0\n",
        "\n",
        "# Vòng lặp Grid Search alpha\n",
        "alphas = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]\n",
        "best_alpha = 0.7\n",
        "best_val_map = -1.0\n",
        "\n",
        "print(\"Đang chạy Grid Search alpha trên tập Validation...\")\n",
        "for alpha in alphas:\n",
        "    val_map = evaluate_subset(val_gallery_indices, val_query_df, alpha, top_k=50)\n",
        "    print(f\"  alpha = {alpha:.2f} -> val mAP@5 = {val_map:.4f}\")\n",
        "    if val_map > best_val_map:\n",
        "        best_val_map = val_map\n",
        "        best_alpha = alpha\n",
        "\n",
        "print(f\"\\n🏆 BEST alpha = {best_alpha:.2f} với val mAP@5 = {best_val_map:.4f}\")"
    ]
}

new_cells.append(step4_md)
new_cells.append(step4_code)

# Add Step 5 (Final Evaluation on Test Set)
step5_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "## 🏆 Bước 5: Đánh giá cuối cùng trên Test Set\n",
        "- Áp dụng `best_alpha` tìm được từ tập Val lên tập Test\n",
        "- Tính toán Precision@K, Recall@K và mAP@5\n",
        "- Lưu file `metrics_dinov3_minilm.csv` để đối chiếu"
    ]
}

step5_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================\n",
        "# 🏆 BƯỚC 5: ĐÁNH GIÁ CUỐI CÙNG TRÊN TEST SET\n",
        "# ============================================================\n",
        "print('=' * 60)\n",
        "print(f'BƯỚC 5: ĐÁNH GIÁ TRÊN TEST SET (alpha = {best_alpha:.2f})')\n",
        "print('=' * 60)\n",
        "\n",
        "test_gallery_indices = np.array(\n",
        "    [posting_id_to_gidx[pid] for pid in test_query_df['posting_id']],\n",
        "    dtype=np.int64\n",
        ")\n",
        "\n",
        "num_queries = len(test_gallery_indices)\n",
        "BATCH_SIZE = 1024\n",
        "ap5_scores = []\n",
        "p1, r1 = [], []\n",
        "p3, r3 = [], []\n",
        "p5, r5 = [], []\n",
        "p10, r10 = [], []\n",
        "\n",
        "print(\"Đang tính toán ma trận tương đồng và các metric trên Test Set...\")\n",
        "for start_idx in tqdm(range(0, num_queries, BATCH_SIZE)):\n",
        "    end_idx = min(start_idx + BATCH_SIZE, num_queries)\n",
        "    batch_q_indices = test_gallery_indices[start_idx:end_idx]\n",
        "    \n",
        "    # 1. Similarity\n",
        "    img_sim_batch = torch.matmul(image_norm[batch_q_indices], image_norm.T)\n",
        "    txt_sim_batch = torch.matmul(text_norm[batch_q_indices], text_norm.T)\n",
        "    sim_batch = (best_alpha * img_sim_batch + (1.0 - best_alpha) * txt_sim_batch).cpu().numpy().astype(np.float32)\n",
        "    \n",
        "    del img_sim_batch, txt_sim_batch\n",
        "    torch.cuda.empty_cache()\n",
        "    \n",
        "    # 2. Hamming distance\n",
        "    batch_phash = phash_ints[batch_q_indices]\n",
        "    phash_diff = batch_phash[:, None] ^ phash_ints[None, :]\n",
        "    \n",
        "    x = phash_diff\n",
        "    x = (x & 0x5555555555555555) + ((x >> 1) & 0x5555555555555555)\n",
        "    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)\n",
        "    x = (x & 0x0F0F0F0F0F0F0F0F) + ((x >> 4) & 0x0F0F0F0F0F0F0F0F)\n",
        "    x = (x & 0x00FF00FF00FF00FF) + ((x >> 8) & 0x00FF00FF00FF00FF)\n",
        "    x = (x & 0x0000FFFF0000FFFF) + ((x >> 16) & 0x0000FFFF0000FFFF)\n",
        "    hamming_dist = ((x & 0x00000000FFFFFFFF) + (x >> 32)).astype(np.uint8)\n",
        "    \n",
        "    del phash_diff, x\n",
        "    \n",
        "    sim_batch[hamming_dist <= 2] += 0.5\n",
        "    sim_batch[hamming_dist == 0] += 0.5\n",
        "    \n",
        "    del hamming_dist\n",
        "    \n",
        "    # Loại self-match\n",
        "    for local_idx, global_q_idx in enumerate(batch_q_indices):\n",
        "        sim_batch[local_idx, global_q_idx] = -999\n",
        "        \n",
        "    # Lấy top_10\n",
        "    sorted_indices = np.argsort(-sim_batch, axis=-1)[:, :10].astype(np.int32)\n",
        "    \n",
        "    del sim_batch\n",
        "    gc.collect()\n",
        "    \n",
        "    # Tính các metric\n",
        "    for local_idx, global_q_idx in enumerate(batch_q_indices):\n",
        "        retrieved_indices = sorted_indices[local_idx]\n",
        "        query_label = labels[global_q_idx]\n",
        "        ground_truth_indices = group_to_indices[query_label]\n",
        "        gt_without_self = ground_truth_indices[ground_truth_indices != global_q_idx]\n",
        "        gt_len = len(gt_without_self)\n",
        "        \n",
        "        if gt_len > 0:\n",
        "            is_relevant = np.isin(retrieved_indices, gt_without_self)\n",
        "            \n",
        "            h1 = is_relevant[:1].sum()\n",
        "            p1.append(h1 / 1)\n",
        "            r1.append(h1 / gt_len)\n",
        "            \n",
        "            h3 = is_relevant[:3].sum()\n",
        "            p3.append(h3 / 3)\n",
        "            r3.append(h3 / gt_len)\n",
        "            \n",
        "            h5 = is_relevant[:5].sum()\n",
        "            p5.append(h5 / 5)\n",
        "            r5.append(h5 / gt_len)\n",
        "            \n",
        "            h10 = is_relevant[:10].sum()\n",
        "            p10.append(h10 / 10)\n",
        "            r10.append(h10 / gt_len)\n",
        "            \n",
        "            is_relevant_5 = is_relevant[:5]\n",
        "            hit_positions = np.where(is_relevant_5)[0] + 1\n",
        "            if len(hit_positions) > 0:\n",
        "                hits_cum = np.arange(1, len(hit_positions) + 1)\n",
        "                ap_5 = np.sum(hits_cum / hit_positions) / min(5, gt_len)\n",
        "                ap5_scores.append(ap_5)\n",
        "            else:\n",
        "                ap5_scores.append(0.0)\n",
        "\n",
        "summary_metrics = pd.DataFrame(\n",
        "    {\n",
        "        \"K\": [1, 3, 5, 10],\n",
        "        \"Precision\": [np.mean(p1), np.mean(p3), np.mean(p5), np.mean(p10)],\n",
        "        \"Recall\": [np.mean(r1), np.mean(r3), np.mean(r5), np.mean(r10)],\n",
        "    }\n",
        ")\n",
        "\n",
        "summary_metrics[[\"Precision\", \"Recall\"]] = summary_metrics[[\"Precision\", \"Recall\"]].round(4)\n",
        "map_at_5 = round(np.mean(ap5_scores), 4)\n",
        "\n",
        "print(\"\\n=== KẾT QUẢ CUỐI — DINOv3 + MiniML ===\")\n",
        "print(summary_metrics.to_string(index=False))\n",
        "print(f\"mAP@5: {map_at_5}\")\n",
        "\n",
        "# Lưu metric ra CSV để file chung đọc\n",
        "summary_metrics['mAP@5'] = map_at_5\n",
        "summary_metrics['method'] = f'DINOv3 + MultilingualMiniLM + pHash (alpha={best_alpha:.2f})'\n",
        "summary_metrics.to_csv(os.path.join(output_dir, 'metrics_dinov3_minilm.csv'), index=False)\n",
        "print('\\nĐã lưu metrics_dinov3_minilm.csv!')"
    ]
}

new_cells.append(step5_md)
new_cells.append(step5_code)

# Update the cells in the notebook
nb['cells'] = new_cells

# Write to output file
with open('notebooks/Tuan4_GiaVy_Dinov3+miniML.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done writing notebooks/Tuan4_GiaVy_Dinov3+miniML.ipynb")
