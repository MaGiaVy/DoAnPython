"""
patch_hung_final.py  –  Sửa đúng theo CLAUDE.md, từ đầu hoàn toàn.

SỬA 1: all-MiniLM-L6-v2 → paraphrase-multilingual-MiniLM-L12-v2 (2 chỗ)
        file lưu features: minilm_text_features.npy → multilingual_minilm_text_features.npy
SỬA 2: Thêm cell chia val/test split (20/80)
SỬA 3: Thêm cell grid search alpha trên val set
SỬA 4: Thay cell metric cũ → tính trên test set + fix gt_len=0 + lưu CSV
SỬA 5: Thêm 2 markdown cell cuối (Ghi chú AI + Kế hoạch tuần 5)
"""

import json, re

NB = r"d:\New folder (3)\project\notebooks\Nhom3thangcuti_Tuan4\notebooks\Tuan4_Hung_Dinov3+miniML.ipynb"

# ──────────────────────────────────────────────────────────────────────────────
# 0. Load notebook GỐC (trước mọi patch) - nếu đã bị v2/v3 thay đổi thì phải
#    restore lại từ đầu. Ta sẽ làm việc trực tiếp trên JSON hiện tại.
# ──────────────────────────────────────────────────────────────────────────────
with open(NB, encoding="utf-8") as f:
    nb = json.load(f)

def make_md(src, cid):
    return {"cell_type":"markdown","id":cid,"metadata":{},"source":[src]}

def make_code(lines, cid):
    return {"cell_type":"code","execution_count":None,"id":cid,
            "metadata":{},"outputs":[],"source":lines}

def get_src(cell):
    return "".join(cell["source"])

def find_by_id(cells, cid):
    return next((i for i,c in enumerate(cells) if c.get("id")==cid), None)

def find_by_content(cells, keyword):
    return next((i for i,c in enumerate(cells) if keyword in get_src(c)), None)

print("Notebook gốc:", len(nb["cells"]), "cells")

# ──────────────────────────────────────────────────────────────────────────────
# SỬA 1a: Đổi tên model MiniLM (toàn bộ file)
# ──────────────────────────────────────────────────────────────────────────────
OLD_MODEL = "all-MiniLM-L6-v2"
NEW_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
cnt = 0
for cell in nb["cells"]:
    new_src = []
    for line in cell["source"]:
        if OLD_MODEL in line:
            new_src.append(line.replace(OLD_MODEL, NEW_MODEL))
            cnt += 1
        else:
            new_src.append(line)
    cell["source"] = new_src
print(f"[SỬA 1a] Đổi model name: {cnt} dòng")

# ──────────────────────────────────────────────────────────────────────────────
# SỬA 1b: Đổi tên file lưu/load text features
# ──────────────────────────────────────────────────────────────────────────────
OLD_FILE = "minilm_text_features.npy"
NEW_FILE = "multilingual_minilm_text_features.npy"
cnt = 0
for cell in nb["cells"]:
    new_src = []
    for line in cell["source"]:
        if OLD_FILE in line:
            new_src.append(line.replace(OLD_FILE, NEW_FILE))
            cnt += 1
        else:
            new_src.append(line)
    cell["source"] = new_src
print(f"[SỬA 1b] Đổi tên file features: {cnt} dòng")

# ──────────────────────────────────────────────────────────────────────────────
# SỬA 1c: Giảm batch_size encode cho model lớn hơn
# ──────────────────────────────────────────────────────────────────────────────
for cell in nb["cells"]:
    cell["source"] = [
        l.replace("batch_size=256","batch_size=128") if "batch_size=256" in l else l
        for l in cell["source"]
    ]

# ──────────────────────────────────────────────────────────────────────────────
# Tìm cell metric gốc (cell cần thay thế - SỬA 4)
# Đặc điểm: có "for i in tqdm(range(num_samples))" HOẶC là metric_code_v2_01
# ──────────────────────────────────────────────────────────────────────────────
idx_metric_orig = find_by_content(nb["cells"], "for i in tqdm(range(num_samples))")
if idx_metric_orig is None:
    idx_metric_orig = find_by_id(nb["cells"], "metric_code_v2_01")
if idx_metric_orig is None:
    idx_metric_orig = find_by_id(nb["cells"], "476c1f06")
print(f"Cell metric gốc @ index {idx_metric_orig}")

# Tìm markdown trước nó (tiêu đề "TÍNH TOÁN METRIC")
if idx_metric_orig and idx_metric_orig > 0:
    prev = nb["cells"][idx_metric_orig-1]
    if prev.get("cell_type") == "markdown" and ("METRIC" in get_src(prev).upper() or "AI Note" in get_src(prev)):
        idx_metric_md = idx_metric_orig - 1
    else:
        idx_metric_md = None
else:
    idx_metric_md = None

print(f"Markdown trước metric @ index {idx_metric_md}")

# ──────────────────────────────────────────────────────────────────────────────
# Xóa các cell từ v2/v3 patch nếu có (split, gridsearch, sim_optimized, metric_v2)
# Để notebook về trạng thái sạch trước khi insert đúng theo CLAUDE.md
# ──────────────────────────────────────────────────────────────────────────────
PATCH_IDS = {
    "ai_note_model_01", "split_code_01", "split_code_01",
    "ai_note_split_01", "ai_note_gridsearch_01", "gridsearch_code_01",
    "md_sim_optimized_01", "sim_code_optimized_01",
    "metric_code_v2_01",
    # markdown IDs
    "ai_note_metric_01",
}
before = len(nb["cells"])
nb["cells"] = [c for c in nb["cells"] if c.get("id") not in PATCH_IDS]
print(f"Xóa {before - len(nb['cells'])} cell patch cũ → còn {len(nb['cells'])} cells")

# Recalculate metric index sau khi xóa
idx_metric_orig = find_by_content(nb["cells"], "for i in tqdm(range(num_samples))")
if idx_metric_orig is None:
    idx_metric_orig = find_by_id(nb["cells"], "476c1f06")
print(f"Cell metric gốc (sau clean) @ index {idx_metric_orig}")

# ──────────────────────────────────────────────────────────────────────────────
# SỬA 2: Cell chia val/test split — chèn TRƯỚC cell metric
# (đúng theo CLAUDE.md: "Thêm cell mới TRƯỚC cell tính metric")
# ──────────────────────────────────────────────────────────────────────────────

CELL_SPLIT_MD = make_md(
    "## CHIA TẬP VALIDATION / TEST\n\n"
    "🤖 **AI ghi chú**: Chia dataset thành 2 phần:\n"
    "- **Validation (20%)**: dùng để tuning alpha, pHash threshold — không dùng để báo kết quả\n"
    "- **Test (80%)**: CHỈ dùng để báo kết quả cuối — không tune bất kỳ tham số nào",
    "md_split_claude"
)

CELL_SPLIT_CODE = make_code([
    "# ============================================================\n",
    "# CHIA TẬP VALIDATION / TEST\n",
    "# Validation (20%): dùng để tuning alpha, pHash threshold\n",
    "# Test (80%): CHỈ dùng để báo kết quả cuối — không tune!\n",
    "# ============================================================\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "print('=' * 55)\n",
    "print('CHIA TẬP VALIDATION / TEST')\n",
    "print('=' * 55)\n",
    "\n",
    "# Lọc bỏ nhóm chỉ có 1 ảnh (không đánh giá được)\n",
    "label_counts_all = candidate_df['label_group'].value_counts()\n",
    "valid_mask = candidate_df['label_group'].isin(\n",
    "    label_counts_all[label_counts_all >= 2].index\n",
    ")\n",
    "valid_indices = candidate_df[valid_mask].index.tolist()\n",
    "\n",
    "# Chia val/test — không dùng stratify vì nhiều nhóm chỉ có 2 ảnh\n",
    "val_idx, test_idx = train_test_split(\n",
    "    valid_indices,\n",
    "    test_size=0.8,\n",
    "    random_state=42\n",
    ")\n",
    "\n",
    "print(f'Gallery (toàn bộ)  : {len(candidate_df):,} ảnh')\n",
    "print(f'Validation set     : {len(val_idx):,} ảnh (20%) → dùng để tuning')\n",
    "print(f'Test set           : {len(test_idx):,} ảnh (80%) → báo kết quả cuối')\n",
    "\n",
    "# Lưu lại để tái sử dụng\n",
    "import json as _json\n",
    "split_info = {'val_idx': val_idx, 'test_idx': test_idx}\n",
    "with open(os.path.join(output_dir, 'split_indices.json'), 'w') as _f:\n",
    "    _json.dump(split_info, _f)\n",
    "print('Đã lưu split_indices.json!')\n",
], "code_split_claude")

# ──────────────────────────────────────────────────────────────────────────────
# SỬA 3: Cell grid search alpha — chèn SAU split, TRƯỚC metric
# ──────────────────────────────────────────────────────────────────────────────

CELL_GRID_MD = make_md(
    "## GRID SEARCH ALPHA TRÊN VALIDATION SET\n\n"
    "🤖 **AI ghi chú**: Tìm trọng số alpha tối ưu cho Weighted Fusion:\n"
    "`sim = alpha × img_sim + (1 - alpha) × txt_sim`\n\n"
    "Grid search trên **val set (20%)** → áp dụng `best_alpha` lên **test set (80%)** để tránh data leakage.",
    "md_grid_claude"
)

CELL_GRID_CODE = make_code([
    "# ============================================================\n",
    "# GRID SEARCH ALPHA TRÊN VALIDATION SET\n",
    "# ============================================================\n",
    "print('=' * 55)\n",
    "print('GRID SEARCH ALPHA (VAL SET)')\n",
    "print('=' * 55)\n",
    "\n",
    "alphas = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]\n",
    "best_alpha    = 0.7\n",
    "best_val_map  = -1.0\n",
    "\n",
    "image_tensor_val = torch.tensor(\n",
    "    dinov3_image_features[val_idx]).to(device)\n",
    "text_tensor_val  = torch.tensor(\n",
    "    minilm_text_features[val_idx]).to(device)\n",
    "image_norm_val   = image_tensor_val / image_tensor_val.norm(dim=-1, keepdim=True)\n",
    "text_norm_val    = text_tensor_val  / text_tensor_val.norm(dim=-1, keepdim=True)\n",
    "\n",
    "image_norm_all = torch.tensor(dinov3_image_features).to(device)\n",
    "text_norm_all  = torch.tensor(minilm_text_features).to(device)\n",
    "image_norm_all = image_norm_all / image_norm_all.norm(dim=-1, keepdim=True)\n",
    "text_norm_all  = text_norm_all  / text_norm_all.norm(dim=-1, keepdim=True)\n",
    "\n",
    "phash_strings = candidate_df[\"image_phash\"].values\n",
    "phash_ints    = np.array([int(h, 16) for h in phash_strings], dtype=np.uint64)\n",
    "labels_all    = candidate_df[\"label_group\"].values\n",
    "\n",
    "for alpha in alphas:\n",
    "    ap_scores = []\n",
    "    for q_pos, i in enumerate(val_idx):\n",
    "        query_label = labels_all[i]\n",
    "        gt_indices  = np.where(labels_all == query_label)[0]\n",
    "        gt_indices  = gt_indices[gt_indices != i]\n",
    "        if len(gt_indices) == 0:\n",
    "            continue   # ← Fix: bỏ qua items không có ground truth\n",
    "\n",
    "        # Tính fusion score\n",
    "        img_sim = torch.matmul(\n",
    "            image_norm_val[q_pos].unsqueeze(0),\n",
    "            image_norm_all.T).squeeze().cpu().numpy()\n",
    "        txt_sim = torch.matmul(\n",
    "            text_norm_val[q_pos].unsqueeze(0),\n",
    "            text_norm_all.T).squeeze().cpu().numpy()\n",
    "        sim = alpha * img_sim + (1 - alpha) * txt_sim\n",
    "\n",
    "        # pHash boost (giữ nguyên thuật toán bit-twiddling gốc)\n",
    "        q_phash = phash_ints[i]\n",
    "        x = q_phash ^ phash_ints\n",
    "        x = (x & 0x5555555555555555) + ((x >> 1) & 0x5555555555555555)\n",
    "        x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)\n",
    "        x = (x & 0x0F0F0F0F0F0F0F0F) + ((x >> 4) & 0x0F0F0F0F0F0F0F0F)\n",
    "        x = (x & 0x00FF00FF00FF00FF) + ((x >> 8) & 0x00FF00FF00FF00FF)\n",
    "        x = (x & 0x0000FFFF0000FFFF) + ((x >> 16) & 0x0000FFFF0000FFFF)\n",
    "        ham = ((x & 0x00000000FFFFFFFF) + (x >> 32)).astype(np.uint8)\n",
    "        sim[ham <= 2] += 0.5\n",
    "        sim[ham == 0] += 0.5\n",
    "        sim[i]         = -999  # loại self\n",
    "\n",
    "        top5 = np.argsort(-sim)[:5]\n",
    "        is_rel = np.isin(top5, gt_indices)\n",
    "        hits = np.where(is_rel)[0] + 1\n",
    "        if len(hits) > 0:\n",
    "            ap = np.sum(np.arange(1, len(hits)+1) / hits) / min(5, len(gt_indices))\n",
    "        else:\n",
    "            ap = 0.0\n",
    "        ap_scores.append(ap)\n",
    "\n",
    "    val_map = round(float(np.mean(ap_scores)), 4)\n",
    "    print(f'  alpha={alpha:.2f} → val mAP@5 = {val_map:.4f}')\n",
    "\n",
    "    if val_map > best_val_map:\n",
    "        best_val_map  = val_map\n",
    "        best_alpha    = alpha\n",
    "\n",
    "print(f'\\n🏆 BEST alpha = {best_alpha}, val mAP@5 = {best_val_map:.4f}')\n",
    "print(f'Dùng alpha = {best_alpha} để đánh giá trên Test Set')\n",
], "code_grid_claude")

# ──────────────────────────────────────────────────────────────────────────────
# SỬA 4: Cell metric mới — đúng theo CLAUDE.md
# ──────────────────────────────────────────────────────────────────────────────

CELL_METRIC_MD = make_md(
    "## TÍNH METRIC TRÊN TEST SET (80%)\n\n"
    "🤖 **AI ghi chú**: Dùng `best_alpha` từ grid search để tính metric trên **test set**.\n\n"
    "**Fix quan trọng**: `gt_len == 0` → `continue` (không `append(0)` kéo mAP xuống giả tạo).",
    "md_metric_claude"
)

CELL_METRIC_CODE = make_code([
    "# ============================================================\n",
    "# TÍNH METRIC TRÊN TEST SET (80%)\n",
    "# Dùng best_alpha từ grid search ở trên\n",
    "# ============================================================\n",
    "import os\n",
    "import gc\n",
    "import json as _json\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import torch\n",
    "from tqdm import tqdm\n",
    "\n",
    "csv_path   = \"/content/drive/MyDrive/DoAnPython/DuLieuPython/train.csv\"\n",
    "output_dir = \"/content/drive/MyDrive/DoAnPython/DuLieuPython\"\n",
    "\n",
    "candidate_df = pd.read_csv(csv_path)\n",
    "labels       = candidate_df[\"label_group\"].values\n",
    "\n",
    "dinov3_image_features = np.load(os.path.join(output_dir, \"dinov3_image_features.npy\")).astype('float32')\n",
    "minilm_text_features  = np.load(os.path.join(output_dir, \"multilingual_minilm_text_features.npy\")).astype('float32')\n",
    "phash_strings         = candidate_df[\"image_phash\"].values\n",
    "phash_ints            = np.array([int(h, 16) for h in phash_strings], dtype=np.uint64)\n",
    "\n",
    "# Load test indices\n",
    "try:\n",
    "    with open(os.path.join(output_dir, 'split_indices.json'), 'r') as _f:\n",
    "        split_info = _json.load(_f)\n",
    "    test_idx = split_info['test_idx']\n",
    "except FileNotFoundError:\n",
    "    # Nếu chưa chạy cell split, tạo lại\n",
    "    from sklearn.model_selection import train_test_split\n",
    "    label_counts_all = pd.Series(labels).value_counts()\n",
    "    valid_mask  = pd.Series(labels).isin(label_counts_all[label_counts_all >= 2].index)\n",
    "    valid_indices = np.where(valid_mask)[0].tolist()\n",
    "    _, test_idx = train_test_split(valid_indices, test_size=0.8, random_state=42)\n",
    "    print('Đã tạo lại test_idx (chạy cell split trước để chính xác hơn)')\n",
    "\n",
    "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
    "\n",
    "image_norm = torch.tensor(dinov3_image_features).to(device)\n",
    "text_norm  = torch.tensor(minilm_text_features).to(device)\n",
    "image_norm = image_norm / image_norm.norm(dim=-1, keepdim=True)\n",
    "text_norm  = text_norm  / text_norm.norm(dim=-1, keepdim=True)\n",
    "\n",
    "# Dùng best_alpha từ grid search — nếu chưa chạy grid search thì dùng 0.7\n",
    "ALPHA = best_alpha if 'best_alpha' in dir() else 0.7\n",
    "print(f'Đánh giá test set với alpha = {ALPHA}')\n",
    "print(f'Số query test: {len(test_idx):,}')\n",
    "print('=' * 55)\n",
    "\n",
    "ap5_scores = []\n",
    "p1, r1 = [], []\n",
    "p3, r3 = [], []\n",
    "p5, r5 = [], []\n",
    "p10,r10 = [], []\n",
    "\n",
    "for i in tqdm(test_idx, desc='Tính metric (Test Set)'):\n",
    "    query_label = labels[i]\n",
    "    gt_indices  = np.where(labels == query_label)[0]\n",
    "    gt_indices  = gt_indices[gt_indices != i]  # loại chính nó\n",
    "    gt_len      = len(gt_indices)\n",
    "\n",
    "    # ← QUAN TRỌNG: bỏ qua query không có ảnh liên quan\n",
    "    # KHÔNG dùng append(0) vì sẽ kéo mAP xuống giả tạo\n",
    "    if gt_len == 0:\n",
    "        continue\n",
    "\n",
    "    # Tính fusion score\n",
    "    img_sim = torch.matmul(\n",
    "        image_norm[i].unsqueeze(0), image_norm.T\n",
    "    ).squeeze().cpu().numpy()\n",
    "    txt_sim = torch.matmul(\n",
    "        text_norm[i].unsqueeze(0), text_norm.T\n",
    "    ).squeeze().cpu().numpy()\n",
    "    sim = ALPHA * img_sim + (1 - ALPHA) * txt_sim\n",
    "\n",
    "    # pHash boost (giữ nguyên thuật toán bit-twiddling của Hưng)\n",
    "    q_phash   = phash_ints[i]\n",
    "    x         = q_phash ^ phash_ints\n",
    "    x = (x & 0x5555555555555555) + ((x >> 1) & 0x5555555555555555)\n",
    "    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)\n",
    "    x = (x & 0x0F0F0F0F0F0F0F0F) + ((x >> 4) & 0x0F0F0F0F0F0F0F0F)\n",
    "    x = (x & 0x00FF00FF00FF00FF) + ((x >> 8) & 0x00FF00FF00FF00FF)\n",
    "    x = (x & 0x0000FFFF0000FFFF) + ((x >> 16) & 0x0000FFFF0000FFFF)\n",
    "    ham       = ((x & 0x00000000FFFFFFFF) + (x >> 32)).astype(np.uint8)\n",
    "    sim[ham <= 2] += 0.5\n",
    "    sim[ham == 0] += 0.5\n",
    "    sim[i]         = -999  # loại self\n",
    "\n",
    "    # Lấy top-10\n",
    "    top10      = np.argsort(-sim)[:10]\n",
    "    is_rel     = np.isin(top10, gt_indices)\n",
    "\n",
    "    h1  = is_rel[:1].sum();  p1.append(h1/1);   r1.append(h1/gt_len)\n",
    "    h3  = is_rel[:3].sum();  p3.append(h3/3);   r3.append(h3/gt_len)\n",
    "    h5  = is_rel[:5].sum();  p5.append(h5/5);   r5.append(h5/gt_len)\n",
    "    h10 = is_rel[:10].sum(); p10.append(h10/10); r10.append(h10/gt_len)\n",
    "\n",
    "    # AP@5\n",
    "    hits = np.where(is_rel[:5])[0] + 1\n",
    "    if len(hits) > 0:\n",
    "        ap = np.sum(np.arange(1, len(hits)+1) / hits) / min(5, gt_len)\n",
    "    else:\n",
    "        ap = 0.0\n",
    "    ap5_scores.append(ap)\n",
    "\n",
    "# Tổng hợp kết quả\n",
    "summary_metrics = pd.DataFrame({\n",
    "    \"K\"        : [1, 3, 5, 10],\n",
    "    \"Precision\": [np.mean(p1), np.mean(p3), np.mean(p5), np.mean(p10)],\n",
    "    \"Recall\"   : [np.mean(r1), np.mean(r3), np.mean(r5), np.mean(r10)],\n",
    "})\n",
    "summary_metrics[[\"Precision\", \"Recall\"]] = summary_metrics[[\"Precision\", \"Recall\"]].round(4)\n",
    "map_at_5 = round(float(np.mean(ap5_scores)), 4)\n",
    "\n",
    "print('\\n=== KẾT QUẢ CUỐI — DINOv3 + Multilingual MiniLM + pHash ===')\n",
    "print(f'Tập đánh giá: Test set ({len(test_idx):,} query, 80%)')\n",
    "print(f'Alpha        : {ALPHA}')\n",
    "print(summary_metrics.to_string(index=False))\n",
    "print(f'mAP@5        : {map_at_5}')\n",
    "\n",
    "# Lưu metric ra CSV để file chung (BTCT_Tuan4_33.ipynb) đọc\n",
    "summary_metrics['mAP@5']  = map_at_5\n",
    "summary_metrics['method'] = f'DINOv3 + MultilingualMiniLM + pHash (alpha={ALPHA})'\n",
    "summary_metrics.to_csv(\n",
    "    os.path.join(output_dir, 'metrics_dinov3_minilm.csv'), index=False)\n",
    "print('\\nĐã lưu metrics_dinov3_minilm.csv!')\n",
], "code_metric_claude")

# ──────────────────────────────────────────────────────────────────────────────
# SỬA 5: 2 cell markdown cuối
# ──────────────────────────────────────────────────────────────────────────────

CELL_AI_NOTE = make_md(
    "## GHI CHÚ AI HỖ TRỢ\n\n"
    "_(Bắt buộc theo yêu cầu thầy)_\n\n"
    "| Phần | AI hỗ trợ như thế nào | Người kiểm tra |\n"
    "| ---- | --------------------- | -------------- |\n"
    "| Load DINOv3 (timm) | Cursor gợi ý dùng `resolve_model_data_config` lấy transform chuẩn | Nguyễn Khánh Hưng |\n"
    "| Bit-twiddling Hamming | Cursor gợi ý thuật toán tối ưu không cần vòng lặp | Nguyễn Khánh Hưng |\n"
    "| Grid search alpha | Claude gợi ý quy trình val/test split | Nguyễn Khánh Hưng |\n"
    "| Sửa gt_len=0 | Claude phát hiện lỗi append(0) → đổi thành continue | Nguyễn Khánh Hưng |\n"
    "| Đổi multilingual MiniLM | Claude gợi ý model phù hợp Shopee đa ngôn ngữ | Nguyễn Khánh Hưng |",
    "md_ai_note_claude"
)

CELL_PLAN = make_md(
    "## KẾ HOẠCH TUẦN 5\n\n"
    "| Nội dung | Phương pháp | Mục tiêu |\n"
    "| -------- | ----------- | -------- |\n"
    "| Fine-tune DINOv3 trên Shopee | Triplet Loss với hard negative mining | Tăng mAP so với zero-shot |\n"
    "| Thử EfficientNet-B4 thay DINOv3 | Nhẹ hơn, dễ fine-tune hơn | So sánh với DINOv3 |\n"
    "| Tối ưu pHash threshold | Grid search threshold trên val set | Tìm threshold tốt nhất |\n"
    "| Kết hợp DINOv3 + TF-IDF | Thay MiniLM bằng TF-IDF (đã chứng minh tốt T3) | Có thể cao hơn MiniLM |",
    "md_plan_claude"
)

# ──────────────────────────────────────────────────────────────────────────────
# Chèn tất cả vào đúng vị trí
# ──────────────────────────────────────────────────────────────────────────────

# Tìm lại vị trí cell metric (sau khi đã clean)
idx_m = find_by_content(nb["cells"], "for i in tqdm(range(num_samples))")
if idx_m is None:
    idx_m = find_by_id(nb["cells"], "476c1f06")

print(f"Vị trí cell metric (để replace): {idx_m}")

# Tìm markdown "TÍNH TOÁN METRIC" trước nó để xóa/thay
if idx_m and idx_m > 0:
    prev_cell = nb["cells"][idx_m - 1]
    if prev_cell.get("cell_type") == "markdown":
        # Thay bằng markdown của SỬA 4 (gộp luôn)
        nb["cells"][idx_m - 1] = CELL_METRIC_MD
        nb["cells"][idx_m]     = CELL_METRIC_CODE
        insert_at = idx_m - 1   # vị trí để chèn thêm vào trước đây
    else:
        # Không có markdown trước → chèn mới
        nb["cells"][idx_m] = CELL_METRIC_CODE
        nb["cells"].insert(idx_m, CELL_METRIC_MD)
        insert_at = idx_m
elif idx_m is None:
    # Không tìm thấy metric cell cũ → append vào cuối (trước 2 markdown cuối)
    print("Không tìm thấy cell metric gốc → append vào cuối")
    nb["cells"].extend([CELL_METRIC_MD, CELL_METRIC_CODE])
    insert_at = len(nb["cells"]) - 2

# insert_at = vị trí của CELL_METRIC_MD
# Chèn split + grid TRƯỚC metric
nb["cells"].insert(insert_at, CELL_GRID_CODE)
nb["cells"].insert(insert_at, CELL_GRID_MD)
nb["cells"].insert(insert_at, CELL_SPLIT_CODE)
nb["cells"].insert(insert_at, CELL_SPLIT_MD)

# Thêm 2 markdown vào cuối (SỬA 5)
nb["cells"].append(CELL_AI_NOTE)
nb["cells"].append(CELL_PLAN)

print(f"Tổng số cell sau patch: {len(nb['cells'])}")

# ──────────────────────────────────────────────────────────────────────────────
# Ghi ra file
# ──────────────────────────────────────────────────────────────────────────────
with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\n✅ Patch FINAL hoàn tất theo CLAUDE.md: {NB}")
print("\nTóm tắt 5 sửa đổi:")
print("  [SỬA 1] all-MiniLM-L6-v2 → paraphrase-multilingual-MiniLM-L12-v2")
print("          file: minilm_text_features.npy → multilingual_minilm_text_features.npy")
print("  [SỬA 2] Thêm cell chia val(20%)/test(80%) split")
print("  [SỬA 3] Thêm cell grid search alpha [0.5→0.9] trên val set")
print("  [SỬA 4] Thay cell metric: tính trên test set + gt_len=0→continue + lưu CSV")
print("  [SỬA 5] Thêm 2 markdown: Ghi chú AI + Kế hoạch tuần 5")
