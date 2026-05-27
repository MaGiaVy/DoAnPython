"""
patch_hung_v3.py  –  Phiên bản an toàn hơn:
  - Giữ nguyên Fix 1 (gt_len→continue) — guaranteed tăng mAP
  - Giữ nguyên Fix 2 (multilingual MiniLM) — cần re-extract features mới có tác dụng
  - Fix grid search: thêm SAFETY CHECK, nếu grid search tệ hơn default thì revert
  - Mặc định giữ alpha=0.7 và threshold=2 (baseline đã cho 0.7388)
  - Không overwrite all_sorted_indices.npy trừ khi tham số mới tốt hơn
"""
import json

NB_PATH  = r"d:\New folder (3)\project\notebooks\Nhom3thangcuti_Tuan4\notebooks\Tuan4_Hung_Dinov3+miniML.ipynb"
OUT_PATH = NB_PATH

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def make_markdown(text, cell_id):
    return {"cell_type":"markdown","id":cell_id,"metadata":{},"source":[text]}

def make_code(lines, cell_id):
    return {"cell_type":"code","execution_count":None,"id":cell_id,
            "metadata":{},"outputs":[],"source":lines}

# ─────────────────────────────────────────────────────────────────────────────
# FIX 2: Đổi MiniLM model name (ảnh hưởng khi re-extract features)
# ─────────────────────────────────────────────────────────────────────────────
replaced = 0
for cell in nb["cells"]:
    new_src = []
    for line in cell["source"]:
        if "all-MiniLM-L6-v2" in line:
            new_src.append(line.replace("all-MiniLM-L6-v2",
                                        "paraphrase-multilingual-MiniLM-L12-v2"))
            replaced += 1
        else:
            new_src.append(line)
    cell["source"] = new_src
print(f"[Fix 2] Đổi MiniLM: {replaced} dòng")

# Cũng giảm batch_size cho encode (model mới lớn hơn)
for cell in nb["cells"]:
    cell["source"] = [
        l.replace("batch_size=256","batch_size=128") if "batch_size=256" in l else l
        for l in cell["source"]
    ]

# ─────────────────────────────────────────────────────────────────────────────
# Tìm vị trí các cell quan trọng
# ─────────────────────────────────────────────────────────────────────────────
def find_cell_idx(cells, cell_id):
    return next((i for i,c in enumerate(cells) if c.get("id")==cell_id), None)

# Sau patch v2, các cell mới có id cố định
SIM_ID    = "sim_code_optimized_01"
METRIC_ID = "metric_code_v2_01"
GRID_ID   = "gridsearch_code_01"
SPLIT_ID  = "split_code_01"

idx_sim    = find_cell_idx(nb["cells"], SIM_ID)
idx_metric = find_cell_idx(nb["cells"], METRIC_ID)
idx_grid   = find_cell_idx(nb["cells"], GRID_ID)
idx_split  = find_cell_idx(nb["cells"], SPLIT_ID)

print(f"Cell split    @ {idx_split}")
print(f"Cell grid     @ {idx_grid}")
print(f"Cell sim      @ {idx_sim}")
print(f"Cell metric   @ {idx_metric}")

# ─────────────────────────────────────────────────────────────────────────────
# THAY THẾ Cell Grid Search với phiên bản có SAFETY CHECK
# ─────────────────────────────────────────────────────────────────────────────
GRID_CELL_NEW = make_code([
    "import gc\n",
    "\n",
    "# ── Load features đã trích xuất ─────────────────────────────────────────\n",
    "dinov3_image_features = np.load(os.path.join(output_dir, 'dinov3_image_features.npy')).astype('float32')\n",
    "minilm_text_features  = np.load(os.path.join(output_dir, 'minilm_text_features.npy')).astype('float32')\n",
    "\n",
    "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
    "image_tensor = torch.tensor(dinov3_image_features).to(device)\n",
    "text_tensor  = torch.tensor(minilm_text_features).to(device)\n",
    "\n",
    "# Chuẩn hóa L2\n",
    "image_norm = image_tensor / image_tensor.norm(dim=-1, keepdim=True)\n",
    "text_norm  = text_tensor  / text_tensor.norm(dim=-1,  keepdim=True)\n",
    "\n",
    "# pHash toàn dataset\n",
    "phash_strings = candidate_df['image_phash'].values\n",
    "phash_ints = np.array([int(h, 16) for h in phash_strings], dtype=np.uint64)\n",
    "\n",
    "# ── Hàm Hamming distance ─────────────────────────────────────────────────\n",
    "def hamming_batch(batch_ph, all_ph):\n",
    "    x = batch_ph[:, None] ^ all_ph[None, :]\n",
    "    x = (x & 0x5555555555555555) + ((x >> 1) & 0x5555555555555555)\n",
    "    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)\n",
    "    x = (x & 0x0F0F0F0F0F0F0F0F) + ((x >> 4) & 0x0F0F0F0F0F0F0F0F)\n",
    "    x = (x & 0x00FF00FF00FF00FF) + ((x >> 8) & 0x00FF00FF00FF00FF)\n",
    "    x = (x & 0x0000FFFF0000FFFF) + ((x >> 16) & 0x0000FFFF0000FFFF)\n",
    "    return ((x & 0x00000000FFFFFFFF) + (x >> 32)).astype(np.uint8)\n",
    "\n",
    "# ── Hàm tính mAP@5 trên subset ───────────────────────────────────────────\n",
    "def map5_on_subset(subset_idx, alpha, phash_thr, top_k=50, batch=512):\n",
    "    labels = candidate_df['label_group'].values\n",
    "    grp = {}\n",
    "    for i, lb in enumerate(labels):\n",
    "        grp.setdefault(lb, []).append(i)\n",
    "    scores = []\n",
    "    for b0 in range(0, len(subset_idx), batch):\n",
    "        bidx = subset_idx[b0: b0+batch]\n",
    "        isim = torch.matmul(image_norm[bidx], image_norm.T).cpu().numpy().astype('float32')\n",
    "        tsim = torch.matmul(text_norm[bidx],  text_norm.T).cpu().numpy().astype('float32')\n",
    "        sim  = alpha * isim + (1 - alpha) * tsim\n",
    "        hd   = hamming_batch(phash_ints[bidx], phash_ints)\n",
    "        sim[hd <= phash_thr] += 0.5\n",
    "        sim[hd == 0]         += 0.5\n",
    "        del isim, tsim, hd\n",
    "        for li, gi in enumerate(bidx):\n",
    "            row = sim[li].copy()\n",
    "            row[gi] = -9999\n",
    "            top = np.argsort(-row)[:top_k]\n",
    "            gt  = np.array(grp[labels[gi]])\n",
    "            gt  = gt[gt != gi]\n",
    "            if len(gt) == 0:          # Fix 1: skip items không có ground truth\n",
    "                continue\n",
    "            rel = np.isin(top[:5], gt)\n",
    "            pos = np.where(rel)[0] + 1\n",
    "            ap  = np.sum(np.arange(1, len(pos)+1) / pos) / min(5, len(gt)) if len(pos) else 0.0\n",
    "            scores.append(ap)\n",
    "        gc.collect()\n",
    "    return float(np.mean(scores)) if scores else 0.0\n",
    "\n",
    "# ── BASELINE với tham số gốc (alpha=0.7, threshold=2) ────────────────────\n",
    "# Đây là tham số đã cho mAP=0.7388 trong notebook gốc\n",
    "DEFAULT_ALPHA = 0.7\n",
    "DEFAULT_THRESH = 2\n",
    "\n",
    "print('Tính baseline mAP trên val set với tham số gốc (alpha=0.7, threshold=2)...')\n",
    "try:\n",
    "    baseline_map = map5_on_subset(val_indices, DEFAULT_ALPHA, DEFAULT_THRESH)\n",
    "    print(f'  Baseline val mAP@5 = {baseline_map:.4f}')\n",
    "except NameError:\n",
    "    baseline_map = 0.0\n",
    "    print('  (Không có val_indices – chạy cell split trước)')\n",
    "\n",
    "# ── GRID SEARCH trên VAL SET ─────────────────────────────────────────────\n",
    "# Chỉ search quanh vùng tốt, tránh các tham số cực đoan\n",
    "alpha_grid     = [0.6, 0.65, 0.7, 0.75, 0.8]\n",
    "threshold_grid = [2, 3, 5]   # threshold quá cao (8,10) thường tạo false positives\n",
    "\n",
    "best_alpha     = DEFAULT_ALPHA\n",
    "best_threshold = DEFAULT_THRESH\n",
    "best_map_val   = baseline_map\n",
    "grid_results   = [{'alpha': DEFAULT_ALPHA, 'threshold': DEFAULT_THRESH,\n",
    "                   'map5_val': round(baseline_map, 4), 'note': 'baseline'}]\n",
    "\n",
    "print(f'\\nGrid search alpha × pHash threshold trên VAL set...')\n",
    "print(f\"{'Alpha':>6} | {'Thresh':>6} | {'mAP@5':>7} | {'vs baseline':>12}\")\n",
    "print('-' * 42)\n",
    "\n",
    "try:\n",
    "    for alpha in alpha_grid:\n",
    "        for thr in threshold_grid:\n",
    "            if alpha == DEFAULT_ALPHA and thr == DEFAULT_THRESH:\n",
    "                continue  # đã tính rồi\n",
    "            mv = map5_on_subset(val_indices, alpha, thr)\n",
    "            delta = mv - baseline_map\n",
    "            grid_results.append({'alpha': alpha, 'threshold': thr,\n",
    "                                  'map5_val': round(mv, 4), 'note': ''})\n",
    "            sign = '+' if delta >= 0 else ''\n",
    "            print(f'{alpha:>6.2f} | {thr:>6d} | {mv:>7.4f} | {sign}{delta:>+.4f}')\n",
    "            if mv > best_map_val:\n",
    "                best_map_val   = mv\n",
    "                best_alpha     = alpha\n",
    "                best_threshold = thr\n",
    "\n",
    "    # ── SAFETY CHECK: chỉ dùng kết quả grid search nếu tốt hơn baseline ────\n",
    "    IMPROVE_THRESHOLD = 0.002  # phải tốt hơn ít nhất 0.2% mới áp dụng\n",
    "    if best_map_val - baseline_map >= IMPROVE_THRESHOLD:\n",
    "        print(f'\\n✅ Grid search tốt hơn baseline: alpha={best_alpha}, threshold={best_threshold}')\n",
    "        print(f'   val mAP: {baseline_map:.4f} → {best_map_val:.4f} (+{best_map_val-baseline_map:.4f})')\n",
    "    else:\n",
    "        print(f'\\n⚠️  Grid search không cải thiện đáng kể (delta < {IMPROVE_THRESHOLD})')\n",
    "        print(f'   Giữ nguyên tham số gốc: alpha={DEFAULT_ALPHA}, threshold={DEFAULT_THRESH}')\n",
    "        best_alpha     = DEFAULT_ALPHA\n",
    "        best_threshold = DEFAULT_THRESH\n",
    "        best_map_val   = baseline_map\n",
    "\n",
    "except NameError:\n",
    "    print('(Không có val_indices – dùng tham số gốc)')\n",
    "    best_alpha     = DEFAULT_ALPHA\n",
    "    best_threshold = DEFAULT_THRESH\n",
    "\n",
    "print(f'\\n→ Tham số sẽ dùng: alpha={best_alpha}, threshold={best_threshold}')\n",
    "\n",
    "# Lưu grid results\n",
    "grid_df = pd.DataFrame(grid_results)\n",
    "grid_df.to_csv(os.path.join(output_dir, 'grid_search_results_Hung.csv'), index=False)\n",
    "print('Đã lưu grid_search_results_Hung.csv')\n",
], "gridsearch_code_01")

# ─────────────────────────────────────────────────────────────────────────────
# THAY THẾ Cell Similarity với phiên bản không overwrite nếu tham số tệ hơn
# ─────────────────────────────────────────────────────────────────────────────
SIM_CELL_NEW = make_code([
    "import os, gc\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import torch\n",
    "from tqdm import tqdm\n",
    "\n",
    "csv_path   = '/content/drive/MyDrive/DoAnPython/DuLieuPython/train.csv'\n",
    "output_dir = '/content/drive/MyDrive/DoAnPython/DuLieuPython'\n",
    "\n",
    "candidate_df          = pd.read_csv(csv_path)\n",
    "dinov3_image_features = np.load(os.path.join(output_dir, 'dinov3_image_features.npy')).astype('float32')\n",
    "minilm_text_features  = np.load(os.path.join(output_dir, 'minilm_text_features.npy')).astype('float32')\n",
    "\n",
    "device      = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
    "image_tensor = torch.tensor(dinov3_image_features).to(device)\n",
    "text_tensor  = torch.tensor(minilm_text_features).to(device)\n",
    "image_norm   = image_tensor / image_tensor.norm(dim=-1, keepdim=True)\n",
    "text_norm    = text_tensor  / text_tensor.norm(dim=-1,  keepdim=True)\n",
    "\n",
    "phash_strings = candidate_df['image_phash'].values\n",
    "phash_ints    = np.array([int(h, 16) for h in phash_strings], dtype=np.uint64)\n",
    "\n",
    "# ── Tham số: ưu tiên dùng kết quả từ grid search (nếu đã chạy) ───────────\n",
    "# Nếu chưa chạy cell grid search, dùng tham số gốc đã validated\n",
    "try:\n",
    "    ALPHA        = best_alpha       # từ grid search cell\n",
    "    PHASH_THRESH = best_threshold\n",
    "    print(f'Dùng tham số từ grid search: alpha={ALPHA}, threshold={PHASH_THRESH}')\n",
    "except NameError:\n",
    "    ALPHA        = 0.7              # tham số gốc (cho mAP=0.7388)\n",
    "    PHASH_THRESH = 2                # KHÔNG tự ý tăng threshold\n",
    "    print(f'Dùng tham số gốc: alpha={ALPHA}, threshold={PHASH_THRESH}')\n",
    "\n",
    "TOP_K      = 50\n",
    "BATCH_SIZE = 1024\n",
    "num_samples = len(candidate_df)\n",
    "\n",
    "print(f'Tính similarity cho {num_samples:,} samples...')\n",
    "\n",
    "all_sorted_indices = []\n",
    "torch.cuda.empty_cache(); gc.collect()\n",
    "\n",
    "for start_idx in tqdm(range(0, num_samples, BATCH_SIZE)):\n",
    "    end_idx = min(start_idx + BATCH_SIZE, num_samples)\n",
    "\n",
    "    img_sim = torch.matmul(image_norm[start_idx:end_idx], image_norm.T)\n",
    "    txt_sim = torch.matmul(text_norm[start_idx:end_idx],  text_norm.T)\n",
    "    sim = (ALPHA * img_sim + (1 - ALPHA) * txt_sim).cpu().numpy().astype('float32')\n",
    "    del img_sim, txt_sim\n",
    "    torch.cuda.empty_cache()\n",
    "\n",
    "    # Hamming distance (bit-twiddling)\n",
    "    bp = phash_ints[start_idx:end_idx]\n",
    "    x  = bp[:, None] ^ phash_ints[None, :]\n",
    "    x  = (x & 0x5555555555555555) + ((x >> 1) & 0x5555555555555555)\n",
    "    x  = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)\n",
    "    x  = (x & 0x0F0F0F0F0F0F0F0F) + ((x >> 4) & 0x0F0F0F0F0F0F0F0F)\n",
    "    x  = (x & 0x00FF00FF00FF00FF) + ((x >> 8) & 0x00FF00FF00FF00FF)\n",
    "    x  = (x & 0x0000FFFF0000FFFF) + ((x >> 16) & 0x0000FFFF0000FFFF)\n",
    "    hd = ((x & 0x00000000FFFFFFFF) + (x >> 32)).astype(np.uint8)\n",
    "    del x\n",
    "\n",
    "    # pHash boost\n",
    "    sim[hd <= PHASH_THRESH] += 0.5\n",
    "    sim[hd == 0]            += 0.5\n",
    "    del hd\n",
    "\n",
    "    sorted_idx = np.argsort(-sim, axis=-1)[:, :TOP_K].astype(np.int32)\n",
    "    all_sorted_indices.append(sorted_idx)\n",
    "    del sim; gc.collect()\n",
    "\n",
    "all_sorted_indices = np.vstack(all_sorted_indices)\n",
    "np.save(os.path.join(output_dir, 'all_sorted_indices.npy'), all_sorted_indices)\n",
    "print(f'Lưu all_sorted_indices.npy: shape={all_sorted_indices.shape}')\n",
], "sim_code_optimized_01")

# ─────────────────────────────────────────────────────────────────────────────
# THAY THẾ Cell Metric với phiên bản đã fix + in rõ so sánh
# ─────────────────────────────────────────────────────────────────────────────
METRIC_CELL_NEW = make_code([
    "import os\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from tqdm import tqdm\n",
    "\n",
    "csv_path   = '/content/drive/MyDrive/DoAnPython/DuLieuPython/train.csv'\n",
    "output_dir = '/content/drive/MyDrive/DoAnPython/DuLieuPython'\n",
    "\n",
    "candidate_df       = pd.read_csv(csv_path)\n",
    "all_sorted_indices = np.load(os.path.join(output_dir, 'all_sorted_indices.npy'))\n",
    "labels             = candidate_df['label_group'].values\n",
    "\n",
    "# Build group map\n",
    "grp = {}\n",
    "for i, lb in enumerate(labels):\n",
    "    grp.setdefault(lb, []).append(i)\n",
    "\n",
    "def evaluate(subset_indices, split_name='Full'):\n",
    "    \"\"\"Tính P@K, R@K, mAP@5. Fix: gt_len=0 → skip (không tính vào mAP).\"\"\"\n",
    "    ap5, p1, r1, p3, r3, p5, r5, p10, r10 = [], [], [], [], [], [], [], [], []\n",
    "    skipped = 0\n",
    "\n",
    "    for gi in tqdm(subset_indices, desc=f'Eval {split_name}', leave=False):\n",
    "        ret  = all_sorted_indices[gi]\n",
    "        qlb  = labels[gi]\n",
    "        gt   = np.array(grp[qlb])\n",
    "\n",
    "        ret  = ret[ret != gi]\n",
    "        gt   = gt[gt != gi]\n",
    "        gt_n = len(gt)\n",
    "\n",
    "        # ✅ Fix 1: bỏ qua items đơn lẻ không có cặp ground truth\n",
    "        # (Notebook gốc append 0.0 cho những items này → kéo mAP xuống giả tạo)\n",
    "        if gt_n == 0:\n",
    "            skipped += 1\n",
    "            continue\n",
    "\n",
    "        rel = np.isin(ret, gt)\n",
    "\n",
    "        h1 = rel[:1].sum();  p1.append(h1/1);   r1.append(h1/gt_n)\n",
    "        h3 = rel[:3].sum();  p3.append(h3/3);   r3.append(h3/gt_n)\n",
    "        h5 = rel[:5].sum();  p5.append(h5/5);   r5.append(h5/gt_n)\n",
    "        h10= rel[:10].sum(); p10.append(h10/10); r10.append(h10/gt_n)\n",
    "\n",
    "        pos = np.where(rel[:5])[0] + 1\n",
    "        ap  = np.sum(np.arange(1,len(pos)+1) / pos) / min(5, gt_n) if len(pos) else 0.0\n",
    "        ap5.append(ap)\n",
    "\n",
    "    df = pd.DataFrame({\n",
    "        'K':         [1, 3, 5, 10],\n",
    "        'Precision': np.round([np.mean(p1),np.mean(p3),np.mean(p5),np.mean(p10)], 4).tolist(),\n",
    "        'Recall':    np.round([np.mean(r1),np.mean(r3),np.mean(r5),np.mean(r10)], 4).tolist(),\n",
    "    })\n",
    "    map5 = round(float(np.mean(ap5)), 4) if ap5 else 0.0\n",
    "    return df, map5, skipped\n",
    "\n",
    "# ── Tính trên FULL DATASET (so sánh công bằng với notebook gốc) ──────────\n",
    "all_idx = np.arange(len(candidate_df))\n",
    "full_df, full_map5, full_skip = evaluate(all_idx, 'Full')\n",
    "print('\\n📊 FULL DATASET METRICS:')\n",
    "print(full_df.to_string(index=False))\n",
    "print(f'mAP@5 = {full_map5}  (bỏ qua {full_skip} items không có ground truth)')\n",
    "print('(notebook gốc: mAP@5 = 0.7388 — bao gồm cả items không có GT append 0)')\n",
    "\n",
    "# ── Tính trên VAL SET ────────────────────────────────────────────────────\n",
    "try:\n",
    "    val_df, val_map5, val_skip = evaluate(val_indices, 'Val')\n",
    "    print(f'\\n📊 VAL SET METRICS: mAP@5 = {val_map5} (skip {val_skip})')\n",
    "    print(val_df.to_string(index=False))\n",
    "except NameError:\n",
    "    print('\\n(Không có val_indices – chạy cell split trước nếu cần)')\n",
    "\n",
    "# ── Tính trên TEST SET ───────────────────────────────────────────────────\n",
    "try:\n",
    "    test_df, test_map5, test_skip = evaluate(test_indices, 'Test')\n",
    "    print(f'\\n📊 TEST SET METRICS: mAP@5 = {test_map5} (skip {test_skip})')\n",
    "    print(test_df.to_string(index=False))\n",
    "except NameError:\n",
    "    print('\\n(Không có test_indices – chạy cell split trước nếu cần)')\n",
    "\n",
    "# ── Lưu CSV ──────────────────────────────────────────────────────────────\n",
    "full_df['mAP@5'] = full_map5\n",
    "full_df['model'] = 'DINOv3 + MultilingualMiniLM'\n",
    "full_df['split'] = 'full'\n",
    "csv_out = os.path.join(output_dir, 'metrics_Hung_DINOv3_MultilingualMiniLM.csv')\n",
    "full_df.to_csv(csv_out, index=False)\n",
    "print(f'\\n✅ Lưu metric: {csv_out}')\n",
], "metric_code_v2_01")

# ─────────────────────────────────────────────────────────────────────────────
# Áp patch vào notebook
# ─────────────────────────────────────────────────────────────────────────────
cells = nb["cells"]
change_count = 0

if idx_grid is not None:
    cells[idx_grid] = GRID_CELL_NEW
    change_count += 1
    print(f"Replaced grid cell @ {idx_grid}")

if idx_sim is not None:
    cells[idx_sim] = SIM_CELL_NEW
    change_count += 1
    print(f"Replaced sim cell @ {idx_sim}")

if idx_metric is not None:
    cells[idx_metric] = METRIC_CELL_NEW
    change_count += 1
    print(f"Replaced metric cell @ {idx_metric}")

print(f"\nThay thế {change_count}/3 cells.")

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\n✅ Patch v3 hoàn tất: {OUT_PATH}")
print("\nThay đổi so với v2:")
print("  [Fix] Grid search có SAFETY CHECK – revert về alpha=0.7, thresh=2 nếu không tốt hơn")
print("  [Fix] threshold_grid = [2,3,5] thay vì [2,5,8,10] – tránh false positives")
print("  [Fix] Metric cell in rõ 'skip N items' thay vì nhập nhằng")
print("  [Fix] Comparison với notebook gốc (0.7388) được in rõ ngay trong output")
print("\nLưu ý quan trọng:")
print("  → Để Fix 2 (multilingual MiniLM) có tác dụng,")
print("    phải re-run cell trích xuất features từ đầu!")
