import json

nb_path = r'd:\New folder (3)\project\notebooks\Nhom3thangcuti_Tuan4\notebooks\Tuan4_GiaVy_Test.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ============================================================
# Các cell mới cần thêm vào cuối notebook
# ============================================================

new_cells = []

# --- Cell 12 (index): Markdown header Pipeline 2 giai đoạn ---
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "# 🚀 PIPELINE 2 GIAI ĐOẠN: MobileCLIP + DINOv2 Re-ranking\n",
        "\n",
        "**Kiến trúc:**\n",
        "- **Giai đoạn 1 (GĐ1):** MobileCLIP lấy top-K ứng viên (K = 100)\n",
        "- **Giai đoạn 2 (GĐ2):** DINOv2-Small tính lại similarity → sắp xếp lại → Top-5 cuối cùng\n",
        "\n",
        "**Mục tiêu:** Nâng cấp baseline MobileCLIP + Alpha Tuning (mAP@5 ≈ 0.77)"
    ]
})

# --- Cell: Markdown bước 0 ---
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 🧠 Bước 0: Chuẩn bị – Lưu các thành phần MobileCLIP đã có"
    ]
})

# --- Cell: Code bước 0 - lưu gallery features và build FAISS ---
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import os\n",
        "import numpy as np\n",
        "import faiss\n",
        "\n",
        "# ── Đường dẫn lưu cache features ──────────────────────────────\n",
        "FEAT_DIR = '/content/features'\n",
        "os.makedirs(FEAT_DIR, exist_ok=True)\n",
        "\n",
        "# ── Lưu cache gallery features MobileCLIP (nếu chưa có) ───────\n",
        "GALLERY_IMG_CACHE = os.path.join(FEAT_DIR, 'mobileclip_gallery_img.npy')\n",
        "GALLERY_TXT_CACHE = os.path.join(FEAT_DIR, 'mobileclip_gallery_txt.npy')\n",
        "\n",
        "np.save(GALLERY_IMG_CACHE, gallery_img_feats)\n",
        "np.save(GALLERY_TXT_CACHE, gallery_txt_feats)\n",
        "print(f'✅ Đã lưu gallery features MobileCLIP:')\n",
        "print(f'   img: {GALLERY_IMG_CACHE}  shape={gallery_img_feats.shape}')\n",
        "print(f'   txt: {GALLERY_TXT_CACHE}  shape={gallery_txt_feats.shape}')\n",
        "\n",
        "# ── Build FAISS index cho GĐ1 ─────────────────────────────────\n",
        "# Dùng best_alpha_2 từ bước grid search đã chạy\n",
        "gallery_fused_stage1 = fuse_and_normalize_clip(\n",
        "    gallery_img_feats, gallery_txt_feats, best_alpha_2\n",
        ")\n",
        "faiss_index_stage1 = build_faiss_index(gallery_fused_stage1)\n",
        "\n",
        "print(f'\\n✅ FAISS index GĐ1 sẵn sàng, best_alpha_2 = {best_alpha_2:.1f}')\n",
        "print(f'   Gallery size: {faiss_index_stage1.ntotal:,} vectors, dim={gallery_fused_stage1.shape[1]}')"
    ]
})

# --- Cell: Markdown bước 1 ---
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 🦕 Bước 1: Cache Đặc Trưng DINOv2-Small Cho Gallery\n",
        "\n",
        "Chạy **1 lần duy nhất** – kết quả được lưu vào `features/dinov2_gallery.npy`.\n",
        "\n",
        "> **Lưu ý quản lý VRAM:** DINOv2 và MobileCLIP không thể đồng thời trên GPU T4.  \n",
        "> Hãy chắc chắn **xóa MobileCLIP** trước khi load DINOv2 (xem Bước 1.5)."
    ]
})

# --- Cell: Code bước 1 - Load và cache DINOv2 ---
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import torch\n",
        "import torch.nn.functional as F\n",
        "from torchvision import transforms\n",
        "import numpy as np\n",
        "from PIL import Image\n",
        "from tqdm.notebook import tqdm\n",
        "import gc, os\n",
        "\n",
        "DINO_CACHE_PATH = os.path.join(FEAT_DIR, 'dinov2_gallery.npy')\n",
        "\n",
        "if not os.path.exists(DINO_CACHE_PATH):\n",
        "    print('🔄 Đang tính toán DINOv2 gallery features (lần đầu tiên)...')\n",
        "    print('⚠️  Dọn VRAM trước khi load DINOv2...')\n",
        "\n",
        "    # ── 1.5: Giải phóng MobileCLIP khỏi VRAM ──────────────────\n",
        "    try:\n",
        "        del clip_model\n",
        "        print('   clip_model đã xóa')\n",
        "    except NameError:\n",
        "        pass\n",
        "    gc.collect()\n",
        "    torch.cuda.empty_cache()\n",
        "    print('🧹 VRAM đã dọn sạch!')\n",
        "\n",
        "    # ── 1.1: Load DINOv2-Small ─────────────────────────────────\n",
        "    print('\\n🦕 Đang load DINOv2-Small...')\n",
        "    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')\n",
        "    dinov2 = dinov2.to(DEVICE).eval()\n",
        "    print('✅ DINOv2-Small sẵn sàng!')\n",
        "\n",
        "    # ── 1.2: Transform ảnh chuẩn DINOv2 ───────────────────────\n",
        "    transform_dino = transforms.Compose([\n",
        "        transforms.Resize(256),\n",
        "        transforms.CenterCrop(224),\n",
        "        transforms.ToTensor(),\n",
        "        transforms.Normalize(mean=[0.485, 0.456, 0.406],\n",
        "                             std=[0.229, 0.224, 0.225])\n",
        "    ])\n",
        "\n",
        "    @torch.no_grad()\n",
        "    def get_dinov2_embedding(img_path):\n",
        "        \"\"\"Trả về vector 384 chiều đã chuẩn hóa L2.\"\"\"\n",
        "        try:\n",
        "            img = Image.open(img_path).convert('RGB')\n",
        "        except Exception:\n",
        "            img = Image.new('RGB', (224, 224), (128, 128, 128))\n",
        "        img_tensor = transform_dino(img).unsqueeze(0).to(DEVICE)\n",
        "        feats = dinov2.forward_features(img_tensor)['x_norm_clstoken']\n",
        "        feats = F.normalize(feats, dim=-1)\n",
        "        return feats.cpu().numpy().flatten()\n",
        "\n",
        "    # ── 1.3: Tính và lưu cache ─────────────────────────────────\n",
        "    gallery_paths = [\n",
        "        os.path.join(IMG_DIR, fname) for fname in df_gallery['image']\n",
        "    ]\n",
        "    gallery_dino = []\n",
        "    for path in tqdm(gallery_paths, desc='🦕 DINOv2 gallery features'):\n",
        "        gallery_dino.append(get_dinov2_embedding(path))\n",
        "    gallery_dino = np.array(gallery_dino, dtype='float32')\n",
        "    np.save(DINO_CACHE_PATH, gallery_dino)\n",
        "    print(f'\\n✅ Đặc trưng DINOv2 gallery: {gallery_dino.shape}')\n",
        "    print(f'   Đã lưu: {DINO_CACHE_PATH}')\n",
        "\n",
        "else:\n",
        "    print(f'✅ Load DINOv2 gallery cache từ file...')\n",
        "    gallery_dino = np.load(DINO_CACHE_PATH)\n",
        "\n",
        "    # Cần load lại DINOv2 model cho inference query\n",
        "    if 'dinov2' not in dir() or dinov2 is None:\n",
        "        print('🔄 Đang load lại DINOv2-Small model...')\n",
        "        try:\n",
        "            del clip_model\n",
        "        except NameError:\n",
        "            pass\n",
        "        gc.collect()\n",
        "        torch.cuda.empty_cache()\n",
        "        dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')\n",
        "        dinov2 = dinov2.to(DEVICE).eval()\n",
        "\n",
        "        transform_dino = transforms.Compose([\n",
        "            transforms.Resize(256),\n",
        "            transforms.CenterCrop(224),\n",
        "            transforms.ToTensor(),\n",
        "            transforms.Normalize(mean=[0.485, 0.456, 0.406],\n",
        "                                 std=[0.229, 0.224, 0.225])\n",
        "        ])\n",
        "\n",
        "        @torch.no_grad()\n",
        "        def get_dinov2_embedding(img_path):\n",
        "            \"\"\"Trả về vector 384 chiều đã chuẩn hóa L2.\"\"\"\n",
        "            try:\n",
        "                img = Image.open(img_path).convert('RGB')\n",
        "            except Exception:\n",
        "                img = Image.new('RGB', (224, 224), (128, 128, 128))\n",
        "            img_tensor = transform_dino(img).unsqueeze(0).to(DEVICE)\n",
        "            feats = dinov2.forward_features(img_tensor)['x_norm_clstoken']\n",
        "            feats = F.normalize(feats, dim=-1)\n",
        "            return feats.cpu().numpy().flatten()\n",
        "\n",
        "print(f'\\n✅ DINOv2 gallery shape: {gallery_dino.shape}')"
    ]
})

# --- Cell: Markdown bước 2 ---
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 🔎 Bước 2: Hàm Tìm Kiếm 2 Giai Đoạn"
    ]
})

# --- Cell: Code bước 2 - search_two_stage ---
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Xây dựng mapping index: posting_id -> index trong gallery\n",
        "gallery_pids_list = df_gallery['posting_id'].tolist()\n",
        "\n",
        "\n",
        "def search_mobileclip_stage1(query_fused_vec, top_k=100):\n",
        "    \"\"\"\n",
        "    GĐ1: Dùng FAISS để tìm top_k ứng viên.\n",
        "    query_fused_vec: np.array shape (dim,) đã fusion + normalize.\n",
        "    Trả về: (candidate_indices, candidate_scores)\n",
        "    \"\"\"\n",
        "    q = query_fused_vec.reshape(1, -1).astype(np.float32)\n",
        "    scores, indices = faiss_index_stage1.search(q, top_k)\n",
        "    return indices[0].tolist(), scores[0].tolist()\n",
        "\n",
        "\n",
        "def fuse_single_clip(img_feat, txt_feat, alpha):\n",
        "    \"\"\"Fusion + L2-normalize cho 1 query.\"\"\"\n",
        "    fused = alpha * img_feat + (1 - alpha) * txt_feat\n",
        "    norm = np.linalg.norm(fused)\n",
        "    if norm < 1e-10:\n",
        "        norm = 1e-10\n",
        "    return (fused / norm).astype(np.float32)\n",
        "\n",
        "\n",
        "def search_two_stage(query_img_path, query_img_feat, query_txt_feat,\n",
        "                     alpha, retrieval_k=100, final_k=5):\n",
        "    \"\"\"\n",
        "    Pipeline 2 giai đoạn:\n",
        "      GĐ1: MobileCLIP FAISS → top retrieval_k ứng viên\n",
        "      GĐ2: DINOv2 tính similarity lại → chọn final_k tốt nhất\n",
        "\n",
        "    Args:\n",
        "        query_img_path  : đường dẫn ảnh query (để DINOv2 encode)\n",
        "        query_img_feat  : np.array (dim,) – MobileCLIP image feature\n",
        "        query_txt_feat  : np.array (dim,) – MobileCLIP text feature\n",
        "        alpha           : trọng số fusion (từ grid search)\n",
        "        retrieval_k     : số ứng viên lấy từ GĐ1\n",
        "        final_k         : số kết quả trả về sau GĐ2\n",
        "\n",
        "    Returns:\n",
        "        final_indices   : list[int] – indices trong df_gallery\n",
        "        final_scores    : list[float] – DINOv2 similarity scores\n",
        "    \"\"\"\n",
        "    # ─── Giai đoạn 1: MobileCLIP FAISS ───────────────────────────\n",
        "    q_fused = fuse_single_clip(query_img_feat, query_txt_feat, alpha)\n",
        "    candidate_indices, _ = search_mobileclip_stage1(q_fused, top_k=retrieval_k)\n",
        "\n",
        "    # ─── Giai đoạn 2: DINOv2 Re-ranking ─────────────────────────\n",
        "    q_dino = get_dinov2_embedding(query_img_path)  # (384,)\n",
        "    rerank_scores = [\n",
        "        float(np.dot(q_dino, gallery_dino[idx]))\n",
        "        for idx in candidate_indices\n",
        "    ]\n",
        "\n",
        "    # Sắp xếp lại theo score DINOv2 (giảm dần)\n",
        "    new_order = np.argsort(rerank_scores)[::-1]\n",
        "    final_indices = [candidate_indices[i] for i in new_order[:final_k]]\n",
        "    final_scores  = [rerank_scores[i]     for i in new_order[:final_k]]\n",
        "    return final_indices, final_scores\n",
        "\n",
        "\n",
        "print('✅ Hàm search_two_stage sẵn sàng!')"
    ]
})

# --- Cell: Markdown bước 3 ---
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 🎛️ Bước 3: Hàm Đánh Giá + Tuning `retrieval_k` Trên Validation Set\n",
        "\n",
        "> **Lưu ý:** Chỉ tuning trên **Val set** – KHÔNG dùng Test set để tuning!"
    ]
})

# --- Cell: Code bước 3 - evaluate_map_two_stage + tuning k ---
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from tqdm.notebook import tqdm\n",
        "\n",
        "\n",
        "def evaluate_map_two_stage(query_df, alpha, retrieval_k, final_k=5):\n",
        "    \"\"\"\n",
        "    Đánh giá mAP@final_k với pipeline 2 giai đoạn.\n",
        "    Chỉ dùng trên Val set!\n",
        "\n",
        "    Args:\n",
        "        query_df    : DataFrame (val hoặc test)\n",
        "        alpha       : trọng số fusion MobileCLIP\n",
        "        retrieval_k : số ứng viên lấy từ GĐ1\n",
        "        final_k     : số kết quả đánh giá cuối cùng\n",
        "    Returns:\n",
        "        dict chứa mAP@5, Precision@1, Recall@5\n",
        "    \"\"\"\n",
        "    gt_dict      = get_ground_truth_dict(df_gallery)\n",
        "    gallery_pids = df_gallery['posting_id'].tolist()\n",
        "    ap_list, p1_list, r5_list = [], [], []\n",
        "\n",
        "    # Pre-compute query features (img + txt)\n",
        "    q_idxs = query_df.index.tolist()\n",
        "\n",
        "    for i, row in tqdm(\n",
        "        enumerate(query_df.itertuples()),\n",
        "        total=len(query_df),\n",
        "        desc=f'🔍 Eval 2-stage (k={retrieval_k})'\n",
        "    ):\n",
        "        qid = row.posting_id\n",
        "        relevant = gt_dict.get(qid, set()) - {qid}\n",
        "        if not relevant:\n",
        "            continue\n",
        "\n",
        "        query_img_path = os.path.join(IMG_DIR, row.image)\n",
        "\n",
        "        # Lấy feature MobileCLIP của query từ ma trận đã trích xuất\n",
        "        if query_df is df_val:\n",
        "            q_img_feat = val_img_feats[i]\n",
        "            q_txt_feat = val_txt_feats[i]\n",
        "        else:  # df_test\n",
        "            q_img_feat = test_img_feats[i]\n",
        "            q_txt_feat = test_txt_feats[i]\n",
        "\n",
        "        try:\n",
        "            pred_indices, _ = search_two_stage(\n",
        "                query_img_path, q_img_feat, q_txt_feat,\n",
        "                alpha=alpha, retrieval_k=retrieval_k, final_k=final_k\n",
        "            )\n",
        "        except Exception as e:\n",
        "            ap_list.append(0.0)\n",
        "            p1_list.append(0.0)\n",
        "            r5_list.append(0.0)\n",
        "            continue\n",
        "\n",
        "        retrieved = [gallery_pids[idx] for idx in pred_indices]\n",
        "        retrieved = [p for p in retrieved if p != qid][:final_k]\n",
        "\n",
        "        hits, ap = 0, 0.0\n",
        "        for rank, pid in enumerate(retrieved, 1):\n",
        "            if pid in relevant:\n",
        "                hits += 1\n",
        "                ap   += hits / rank\n",
        "        ap_list.append(ap / min(len(relevant), final_k))\n",
        "        p1_list.append(1.0 if (retrieved and retrieved[0] in relevant) else 0.0)\n",
        "        r5_list.append(len(set(retrieved) & relevant) / len(relevant))\n",
        "\n",
        "    return {\n",
        "        'mAP@5'      : float(np.mean(ap_list)),\n",
        "        'Precision@1': float(np.mean(p1_list)),\n",
        "        'Recall@5'   : float(np.mean(r5_list)),\n",
        "    }\n",
        "\n",
        "\n",
        "# ── Tuning retrieval_k trên VAL SET ───────────────────────────\n",
        "best_k_dino     = 100\n",
        "best_val_map_dino = 0.0\n",
        "k_results_dino  = []\n",
        "\n",
        "print(f'🎛️  Tuning retrieval_k với best_alpha_2 = {best_alpha_2:.1f}')\n",
        "print('─' * 60)\n",
        "\n",
        "for k in [50, 100, 150, 200]:\n",
        "    val_m = evaluate_map_two_stage(\n",
        "        df_val, best_alpha_2, retrieval_k=k, final_k=5\n",
        "    )\n",
        "    k_results_dino.append({'retrieval_k': k, **val_m})\n",
        "    marker = ' ← BEST' if val_m['mAP@5'] > best_val_map_dino else ''\n",
        "    print(\n",
        "        f'  retrieval_k={k:3d} | '\n",
        "        f'mAP@5={val_m[\"mAP@5\"]:.4f} | '\n",
        "        f'P@1={val_m[\"Precision@1\"]:.4f} | '\n",
        "        f'R@5={val_m[\"Recall@5\"]:.4f}{marker}'\n",
        "    )\n",
        "    if val_m['mAP@5'] > best_val_map_dino:\n",
        "        best_val_map_dino = val_m['mAP@5']\n",
        "        best_k_dino       = k\n",
        "\n",
        "print('─' * 60)\n",
        "print(f'\\n✅ Chọn retrieval_k = {best_k_dino} với Val mAP@5 = {best_val_map_dino:.4f}')\n",
        "\n",
        "# Bảng đầy đủ\n",
        "import pandas as pd\n",
        "df_k_results = pd.DataFrame(k_results_dino)\n",
        "print('\\n📊 Bảng đầy đủ:')\n",
        "print(df_k_results.to_string(index=False))"
    ]
})

# --- Cell: Markdown bước 4 ---
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 🧪 Bước 4: Đánh Giá CUỐI CÙNG Trên Test Set (Chạy 1 Lần!)\n",
        "\n",
        "> ⚠️ **CHỈ CHẠY CELL NÀY 1 LẦN DUY NHẤT** sau khi đã hoàn tất tuning trên Val set."
    ]
})

# --- Cell: Code bước 4 - đánh giá test ---
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import pandas as pd\n",
        "\n",
        "print('⚠️  Đánh giá TEST SET – CHỈ CHẠY 1 LẦN DUY NHẤT!')\n",
        "print(f'   best_alpha_2 = {best_alpha_2:.1f}')\n",
        "print(f'   best_k_dino  = {best_k_dino}')\n",
        "print('─' * 60)\n",
        "\n",
        "test_metrics_dino = evaluate_map_two_stage(\n",
        "    df_test, best_alpha_2,\n",
        "    retrieval_k=best_k_dino, final_k=5\n",
        ")\n",
        "\n",
        "print(f'\\n🏆 KẾT QUẢ CUỐI CÙNG – MobileCLIP + DINOv2 Re-ranking:')\n",
        "print(f'   mAP@5        = {test_metrics_dino[\"mAP@5\"]:.4f}')\n",
        "print(f'   Precision@1  = {test_metrics_dino[\"Precision@1\"]:.4f}')\n",
        "print(f'   Recall@5     = {test_metrics_dino[\"Recall@5\"]:.4f}')\n",
        "\n",
        "# So sánh với baseline\n",
        "improvement = test_metrics_dino['mAP@5'] - test_metrics_2['mAP@5']\n",
        "print(f'\\n📈 So sánh với Baseline MobileCLIP:')\n",
        "print(f'   Baseline mAP@5      = {test_metrics_2[\"mAP@5\"]:.4f}')\n",
        "print(f'   2-Stage mAP@5       = {test_metrics_dino[\"mAP@5\"]:.4f}')\n",
        "print(f'   Cải thiện (Δ mAP@5) = {improvement:+.4f}')"
    ]
})

# --- Cell: Markdown bước 5 xuất file ---
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 💾 Bước 5: Xuất Kết Quả Ra File CSV"
    ]
})

# --- Cell: Code bước 5 - lưu CSV ---
new_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import pandas as pd, os\n",
        "\n",
        "# ── Tổng hợp tất cả metrics ───────────────────────────────────\n",
        "metrics_data_full = [\n",
        "    {\n",
        "        'Method'          : 'MobileCLIP (Zero-Shot Fusion)',\n",
        "        'Dim'             : str(embed_dim),\n",
        "        'Alpha'           : round(float(best_alpha_2), 1),\n",
        "        'Retrieval_K'     : '-',\n",
        "        'Test_mAP@5'      : round(test_metrics_2['mAP@5'], 4),\n",
        "        'Test_Precision@1': round(test_metrics_2['Precision@1'], 4),\n",
        "        'Test_Recall@5'   : round(test_metrics_2['Recall@5'], 4),\n",
        "    },\n",
        "    {\n",
        "        'Method'          : 'MobileCLIP + DINOv2 Re-ranking',\n",
        "        'Dim'             : f'{embed_dim}+384',\n",
        "        'Alpha'           : round(float(best_alpha_2), 1),\n",
        "        'Retrieval_K'     : best_k_dino,\n",
        "        'Test_mAP@5'      : round(test_metrics_dino['mAP@5'], 4),\n",
        "        'Test_Precision@1': round(test_metrics_dino['Precision@1'], 4),\n",
        "        'Test_Recall@5'   : round(test_metrics_dino['Recall@5'], 4),\n",
        "    },\n",
        "]\n",
        "\n",
        "df_final = pd.DataFrame(metrics_data_full)\n",
        "print('📊 Final Metrics Table:')\n",
        "print(df_final.to_string(index=False))\n",
        "\n",
        "# ── Lưu local (/content/) ─────────────────────────────────────\n",
        "LOCAL_CSV_DINO = '/content/final_metric_dinov2.csv'\n",
        "df_final.to_csv(LOCAL_CSV_DINO, index=False, encoding='utf-8-sig')\n",
        "print(f'\\n✅ Đã lưu local : {LOCAL_CSV_DINO}')\n",
        "\n",
        "# ── Lưu lên Google Drive ──────────────────────────────────────\n",
        "DRIVE_CSV_DINO = os.path.join(DATA_DIR, 'final_metric_dinov2.csv')\n",
        "df_final.to_csv(DRIVE_CSV_DINO, index=False, encoding='utf-8-sig')\n",
        "print(f'✅ Đã lưu Drive  : {DRIVE_CSV_DINO}')\n",
        "\n",
        "print('\\n🎉 Hoàn tất pipeline MobileCLIP + DINOv2 Re-ranking!')"
    ]
})

# ============================================================
# Thêm các cell mới vào notebook
# ============================================================
nb['cells'].extend(new_cells)

# ── Ghi lại notebook ──────────────────────────────────────────
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'✅ Đã thêm {len(new_cells)} cells vào notebook!')
print(f'   Tổng số cells: {len(nb["cells"])}')
print(f'   File: {nb_path}')
