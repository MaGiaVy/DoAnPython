import json

notebook_path = 'd:/New folder (3)/project/notebooks/Nhom3thangcuti_Tuan4/Tuan4_GiaVy_Pipeline+pHash.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Bug 1 & Bug 2
        source = source.replace(
            "feats = clip_model.get_image_features(**inputs)\n            all_feats.append(feats.cpu().float().numpy())",
            "feats = clip_model.get_image_features(**inputs)\n            feats = feats / feats.norm(dim=-1, keepdim=True)\n            all_feats.append(feats.cpu().float().numpy())"
        )
        source = source.replace(
            "feats = clip_model.encode_image(imgs.to(DEVICE))\n            all_feats.append(feats.cpu().float().numpy())",
            "feats = clip_model.encode_image(imgs.to(DEVICE))\n            feats = feats / feats.norm(dim=-1, keepdim=True)\n            all_feats.append(feats.cpu().float().numpy())"
        )
        
        # Cải tiến 4 (which also fixes Bug 3 indirectly since fuse_and_normalize_clip is rewritten, but Bug 3 mentions np.where to np.maximum, let's replace both just in case)
        source = source.replace(
            "def fuse_and_normalize_clip(img_feats, txt_feats, alpha):\n    fused = alpha * img_feats + (1 - alpha) * txt_feats\n    norms = np.linalg.norm(fused, axis=1, keepdims=True)\n    norms = np.where(norms == 0, 1e-10, norms)\n    return (fused / norms).astype(np.float32)",
            "def fuse_and_normalize_clip(img_feats, txt_feats, alpha):\n    fused = (alpha * img_feats + (1 - alpha) * txt_feats).astype(np.float32)\n    faiss.normalize_L2(fused)\n    return fused"
        )
        
        # Bug 4
        source = source.replace(
            "if 'dinov2' not in dir() or dinov2 is None:",
            "if 'dinov2' not in globals() or dinov2 is None:"
        )
        
        # Bug 5
        source = source.replace(
            "r5_list.append(len(set(retrieved) & relevant) / len(relevant))",
            "r5_list.append(len(set(retrieved) & relevant) / min(len(relevant), K))"
        )
        source = source.replace(
            "r5_list.append(len(set(retrieved_pids) & relevant) / len(relevant))",
            "r5_list.append(len(set(retrieved_pids) & relevant) / min(len(relevant), final_k))"
        )
        
        # Cải tiến 1
        source = source.replace(
            "def compute_phash(img_path, hash_size=8):",
            "def compute_phash(img_path, hash_size=16):"
        )
        
        # Cải tiến 3
        source = source.replace(
            "HAMMING_THRESHOLD = 5    # chỉ thưởng cho ảnh rất giống\nPHASH_BONUS       = 0.03 # điểm thưởng nhẹ, đủ để phá thế hoà",
            "HAMMING_THRESHOLD = 8    # chỉ thưởng cho ảnh rất giống\nPHASH_BONUS       = 0.10 # điểm thưởng nhẹ, đủ để phá thế hoà"
        )
        source = source.replace(
            "HAMMING_THRESHOLD=5, PHASH_BONUS=0.03",
            "HAMMING_THRESHOLD=8, PHASH_BONUS=0.10"
        )
        
        # Cải tiến 6
        source = source.replace(
            "dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')",
            "dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')"
        )
        source = source.replace(
            "vector 384 chiều",
            "vector 768 chiều"
        )
        source = source.replace(
            "(34250, 384)",
            "(34250, 768)"
        )
        
        # Cải tiến 2
        # Adding ahash to cell 15
        if "gallery_phashes.append(h)" in source and "average_hash" not in source:
            source = source.replace(
                "gallery_phashes = []\n    for fname in tqdm(df_gallery['image'], desc='🔑 pHash gallery'):\n        img_path = os.path.join(IMG_DIR, fname)\n        h = compute_phash(img_path)\n        gallery_phashes.append(h)",
                "gallery_phashes = []\n    gallery_ahashes = []\n    for fname in tqdm(df_gallery['image'], desc='🔑 pHash & aHash gallery'):\n        img_path = os.path.join(IMG_DIR, fname)\n        h = compute_phash(img_path)\n        gallery_phashes.append(h)\n        ahash = imagehash.average_hash(Image.open(img_path).convert('RGB'))\n        gallery_ahashes.append(ahash)"
            )
            source = source.replace(
                "gallery_phash_strs = [str(h) for h in gallery_phashes]\n    np.save(PHASH_CACHE_PATH, gallery_phash_strs)",
                "gallery_phash_strs = [str(h) for h in gallery_phashes]\n    gallery_ahash_strs = [str(h) for h in gallery_ahashes]\n    np.save(PHASH_CACHE_PATH, gallery_phash_strs)\n    np.save(PHASH_CACHE_PATH.replace('phashes', 'ahashes'), gallery_ahash_strs)"
            )
            source = source.replace(
                "gallery_phash_strs = np.load(PHASH_CACHE_PATH, allow_pickle=True)\n    gallery_phashes = [imagehash.hex_to_hash(s) for s in gallery_phash_strs]",
                "gallery_phash_strs = np.load(PHASH_CACHE_PATH, allow_pickle=True)\n    gallery_phashes = [imagehash.hex_to_hash(s) for s in gallery_phash_strs]\n    gallery_ahash_strs = np.load(PHASH_CACHE_PATH.replace('phashes', 'ahashes'), allow_pickle=True)\n    gallery_ahashes = [imagehash.hex_to_hash(s) for s in gallery_ahash_strs]"
            )

        # Cải tiến 2 for cell 16 (search_two_stage)
        if "if use_phash and 'gallery_phashes' in globals()" in source and "q_ahash" not in source:
            source = source.replace(
                "    if use_phash and 'gallery_phashes' in globals() and gallery_phashes is not None:\n        q_phash = compute_phash(query_img_path)\n        for i, idx in enumerate(candidate_indices):\n            hamming_dist = q_phash - gallery_phashes[idx]  # imagehash tự tính Hamming\n            if hamming_dist <= HAMMING_THRESHOLD:\n                combined[i] += PHASH_BONUS",
                "    if use_phash and 'gallery_phashes' in globals() and gallery_phashes is not None:\n        q_phash = compute_phash(query_img_path)\n        q_ahash = imagehash.average_hash(Image.open(query_img_path).convert('RGB'))\n        for i, idx in enumerate(candidate_indices):\n            phash_dist = q_phash - gallery_phashes[idx]\n            ahash_dist = q_ahash - gallery_ahashes[idx]\n            if phash_dist <= HAMMING_THRESHOLD and ahash_dist <= HAMMING_THRESHOLD:\n                combined[i] += PHASH_BONUS"
            )

        # Cải tiến 5: Fine grained search in cell 9
        # Wait, the markdown shows how to add it, but it might be easier to just append it if 'alphas_coarse' is present, or just inject it.
        # Currently, the script has `alphas = np.arange(0.1, 1.0, 0.1).round(1)`
        if "alphas = np.arange(0.1, 1.0, 0.1).round(1)" in source and "alphas_fine" not in source:
            source = source.replace(
                "alphas = np.arange(0.1, 1.0, 0.1).round(1)\nprint(f'🔍 Grid search alpha ∈ {alphas.tolist()}')",
                "alphas_coarse = np.arange(0.1, 1.0, 0.1).round(1)\nprint(f'🔍 Grid search coarse alpha ∈ {alphas_coarse.tolist()}')"
            )
            source = source.replace(
                "for alpha in alphas:",
                "for alpha in alphas_coarse:"
            )
            source = source.replace(
                "df_val_summary = pd.DataFrame(val_results_2)\nprint('\\n📊 Bảng val đầy đủ:')\nprint(df_val_summary.to_string(index=False))",
                "df_val_summary = pd.DataFrame(val_results_2)\nprint('\\n📊 Bảng val đầy đủ (Coarse):')\nprint(df_val_summary.to_string(index=False))\n\nalphas_fine = np.arange(max(0.0, best_alpha_2 - 0.09), min(1.0, best_alpha_2 + 0.10), 0.02).round(2)\nprint(f'\\n🔍 Grid search fine alpha ∈ {alphas_fine.tolist()}')\nfor alpha in alphas_fine:\n    m = evaluate_retrieval_clip(\n        query_df    = df_val,\n        gallery_df  = df_gallery,\n        query_img   = val_img_feats,\n        query_txt   = val_txt_feats,\n        gallery_img = gallery_img_feats,\n        gallery_txt = gallery_txt_feats,\n        alpha=alpha, K=5\n    )\n    val_results_2.append({'alpha': alpha, **m})\n    marker = ' ← best' if m['mAP@5'] > best_map5_2 else ''\n    print(f'  α={alpha:.2f} | mAP@5={m[\"mAP@5\"]:.4f} | '\n          f'P@1={m[\"Precision@1\"]:.4f} | R@5={m[\"Recall@5\"]:.4f}{marker}')\n    if m['mAP@5'] > best_map5_2:\n        best_map5_2   = m['mAP@5']\n        best_alpha_2  = alpha\n\nprint('─' * 64)\nprint(f'\\n🏆 FINAL BEST_ALPHA_2 = {best_alpha_2:.2f}  (Val mAP@5 = {best_map5_2:.4f})')\ndf_val_summary = pd.DataFrame(val_results_2)\nprint('\\n📊 Bảng val đầy đủ (Coarse + Fine):')\nprint(df_val_summary.to_string(index=False))"
            )
        
        # Reconstruct lines
        lines = source.splitlines(True)
        cell['source'] = lines

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
