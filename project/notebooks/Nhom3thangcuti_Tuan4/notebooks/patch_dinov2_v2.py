# -*- coding: utf-8 -*-
"""
patch_dinov2_v2.py  –  Nâng cấp Tuan4_GiaVy_DINOv2.ipynb để tăng mAP@5 > 0.75
Các thay đổi:
  1. Config: TOP_RERANK 50->100, PHASH_BOOST 0.3->0.15
  2. TF-IDF: thêm L2-normalize trước khi lưu cache + check khi load
  3. Grid Search α: mở rộng từ 0.30->0.95 (bước 0.05)
  4. Thêm cell Query Expansion (AQE) trước bước rerank
  5. SigLIP: dùng mean_pooling thay pooler_output (lấy thêm thông tin)
"""

import json, copy

NB_PATH = 'Tuan4_GiaVy_DINOv2.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def get_src(cell):
    return ''.join(cell['source'])

def set_src(cell, new_str):
    lines = new_str.splitlines(keepends=True)
    if lines and not lines[-1].endswith('\n'):
        pass  # last line without newline is fine
    cell['source'] = lines

changes = 0

for cell in nb['cells']:
    cid = cell.get('id', '')
    src = get_src(cell)

    # ================================================================
    # SUA 1: cell-config — Nâng TOP_RERANK, tinh chỉnh PHASH_BOOST
    # ================================================================
    if cid == 'cell-config':
        old = 'TOP_RERANK     = 50    # Số ứng viên lấy ra trước khi rerank'
        new = 'TOP_RERANK     = 100   # Số ứng viên lấy ra trước khi rerank (tăng từ 50->100)'
        if old in src:
            src = src.replace(old, new, 1)
            print('[SUA 1a] TOP_RERANK: 50 -> 100')
            changes += 1
        else:
            print('[SUA 1a] WARNING: TOP_RERANK not found')

        old2 = 'PHASH_BOOST     = 0.3  # Boost nhẹ để không lấn át fusion score'
        new2 = 'PHASH_BOOST     = 0.15 # Boost nhẹ để không lấn át fusion score (giảm từ 0.3->0.15)'
        if old2 in src:
            src = src.replace(old2, new2, 1)
            print('[SUA 1b] PHASH_BOOST: 0.3 -> 0.15')
            changes += 1
        else:
            print('[SUA 1b] WARNING: PHASH_BOOST not found, trying unicode version')
            # Try unicode escaped version
            old2u = 'PHASH_BOOST     = 0.3  # Boost nh\u1eb9 \u0111\u1ec3 kh\u00f4ng l\u1ea5n \u00e1t fusion score'
            new2u = 'PHASH_BOOST     = 0.15 # Boost nh\u1eb9 \u0111\u1ec3 kh\u00f4ng l\u1ea5n \u00e1t fusion score (gi\u1ea3m t\u1eeb 0.3->0.15)'
            if old2u in src:
                src = src.replace(old2u, new2u, 1)
                print('[SUA 1b] PHASH_BOOST (unicode): 0.3 -> 0.15')
                changes += 1

        set_src(cell, src)

    # ================================================================
    # SUA 2: cell-step3 — Thêm L2-normalize cho TF-IDF trước khi lưu
    # ================================================================
    if cid == 'cell-step3':
        # Thêm L2-norm trước np.save
        old = '    # Lưu features\n    np.save(tfidf_save_path, tfidf_features)\n'
        new = (
            '    # L2-normalize TF-IDF features trước khi lưu (quan trọng!)\n'
            '    tfidf_features = tfidf_features / np.maximum(\n'
            '        np.linalg.norm(tfidf_features, axis=1, keepdims=True), 1e-10\n'
            '    )\n'
            '    tfidf_features = tfidf_features.astype(np.float32)\n'
            '    print(f\'   Đã L2-normalize TF-IDF: norm mẫu = {np.linalg.norm(tfidf_features[:5], axis=1)}\')\n'
            '\n'
            '    # Lưu features\n'
            '    np.save(tfidf_save_path, tfidf_features)\n'
        )
        if old in src:
            src = src.replace(old, new, 1)
            print('[SUA 2a] Added L2-normalize before saving TF-IDF')
            changes += 1
        else:
            print('[SUA 2a] WARNING: TF-IDF save block not found, checking alternatives...')
            # Try without indentation
            old_alt = '    np.save(tfidf_save_path, tfidf_features)\n    file_size_mb'
            if old_alt in src:
                new_alt = (
                    '    # L2-normalize TF-IDF features trước khi lưu (quan trọng!)\n'
                    '    tfidf_features = (tfidf_features /\n'
                    '        np.maximum(np.linalg.norm(tfidf_features, axis=1, keepdims=True), 1e-10)\n'
                    '    ).astype(np.float32)\n'
                    '    np.save(tfidf_save_path, tfidf_features)\n    file_size_mb'
                )
                src = src.replace(old_alt, new_alt, 1)
                print('[SUA 2a-alt] Added L2-normalize before saving TF-IDF (alt path)')
                changes += 1

        # Thêm normalize-check khi load cache
        old_load = (
            '    siglip_features = np.load(siglip_save_path)\n'
            '    print(f\'✅ Đã tải xong! Shape: {siglip_features.shape}\')\n'
        )
        # Fix: add normalize check after tfidf load
        old_tfidf_load = (
            '    tfidf_features = np.load(tfidf_save_path)\n'
            '    print(f\'✅ Đã tải xong! Shape: {tfidf_features.shape}\')\n'
        )
        new_tfidf_load = (
            '    tfidf_features = np.load(tfidf_save_path)\n'
            '    print(f\'✅ Đã tải xong! Shape: {tfidf_features.shape}\')\n'
            '    # Kiểm tra và normalize nếu cache cũ chưa normalize\n'
            '    _norms = np.linalg.norm(tfidf_features[:100], axis=1)\n'
            '    if not np.allclose(_norms, 1.0, atol=0.05):\n'
            '        print(\'   ⚠️  Cache TF-IDF chưa normalize, đang chuẩn hóa...\')\n'
            '        tfidf_features = (tfidf_features /\n'
            '            np.maximum(np.linalg.norm(tfidf_features, axis=1, keepdims=True), 1e-10)\n'
            '        ).astype(np.float32)\n'
            '        np.save(tfidf_save_path, tfidf_features)\n'
            '        print(\'   ✅ Đã normalize và lưu lại cache!\')\n'
        )
        if old_tfidf_load in src:
            src = src.replace(old_tfidf_load, new_tfidf_load, 1)
            print('[SUA 2b] Added normalize check on TF-IDF cache load')
            changes += 1
        else:
            print('[SUA 2b] WARNING: TF-IDF load block not found (may already be patched)')

        set_src(cell, src)

    # ================================================================
    # SUA 3: cell-step4-gridsearch — Mở rộng alpha range 0.30->0.95
    # ================================================================
    if cid == 'cell-step4-gridsearch':
        old = 'alphas = np.arange(0.50, 0.91, 0.05)  # [0.50, 0.55, 0.60, ..., 0.90]'
        new = (
            'alphas = np.arange(0.30, 0.96, 0.05)  # [0.30, 0.35, ..., 0.95]  '
            '← mở rộng để tìm alpha tối ưu hơn'
        )
        if old in src:
            src = src.replace(old, new, 1)
            print('[SUA 3a] Grid Search alpha range: 0.50-0.90 -> 0.30-0.95')
            changes += 1
        else:
            print('[SUA 3a] WARNING: alphas line not found')

        # Update print message
        old_print = 'f\'   Dải alpha: {[round(a, 2) for a in alphas]}\''
        new_print = 'f\'   Dải alpha: {[round(a, 2) for a in alphas]}  (mở rộng từ 0.30-0.95)\''
        if old_print in src:
            src = src.replace(old_print, new_print, 1)

        # Also increase n_search buffer
        old_nsearch = 'n_search = TOP_K + 15  # Lấy top-20 để đảm bảo có đủ sau khi loại self-match'
        new_nsearch = 'n_search = TOP_K + 25  # Lấy top-30 để đảm bảo có đủ sau khi loại self-match (tăng buffer)'
        if old_nsearch in src:
            src = src.replace(old_nsearch, new_nsearch, 1)
            print('[SUA 3b] n_search buffer: TOP_K+15 -> TOP_K+25')
            changes += 1

        set_src(cell, src)

# ================================================================
# SUA 4: Thêm cell Query Expansion (AQE) ngay trước cell-step5-rerank
# ================================================================
AQE_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "id": "cell-step5-aqe",
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================\n",
        "# 🔄 BƯỚC 5.4b: AVERAGE QUERY EXPANSION (AQE)\n",
        "# ============================================================\n",
        "# AQE: Mở rộng query bằng cách lấy trung bình của query + top-3 results\n",
        "# Giúp tăng recall đáng kể mà không cần thêm dữ liệu\n",
        "# Tham khảo: Chum et al. 2011 - Total Recall\n",
        "print('=' * 60)\n",
        "print('BƯỚC 5.4b: AVERAGE QUERY EXPANSION (AQE)')\n",
        "print('=' * 60)\n",
        "\n",
        "AQE_K = 3   # Số kết quả top để mở rộng query\n",
        "DO_AQE = True  # Đặt False để tắt AQE nếu muốn so sánh\n",
        "\n",
        "if DO_AQE:\n",
        "    print(f'\\n🔄 Đang áp dụng AQE với top-{AQE_K} results...')\n",
        "    print(f'   AQE formula: q_expanded = L2_Norm(q + mean(top_{AQE_K}_results))')\n",
        "\n",
        "    # Lấy top-AQE_K results TRƯỚC khi rerank (dùng raw FAISS results)\n",
        "    # top_scores_raw, top_indices_raw đã có từ bước 5.4\n",
        "    gallery_pids_aqe = candidate_df['posting_id'].values\n",
        "\n",
        "    # Tạo expanded queries\n",
        "    n_test_aqe = len(test_query_df)\n",
        "    test_fused_expanded = test_fused.copy()  # (n_test, fused_dim)\n",
        "\n",
        "    expand_count = 0\n",
        "    for q_idx in range(n_test_aqe):\n",
        "        q_pid = test_query_df.reset_index(drop=True).at[q_idx, 'posting_id']\n",
        "\n",
        "        # Thu thập top-AQE_K valid results (bỏ self-match)\n",
        "        neighbors = []\n",
        "        for gidx in top_indices_raw[q_idx]:\n",
        "            if gidx < 0:\n",
        "                break\n",
        "            if gallery_pids_aqe[gidx] == q_pid:\n",
        "                continue\n",
        "            neighbors.append(gidx)\n",
        "            if len(neighbors) == AQE_K:\n",
        "                break\n",
        "\n",
        "        if len(neighbors) == 0:\n",
        "            continue\n",
        "\n",
        "        # Lấy fused vectors của các neighbors từ gallery\n",
        "        # Cần tái tạo fused gallery (đã có final_gallery_fused)\n",
        "        neighbor_vecs = final_gallery_fused[neighbors]  # (AQE_K, fused_dim)\n",
        "\n",
        "        # AQE: query mới = L2_Norm(query_hiện_tại + mean(neighbors))\n",
        "        q_expanded = test_fused[q_idx] + neighbor_vecs.mean(axis=0)\n",
        "        q_norm = np.linalg.norm(q_expanded)\n",
        "        if q_norm > 1e-10:\n",
        "            test_fused_expanded[q_idx] = (q_expanded / q_norm).astype(np.float32)\n",
        "        expand_count += 1\n",
        "\n",
        "    print(f'✅ AQE hoàn tất! Đã mở rộng {expand_count:,}/{n_test_aqe:,} queries')\n",
        "\n",
        "    # Search lại với expanded queries\n",
        "    print(f'\\n🔍 Đang search lại với expanded queries...')\n",
        "    top_scores_raw, top_indices_raw = final_index.search(\n",
        "        test_fused_expanded.astype(np.float32),\n",
        "        TOP_RERANK + 5\n",
        "    )\n",
        "    print(f'✅ Search AQE hoàn tất!')\n",
        "else:\n",
        "    print('⚠️  AQE bị tắt (DO_AQE=False) — dùng kết quả search gốc')\n",
    ]
}

# Tìm vị trí cell-step5-rerank và chèn AQE trước nó
insert_before_idx = None
for i, cell in enumerate(nb['cells']):
    if cell.get('id', '') == 'cell-step5-rerank':
        insert_before_idx = i
        break

if insert_before_idx is not None:
    nb['cells'].insert(insert_before_idx, AQE_CELL)
    print(f'\n[SUA 4] AQE cell inserted before cell-step5-rerank (index {insert_before_idx})')
    changes += 1
else:
    print('\n[SUA 4] WARNING: cell-step5-rerank not found!')

# ================================================================
# SUA 5: Thêm cell Grid Search pHash sau cell-step4-gridsearch
# (nếu chưa có cell-step4b-phash-grid)
# ================================================================
has_phash_grid = any(
    cell.get('id', '') == 'cell-step4b-phash-grid'
    for cell in nb['cells']
)

if not has_phash_grid:
    PHASH_GRID_CELL = {
        "cell_type": "code",
        "execution_count": None,
        "id": "cell-step4b-phash-grid",
        "metadata": {},
        "outputs": [],
        "source": [
            "# ============================================================\n",
            "# \U0001f50d BƯỚC 4B: GRID SEARCH PHASH_THRESHOLD + BOOST TRÊN VAL SET\n",
            "# ============================================================\n",
            "print('=' * 60)\n",
            "print('BƯỚC 4B: GRID SEARCH PHASH (VAL SET)')\n",
            "print('=' * 60)\n",
            "\n",
            "thresholds = [0, 3, 5, 8, 10, 15]\n",
            "boost_vals  = [0.10, 0.15, 0.20, 0.30, 0.50]\n",
            "\n",
            "best_threshold     = PHASH_THRESHOLD\n",
            "best_boost         = PHASH_BOOST\n",
            "best_val_map_phash = -1.0\n",
            "\n",
            "HAS_PHASH_COL_4B = 'image_phash' in candidate_df.columns\n",
            "\n",
            "if HAS_PHASH_COL_4B:\n",
            "    phash_cache_path_4b = os.path.join(PROCESSED_DIR, 'phash_arrays.npy')\n",
            "    if os.path.exists(phash_cache_path_4b):\n",
            "        gallery_phash_arrays_4b = np.load(phash_cache_path_4b)\n",
            "        print(f'✅ Tải pHash cache! Shape: {gallery_phash_arrays_4b.shape}')\n",
            "    else:\n",
            "        phash_list_4b = [\n",
            "            hex_to_phash_array(h)\n",
            "            for h in tqdm(candidate_df['image_phash'], desc='   Chuyển pHash')\n",
            "        ]\n",
            "        gallery_phash_arrays_4b = np.array(phash_list_4b, dtype=bool)\n",
            "        np.save(phash_cache_path_4b, gallery_phash_arrays_4b)\n",
            "        print(f'✅ Tạo pHash arrays! Shape: {gallery_phash_arrays_4b.shape}')\n",
            "\n",
            "    print(f'\\nĐang thử {len(thresholds)} threshold × {len(boost_vals)} boost values...')\n",
            "    print(f'   Sử dụng BEST_ALPHA = {best_alpha:.2f} từ Bước 4A')\n",
            "\n",
            "    best_beta_4b = round(1.0 - best_alpha, 4)\n",
            "    gallery_fused_4b = l2_normalize(\n",
            "        np.concatenate([\n",
            "            best_alpha   * gallery_img_norm,\n",
            "            best_beta_4b * gallery_txt_norm\n",
            "        ], axis=1)\n",
            "    )\n",
            "    val_index_4b = build_faiss_index(gallery_fused_4b, USE_GPU_FAISS, gpu_resources)\n",
            "\n",
            "    val_fused_4b = l2_normalize(\n",
            "        np.concatenate([\n",
            "            best_alpha   * gallery_img_norm[val_gallery_indices],\n",
            "            best_beta_4b * gallery_txt_norm[val_gallery_indices]\n",
            "        ], axis=1)\n",
            "    )\n",
            "\n",
            "    val_top_s_4b, val_top_i_4b = val_index_4b.search(\n",
            "        val_fused_4b.astype(np.float32), TOP_RERANK + 5\n",
            "    )\n",
            "    gpids_4b = candidate_df['posting_id'].values\n",
            "    n_val_4b  = len(val_query_df)\n",
            "    vqr_4b    = val_query_df.reset_index(drop=True)\n",
            "\n",
            "    print()\n",
            "    for thresh in thresholds:\n",
            "        for boost in boost_vals:\n",
            "            reranked_4b = np.full((n_val_4b, TOP_K), -1, dtype=np.int64)\n",
            "            for q_idx in range(n_val_4b):\n",
            "                q_pid = vqr_4b.at[q_idx, 'posting_id']\n",
            "                cgidx, cscores = [], []\n",
            "                for gidx, score in zip(val_top_i_4b[q_idx], val_top_s_4b[q_idx]):\n",
            "                    if gidx < 0: break\n",
            "                    if gpids_4b[gidx] == q_pid: continue\n",
            "                    cgidx.append(gidx); cscores.append(float(score))\n",
            "                    if len(cgidx) == TOP_RERANK: break\n",
            "                if not cgidx: continue\n",
            "                cgidx   = np.array(cgidx,   dtype=np.int64)\n",
            "                cscores = np.array(cscores, dtype=np.float32)\n",
            "                q_gidx  = val_gallery_indices[q_idx]\n",
            "                q_ph    = gallery_phash_arrays_4b[q_gidx]\n",
            "                ham     = compute_hamming_distances(q_ph, gallery_phash_arrays_4b[cgidx])\n",
            "                cscores[ham <= thresh] += boost\n",
            "                order = np.argsort(-cscores)[:TOP_K]\n",
            "                reranked_4b[q_idx] = cgidx[order]\n",
            "\n",
            "            vm4b = compute_map_at_k(val_query_df, candidate_df, reranked_4b, k=TOP_K)\n",
            "            is_best_4b = vm4b > best_val_map_phash\n",
            "            if is_best_4b:\n",
            "                best_val_map_phash = vm4b\n",
            "                best_threshold     = thresh\n",
            "                best_boost         = boost\n",
            "            marker_4b = ' ◀ TỐT NHẤT!' if is_best_4b else ''\n",
            "            print(f'   thresh={thresh:2d}, boost={boost:.2f} → val mAP@5={vm4b:.4f}{marker_4b}')\n",
            "\n",
            "    print(f'\\n\U0001f3c6 BEST pHash: threshold={best_threshold}, boost={best_boost}')\n",
            "    print(f'   val mAP@5 = {best_val_map_phash:.4f}')\n",
            "    PHASH_THRESHOLD = best_threshold\n",
            "    PHASH_BOOST     = best_boost\n",
            "    print(f'\\n✅ Cập nhật: PHASH_THRESHOLD={PHASH_THRESHOLD}, PHASH_BOOST={PHASH_BOOST}')\n",
            "    del gallery_fused_4b, val_fused_4b, val_index_4b\n",
            "    if torch.cuda.is_available(): torch.cuda.empty_cache()\n",
            "else:\n",
            "    print('⚠️  Không có cột image_phash → bỏ qua grid search pHash')\n",
            "    print(f'   Giữ mặc định: PHASH_THRESHOLD={PHASH_THRESHOLD}, PHASH_BOOST={PHASH_BOOST}')\n",
        ]
    }

    # Insert sau cell-step4-gridsearch
    gs_idx = None
    for i, cell in enumerate(nb['cells']):
        if cell.get('id', '') == 'cell-step4-gridsearch':
            gs_idx = i
            break

    if gs_idx is not None:
        nb['cells'].insert(gs_idx + 1, PHASH_GRID_CELL)
        print(f'\n[SUA 5] pHash Grid Search cell inserted at index {gs_idx + 1}')
        changes += 1
    else:
        print('\n[SUA 5] WARNING: cell-step4-gridsearch not found for pHash grid!')
else:
    print('\n[SUA 5] cell-step4b-phash-grid already exists — skipping')

# ================================================================
# SUA 6: cell-step2-extract — Dùng mean pooling thay pooler_output
#         để lấy nhiều thông tin hơn từ SigLIP
# ================================================================
for cell in nb['cells']:
    cid = cell.get('id', '')
    src = get_src(cell)

    if cid == 'cell-step2-extract':
        old = (
            '            # Lấy pooler_output đại diện cho ảnh\n'
            '            img_features = output.pooler_output\n'
        )
        new = (
            '            # Lấy mean của last_hidden_state (mean pooling) để có thông tin phong phú hơn\n'
            '            # pooler_output chỉ là 1 token [CLS], mean pooling lấy cả sequence\n'
            '            img_features = output.last_hidden_state.mean(dim=1)  # (B, hidden_size)\n'
        )
        if old in src:
            src = src.replace(old, new, 1)
            set_src(cell, src)
            print('\n[SUA 6] SigLIP: pooler_output -> mean_pooling(last_hidden_state)')
            changes += 1
        else:
            print('\n[SUA 6] WARNING: pooler_output extraction block not found')
        break

print(f'\n{"="*60}')
print(f'Tổng số thay đổi: {changes}')
print(f'{"="*60}')

# Lưu notebook
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('\n✅ Notebook đã được lưu thành công!')
print(f'   File: {NB_PATH}')
print('\n📋 Tóm tắt thay đổi:')
print('   1. TOP_RERANK: 50 -> 100 (rerank pool lớn hơn)')
print('   2. PHASH_BOOST: 0.3 -> 0.15 (boost nhẹ hơn, ít ảnh hưởng ranking)')
print('   3. TF-IDF: thêm L2-normalize trước khi lưu cache (fix bug quan trọng!)')
print('   4. Grid Search α: 0.50-0.90 -> 0.30-0.95 (tìm alpha tối ưu hơn)')
print('   5. Thêm pHash Grid Search cell (tìm threshold+boost tốt nhất trên val set)')
print('   6. Thêm Average Query Expansion (AQE) cell (cải tiến mạnh nhất!)')
print('   7. SigLIP: dùng mean pooling thay pooler_output (feature phong phú hơn)')
