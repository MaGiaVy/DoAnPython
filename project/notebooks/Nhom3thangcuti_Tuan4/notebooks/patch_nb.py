# -*- coding: utf-8 -*-
import json

NB_PATH = 'Tuan4_GiaVy_DINOv2.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def get_src(cell):
    return ''.join(cell['source'])

def set_src(cell, new_str):
    # Store back as a list of lines (each ending with \n except last)
    lines = new_str.splitlines(keepends=True)
    cell['source'] = lines

changes = 0

for cell in nb['cells']:
    cid = cell.get('id', '')
    src = get_src(cell)

    # -------------------------------------------------------
    # SUA 1: compute_map_at_k — fix n_relevant (remove max)
    # -------------------------------------------------------
    if cid == 'cell-helpers' and 'compute_map_at_k' in src:
        OLD = (
            '        # S\u1ed1 relevant items trong gallery (tr\u1eeb ch\u00ednh query)\n'
            '        n_relevant = max(label_counts.get(q_label, 1) - 1, 1)\n'
        )
        NEW = (
            '        # S\u1ed1 relevant items trong gallery (tr\u1eeb ch\u00ednh query)\n'
            '        n_relevant = label_counts.get(q_label, 0) - 1\n'
            '        # B\u1ecf qua query kh\u00f4ng c\u00f3 \u1ea3nh li\u00ean quan trong gallery\n'
            '        # (kh\u00f4ng d\u00f9ng max(...,1) v\u00ec s\u1ebd l\u00e0m sai metric)\n'
            '        if n_relevant <= 0:\n'
            '            continue\n'
        )
        if OLD in src:
            src = src.replace(OLD, NEW, 1)
            print('[SUA 1] compute_map_at_k n_relevant fixed')
            changes += 1
        else:
            print('[SUA 1] WARNING: target not found in compute_map_at_k')

    # -------------------------------------------------------
    # SUA 2: compute_recall_at_k — fix n_relevant (remove max)
    # -------------------------------------------------------
    if cid == 'cell-helpers' and 'compute_recall_at_k' in src:
        OLD = (
            '        n_relevant = max(label_counts.get(q_label, 1) - 1, 1)\n'
            '\n'
            '        hits = 0\n'
        )
        NEW = (
            '        n_relevant = label_counts.get(q_label, 0) - 1\n'
            '        if n_relevant <= 0:\n'
            '            continue\n'
            '\n'
            '        hits = 0\n'
        )
        if OLD in src:
            src = src.replace(OLD, NEW, 1)
            print('[SUA 2] compute_recall_at_k n_relevant fixed')
            changes += 1
        else:
            print('[SUA 2] WARNING: target not found in compute_recall_at_k')

    if cid == 'cell-helpers':
        set_src(cell, src)

    # -------------------------------------------------------
    # SUA 3: PHASH_THRESHOLD / PHASH_BOOST defaults
    # -------------------------------------------------------
    if cid == 'cell-config':
        OLD = (
            '# --- Tham s\u1ed1 pHash Boosting ---\n'
            'PHASH_THRESHOLD = 0    # Ng\u01b0\u1ee1ng kho\u1ea3ng c\u00e1ch Hamming (0 = gi\u1ed1ng h\u1ec7t, 2 = g\u1ea7n gi\u1ed1ng)\n'
            'PHASH_BOOST     = 1.0  # Gi\u00e1 tr\u1ecb c\u1ed9ng th\u00eam v\u00e0o \u0111i\u1ec3m s\u1ed1 khi pHash g\u1ea7n gi\u1ed1ng\n'
        )
        NEW = (
            '# --- Tham s\u1ed1 pHash Boosting ---\n'
            'PHASH_THRESHOLD = 5    # Boost khi Hamming distance <= 5 (\u0111\u1eb7c th\u00f9 Shopee nhi\u1ec1u seller copy \u1ea3nh)\n'
            'PHASH_BOOST     = 0.3  # Boost nh\u1eb9 \u0111\u1ec3 kh\u00f4ng l\u1ea5n \u00e1t fusion score\n'
        )
        if OLD in src:
            src = src.replace(OLD, NEW, 1)
            set_src(cell, src)
            print('[SUA 3] PHASH_THRESHOLD/BOOST defaults updated')
            changes += 1
        else:
            print('[SUA 3] WARNING: target not found in cell-config')

    # -------------------------------------------------------
    # SUA 4: Add fairness note next to BASELINE_MAP5
    # -------------------------------------------------------
    if cid == 'cell-step5-metrics':
        OLD = (
            'BASELINE_MAP5    = 0.7635\n'
            'BASELINE_PREC1   = None  # Kh\u00f4ng c\u00f3 baseline cho P@1\n'
        )
        NEW = (
            'BASELINE_MAP5    = 0.7635\n'
            '# L\u01afU \u00dd: Baseline tu\u1ea7n 3 \u0111\u00e1nh gi\u00e1 tr\u00ean TO\u00c0N B\u1ed8 34,250 \u1ea3nh l\u00e0m query\n'
            '# Tu\u1ea7n 4 ch\u1ec9 \u0111\u00e1nh gi\u00e1 tr\u00ean TEST SET (80%) \u2192 Kh\u00f4ng so s\u00e1nh tr\u1ef1c ti\u1ebfp \u0111\u01b0\u1ee3c\n'
            'print("\u26a0\ufe0f  L\u01afU \u00dd: Baseline T3 d\u00f9ng to\u00e0n b\u1ed9 query, T4 ch\u1ec9 d\u00f9ng Test Set (80%)")\n'
            'print("   \u2192 \u0110\u1ec3 so s\u00e1nh c\u00f4ng b\u1eb1ng, c\u1ea7n ch\u1ea1y ResNet50 T3 tr\u00ean c\u00f9ng Test Set n\u00e0y")\n'
            'BASELINE_PREC1   = None  # Kh\u00f4ng c\u00f3 baseline cho P@1\n'
        )
        if OLD in src:
            src = src.replace(OLD, NEW, 1)
            set_src(cell, src)
            print('[SUA 4] BASELINE fairness note added')
            changes += 1
        else:
            print('[SUA 4] WARNING: target not found in cell-step5-metrics')

print(f'\nTotal changes: {changes}/4')
if changes < 4:
    print('Some patches failed — check warnings above.')

# -------------------------------------------------------
# SUA 5: Insert new cell-step4b after cell-step4-gridsearch
# -------------------------------------------------------
NEW_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "id": "cell-step4b-phash-grid",
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================\n",
        "# \ud83d\udd0d B\u01af\u1edcC 4B: GRID SEARCH PHASH_THRESHOLD TR\u00caN VALIDATION SET\n",
        "# ============================================================\n",
        "print('=' * 60)\n",
        "print('B\u01af\u1edcC 4B: GRID SEARCH PHASH_THRESHOLD (VAL SET)')\n",
        "print('=' * 60)\n",
        "\n",
        "# D\u00f9ng BEST_ALPHA \u0111\u00e3 t\u00ecm \u0111\u01b0\u1ee3c t\u1eeb B\u01b0\u1edbc 4A\n",
        "thresholds = [0, 3, 5, 8, 10, 15]\n",
        "boost_vals = [0.2, 0.3, 0.5]\n",
        "\n",
        "best_threshold     = PHASH_THRESHOLD  # Gi\u1eef m\u1eb7c \u0111\u1ecbnh n\u1ebfu kh\u00f4ng c\u00f3 pHash\n",
        "best_boost         = PHASH_BOOST\n",
        "best_val_map_phash = -1.0\n",
        "\n",
        "# Ti\u1ec1n t\u00ednh pHash arrays cho val set (n\u1ebfu ch\u01b0a c\u00f3)\n",
        "HAS_PHASH_COL_4B = 'image_phash' in candidate_df.columns\n",
        "\n",
        "if HAS_PHASH_COL_4B:\n",
        "    # T\u1ea3i ho\u1eb7c t\u1ea1o pHash arrays\n",
        "    phash_cache_path_4b = os.path.join(PROCESSED_DIR, 'phash_arrays.npy')\n",
        "    if os.path.exists(phash_cache_path_4b):\n",
        "        gallery_phash_arrays_4b = np.load(phash_cache_path_4b)\n",
        "        print(f'\u2705 T\u1ea3i pHash cache! Shape: {gallery_phash_arrays_4b.shape}')\n",
        "    else:\n",
        "        phash_list_4b = [\n",
        "            hex_to_phash_array(h)\n",
        "            for h in tqdm(candidate_df['image_phash'], desc='   Chuy\u1ec3n pHash')\n",
        "        ]\n",
        "        gallery_phash_arrays_4b = np.array(phash_list_4b, dtype=bool)\n",
        "        np.save(phash_cache_path_4b, gallery_phash_arrays_4b)\n",
        "        print(f'\u2705 T\u1ea1o pHash arrays! Shape: {gallery_phash_arrays_4b.shape}')\n",
        "\n",
        "    print(f'\\n\u0110ang th\u1eed {len(thresholds)} threshold x {len(boost_vals)} boost values...')\n",
        "    print(f'   S\u1eed d\u1ee5ng BEST_ALPHA = {best_alpha:.2f} t\u1eeb B\u01b0\u1edbc 4A')\n",
        "\n",
        "    # X\u00e2y FAISS index m\u1ed9t l\u1ea7n v\u1edbi best_alpha\n",
        "    best_beta_4b = round(1.0 - best_alpha, 4)\n",
        "    gallery_fused_4b = l2_normalize(\n",
        "        np.concatenate([\n",
        "            best_alpha  * gallery_img_norm,\n",
        "            best_beta_4b * gallery_txt_norm\n",
        "        ], axis=1)\n",
        "    )\n",
        "    val_index_4b = build_faiss_index(gallery_fused_4b, USE_GPU_FAISS, gpu_resources)\n",
        "\n",
        "    # Query features val set\n",
        "    val_fused_4b = l2_normalize(\n",
        "        np.concatenate([\n",
        "            best_alpha  * gallery_img_norm[val_gallery_indices],\n",
        "            best_beta_4b * gallery_txt_norm[val_gallery_indices]\n",
        "        ], axis=1)\n",
        "    )\n",
        "\n",
        "    # FAISS search tr\u01b0\u1edbc \u2014 d\u00f9ng l\u1ea1i cho m\u1ecdi threshold\n",
        "    val_top_scores_4b, val_top_raw_4b = val_index_4b.search(\n",
        "        val_fused_4b.astype(np.float32), TOP_RERANK + 5\n",
        "    )\n",
        "    gallery_pids_4b = candidate_df['posting_id'].values\n",
        "    n_val_4b = len(val_query_df)\n",
        "    val_query_reset_4b = val_query_df.reset_index(drop=True)\n",
        "\n",
        "    print()\n",
        "    for thresh in thresholds:\n",
        "        for boost in boost_vals:\n",
        "            val_reranked_4b = np.full((n_val_4b, TOP_K), -1, dtype=np.int64)\n",
        "\n",
        "            for q_idx in range(n_val_4b):\n",
        "                q_pid = val_query_reset_4b.at[q_idx, 'posting_id']\n",
        "                cands_gidx, cands_scores = [], []\n",
        "\n",
        "                for gidx, score in zip(val_top_raw_4b[q_idx], val_top_scores_4b[q_idx]):\n",
        "                    if gidx < 0: break\n",
        "                    if gallery_pids_4b[gidx] == q_pid: continue\n",
        "                    cands_gidx.append(gidx)\n",
        "                    cands_scores.append(float(score))\n",
        "                    if len(cands_gidx) == TOP_RERANK: break\n",
        "\n",
        "                if not cands_gidx: continue\n",
        "                cands_gidx   = np.array(cands_gidx, dtype=np.int64)\n",
        "                cands_scores = np.array(cands_scores, dtype=np.float32)\n",
        "\n",
        "                # pHash boost\n",
        "                q_gidx  = val_gallery_indices[q_idx]\n",
        "                q_phash = gallery_phash_arrays_4b[q_gidx]\n",
        "                ham     = compute_hamming_distances(q_phash, gallery_phash_arrays_4b[cands_gidx])\n",
        "                cands_scores[ham <= thresh] += boost\n",
        "\n",
        "                order = np.argsort(-cands_scores)[:TOP_K]\n",
        "                val_reranked_4b[q_idx] = cands_gidx[order]\n",
        "\n",
        "            val_map_4b = compute_map_at_k(val_query_df, candidate_df, val_reranked_4b, k=TOP_K)\n",
        "\n",
        "            is_best_4b = val_map_4b > best_val_map_phash\n",
        "            if is_best_4b:\n",
        "                best_val_map_phash = val_map_4b\n",
        "                best_threshold     = thresh\n",
        "                best_boost         = boost\n",
        "\n",
        "            marker_4b = ' \u25c4 T\u1ed0T NH\u1ea4T!' if is_best_4b else ''\n",
        "            print(f'   thresh={thresh:2d}, boost={boost:.1f} \u2192 val mAP@5={val_map_4b:.4f}{marker_4b}')\n",
        "\n",
        "    print(f'\\n\ud83c\udfc6 BEST pHash params: threshold={best_threshold}, boost={best_boost}')\n",
        "    print(f'   val mAP@5 = {best_val_map_phash:.4f}')\n",
        "\n",
        "    # C\u1eadp nh\u1eadt tham s\u1ed1 t\u1ed1i \u01b0u \u0111\u1ec3 d\u00f9ng \u1edf B\u01b0\u1edbc 5\n",
        "    PHASH_THRESHOLD = best_threshold\n",
        "    PHASH_BOOST     = best_boost\n",
        "    print(f'\\n\u2705 \u0110\u00e3 c\u1eadp nh\u1eadt: PHASH_THRESHOLD={PHASH_THRESHOLD}, PHASH_BOOST={PHASH_BOOST}')\n",
        "\n",
        "    # D\u1ecdn b\u1ed9 nh\u1edb\n",
        "    del gallery_fused_4b, val_fused_4b, val_index_4b\n",
        "    if torch.cuda.is_available():\n",
        "        torch.cuda.empty_cache()\n",
        "\n",
        "else:\n",
        "    print('\u26a0\ufe0f  Kh\u00f4ng c\u00f3 c\u1ed9t image_phash \u2192 b\u1ecf qua grid search threshold')\n",
        "    print(f'   Gi\u1eef m\u1eb7c \u0111\u1ecbnh: PHASH_THRESHOLD={PHASH_THRESHOLD}, PHASH_BOOST={PHASH_BOOST}')\n"
    ]
}

# Find the index of cell-step4-gridsearch and insert new cell after it
insert_after_idx = None
for i, cell in enumerate(nb['cells']):
    if cell.get('id', '') == 'cell-step4-gridsearch':
        insert_after_idx = i
        break

if insert_after_idx is not None:
    nb['cells'].insert(insert_after_idx + 1, NEW_CELL)
    print(f'\n[SUA 5] New cell-step4b inserted after index {insert_after_idx}')
    changes += 1
else:
    print('\n[SUA 5] WARNING: cell-step4-gridsearch not found!')

print(f'\nFinal total changes: {changes}/5')

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Notebook saved OK.')
