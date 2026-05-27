# -*- coding: utf-8 -*-
"""
add_aqe_cell.py  -  Insert AQE cell into DINOv2 notebook
"""
import json

NB_PATH = 'Tuan4_GiaVy_DINOv2.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Check AQE already exists
ids = [c.get('id','') for c in nb['cells']]
if 'cell-step5-aqe' in ids:
    print('AQE cell already present - skipping')
else:
    AQE_CELL = {
        "cell_type": "code",
        "execution_count": None,
        "id": "cell-step5-aqe",
        "metadata": {},
        "outputs": [],
        "source": [
            "# ============================================================\n",
            "# BUOC 5.4b: AVERAGE QUERY EXPANSION (AQE)\n",
            "# ============================================================\n",
            "# AQE: Mo rong query bang cach lay trung binh cua query + top-3 results\n",
            "# Giup tang recall dang ke ma khong can them du lieu\n",
            "print('=' * 60)\n",
            "print('BUOC 5.4b: AVERAGE QUERY EXPANSION (AQE)')\n",
            "print('=' * 60)\n",
            "\n",
            "AQE_K = 3   # So ket qua top de mo rong query\n",
            "DO_AQE = True  # Dat False de tat AQE neu muon so sanh\n",
            "\n",
            "if DO_AQE:\n",
            "    print(f'\\n[AQE] Dang ap dung AQE voi top-{AQE_K} results...')\n",
            "    print(f'   AQE: q_expanded = L2_Norm(q + mean(top_{AQE_K}_gallery_vecs))')\n",
            "\n",
            "    gallery_pids_aqe = candidate_df['posting_id'].values\n",
            "    n_test_aqe = len(test_query_df)\n",
            "    test_fused_expanded = test_fused.copy()  # (n_test, fused_dim)\n",
            "\n",
            "    expand_count = 0\n",
            "    qreset = test_query_df.reset_index(drop=True)\n",
            "\n",
            "    for q_idx in range(n_test_aqe):\n",
            "        q_pid = qreset.at[q_idx, 'posting_id']\n",
            "\n",
            "        # Thu thap top-AQE_K valid results (bo self-match)\n",
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
            "        # Lay fused vectors cua neighbors tu gallery\n",
            "        neighbor_vecs = final_gallery_fused[neighbors]  # (AQE_K, fused_dim)\n",
            "\n",
            "        # AQE: query moi = L2_Norm(query + mean(neighbors))\n",
            "        q_expanded = test_fused[q_idx] + neighbor_vecs.mean(axis=0)\n",
            "        q_norm = np.linalg.norm(q_expanded)\n",
            "        if q_norm > 1e-10:\n",
            "            test_fused_expanded[q_idx] = (q_expanded / q_norm).astype(np.float32)\n",
            "        expand_count += 1\n",
            "\n",
            "    print(f'[AQE] Hoan tat! Da mo rong {expand_count:,}/{n_test_aqe:,} queries')\n",
            "\n",
            "    # Search lai voi expanded queries\n",
            "    print(f'\\n[AQE] Dang search lai voi expanded queries...')\n",
            "    top_scores_raw, top_indices_raw = final_index.search(\n",
            "        test_fused_expanded.astype(np.float32),\n",
            "        TOP_RERANK + 5\n",
            "    )\n",
            "    print(f'[AQE] Search hoan tat! top_indices_raw updated.')\n",
            "else:\n",
            "    print('[AQE] AQE bi tat (DO_AQE=False) - dung ket qua search goc')\n",
        ]
    }

    # Find cell-step5-rerank and insert before it
    insert_before_idx = None
    for i, cell in enumerate(nb['cells']):
        if cell.get('id', '') == 'cell-step5-rerank':
            insert_before_idx = i
            break

    if insert_before_idx is not None:
        nb['cells'].insert(insert_before_idx, AQE_CELL)
        print(f'[OK] AQE cell inserted before cell-step5-rerank (at index {insert_before_idx})')
    else:
        print('[ERROR] cell-step5-rerank not found!')

# Save
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Notebook saved OK.')

# Verify
nb2 = json.load(open(NB_PATH, 'r', encoding='utf-8'))
ids2 = [c.get('id','') for c in nb2['cells']]
print('Total cells:', len(ids2))
print('AQE present:', 'cell-step5-aqe' in ids2)
print('Order check:', ids2[ids2.index('cell-step5-build-index'):ids2.index('cell-step5-build-index')+4] if 'cell-step5-build-index' in ids2 else 'N/A')
