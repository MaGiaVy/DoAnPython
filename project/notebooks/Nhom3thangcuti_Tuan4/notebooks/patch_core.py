# -*- coding: utf-8 -*-
"""
patch_core.py - Patch core settings in DINOv2 notebook (ASCII safe)
"""
import json, sys

NB_PATH = 'Tuan4_GiaVy_DINOv2.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def get_src(cell):
    return ''.join(cell['source'])

def set_src(cell, new_str):
    lines = new_str.splitlines(keepends=True)
    cell['source'] = lines

changes = 0

for cell in nb['cells']:
    cid = cell.get('id', '')
    src = get_src(cell)
    modified = False

    # ============================================================
    # PATCH 1: cell-config - TOP_RERANK + PHASH_BOOST
    # ============================================================
    if cid == 'cell-config':
        # TOP_RERANK: 50 -> 100
        if 'TOP_RERANK     = 50' in src:
            src = src.replace(
                'TOP_RERANK     = 50    # S\u1ed1 \u1ee9ng vi\u00ean l\u1ea5y ra tr\u01b0\u1edbc khi rerank',
                'TOP_RERANK     = 100   # S\u1ed1 \u1ee9ng vi\u00ean l\u1ea5y ra tr\u01b0\u1edbc khi rerank (t\u0103ng t\u1eeb 50->100)',
            )
            sys.stdout.buffer.write(b'[PATCH 1a] TOP_RERANK: 50 -> 100\n')
            changes += 1
            modified = True
        else:
            sys.stdout.buffer.write(b'[PATCH 1a] TOP_RERANK 50 not found - maybe already patched\n')

        # PHASH_BOOST: 0.3 -> 0.15
        if 'PHASH_BOOST     = 0.3 ' in src:
            src = src.replace(
                'PHASH_BOOST     = 0.3  # Boost nh\u1eb9 \u0111\u1ec3 kh\u00f4ng l\u1ea5n \u00e1t fusion score',
                'PHASH_BOOST     = 0.15 # Boost nh\u1eb9 \u0111\u1ec3 kh\u00f4ng l\u1ea5n \u00e1t fusion score (gi\u1ea3m t\u1eeb 0.3->0.15)',
            )
            sys.stdout.buffer.write(b'[PATCH 1b] PHASH_BOOST: 0.3 -> 0.15\n')
            changes += 1
            modified = True
        else:
            sys.stdout.buffer.write(b'[PATCH 1b] PHASH_BOOST 0.3 not found\n')

        if modified:
            set_src(cell, src)

    # ============================================================
    # PATCH 2: cell-step3 - TF-IDF L2-normalize before save
    # ============================================================
    if cid == 'cell-step3':
        modified = False

        # Add L2-normalize before np.save
        TARGET_SAVE = (
            '    # L\u01b0u features\n'
            '    np.save(tfidf_save_path, tfidf_features)\n'
            '    file_size_mb = os.path.getsize(tfidf_save_path)'
        )
        REPLACEMENT_SAVE = (
            '    # L2-normalize TF-IDF features truoc khi luu (fix bug quan trong!)\n'
            '    _norms = np.linalg.norm(tfidf_features, axis=1, keepdims=True)\n'
            '    tfidf_features = (tfidf_features / np.maximum(_norms, 1e-10)).astype(np.float32)\n'
            '\n'
            '    # L\u01b0u features\n'
            '    np.save(tfidf_save_path, tfidf_features)\n'
            '    file_size_mb = os.path.getsize(tfidf_save_path)'
        )

        if TARGET_SAVE in src and 'L2-normalize TF-IDF' not in src:
            src = src.replace(TARGET_SAVE, REPLACEMENT_SAVE, 1)
            sys.stdout.buffer.write(b'[PATCH 2a] Added L2-normalize before TF-IDF save\n')
            changes += 1
            modified = True
        elif 'L2-normalize TF-IDF' in src:
            sys.stdout.buffer.write(b'[PATCH 2a] L2-normalize already present\n')
        else:
            sys.stdout.buffer.write(b'[PATCH 2a] WARNING: TF-IDF save block not found\n')

        # Add normalize check when loading from cache
        TARGET_LOAD = (
            '    tfidf_features = np.load(tfidf_save_path)\n'
            '    print(f\'\u2705 \u0110\u00e3 t\u1ea3i xong! Shape: {tfidf_features.shape}\')\n'
            'else:'
        )
        REPLACEMENT_LOAD = (
            '    tfidf_features = np.load(tfidf_save_path)\n'
            '    print(f\'\u2705 \u0110\u00e3 t\u1ea3i xong! Shape: {tfidf_features.shape}\')\n'
            '    # Kiem tra va normalize lai neu cache cu chua normalize\n'
            '    _check_norms = np.linalg.norm(tfidf_features[:100], axis=1)\n'
            '    if not np.allclose(_check_norms, 1.0, atol=0.05):\n'
            '        print(\'   Cache TF-IDF chua normalize, dang chuan hoa...\')\n'
            '        _n = np.linalg.norm(tfidf_features, axis=1, keepdims=True)\n'
            '        tfidf_features = (tfidf_features / np.maximum(_n, 1e-10)).astype(np.float32)\n'
            '        np.save(tfidf_save_path, tfidf_features)\n'
            '        print(\'   Da normalize va luu lai cache!\')\n'
            'else:'
        )

        if TARGET_LOAD in src and 'normalize lai neu cache' not in src:
            src = src.replace(TARGET_LOAD, REPLACEMENT_LOAD, 1)
            sys.stdout.buffer.write(b'[PATCH 2b] Added normalize check on cache load\n')
            changes += 1
            modified = True
        elif 'normalize lai neu cache' in src:
            sys.stdout.buffer.write(b'[PATCH 2b] Normalize check already present\n')
        else:
            sys.stdout.buffer.write(b'[PATCH 2b] WARNING: TF-IDF load block not found\n')

        if modified:
            set_src(cell, src)

    # ============================================================
    # PATCH 3: cell-step4-gridsearch - Expand alpha range
    # ============================================================
    if cid == 'cell-step4-gridsearch':
        modified = False

        if 'arange(0.50, 0.91, 0.05)' in src:
            src = src.replace(
                'alphas = np.arange(0.50, 0.91, 0.05)  # [0.50, 0.55, 0.60, ..., 0.90]',
                'alphas = np.arange(0.30, 0.96, 0.05)  # [0.30, 0.35, ..., 0.95] - mo rong de tim alpha tot nhat',
            )
            sys.stdout.buffer.write(b'[PATCH 3a] Grid Search alpha: 0.50-0.90 -> 0.30-0.95\n')
            changes += 1
            modified = True
        elif 'arange(0.30' in src:
            sys.stdout.buffer.write(b'[PATCH 3a] Alpha range already expanded\n')
        else:
            sys.stdout.buffer.write(b'[PATCH 3a] WARNING: alphas line not found\n')

        if 'n_search = TOP_K + 15' in src:
            src = src.replace(
                'n_search = TOP_K + 15  # L\u1ea5y top-20 \u0111\u1ec3 \u0111\u1ea3m b\u1ea3o c\u00f3 \u0111\u1ee7 sau khi lo\u1ea1i self-match',
                'n_search = TOP_K + 25  # L\u1ea5y top-30 \u0111\u1ec3 \u0111\u1ea3m b\u1ea3o c\u00f3 \u0111\u1ee7 sau khi lo\u1ea1i self-match (t\u0103ng buffer)',
            )
            sys.stdout.buffer.write(b'[PATCH 3b] n_search buffer: TOP_K+15 -> TOP_K+25\n')
            changes += 1
            modified = True
        elif 'n_search = TOP_K + 25' in src:
            sys.stdout.buffer.write(b'[PATCH 3b] n_search buffer already increased\n')

        if modified:
            set_src(cell, src)

    # ============================================================
    # PATCH 4: cell-step2-extract - mean pooling thay pooler_output
    # ============================================================
    if cid == 'cell-step2-extract':
        TARGET_POOL = (
            '            # L\u1ea5y pooler_output \u0111\u1ea1i di\u1ec7n cho \u1ea3nh\n'
            '            img_features = output.pooler_output\n'
        )
        REPLACEMENT_POOL = (
            '            # Dung mean pooling cua last_hidden_state (phong phu hon pooler_output)\n'
            '            # pooler_output = 1 token [CLS]; mean pooling = trung binh ca sequence\n'
            '            img_features = output.last_hidden_state.mean(dim=1)  # (B, hidden_size)\n'
        )

        if TARGET_POOL in src and 'mean(dim=1)' not in src:
            src = src.replace(TARGET_POOL, REPLACEMENT_POOL, 1)
            set_src(cell, src)
            sys.stdout.buffer.write(b'[PATCH 4] SigLIP: pooler_output -> mean_pooling\n')
            changes += 1
        elif 'mean(dim=1)' in src:
            sys.stdout.buffer.write(b'[PATCH 4] mean_pooling already present\n')
        else:
            sys.stdout.buffer.write(b'[PATCH 4] WARNING: pooler_output block not found\n')

sys.stdout.buffer.write(f'\nTotal changes applied: {changes}\n'.encode('ascii'))

# Save
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

sys.stdout.buffer.write(b'Notebook saved OK.\n')
