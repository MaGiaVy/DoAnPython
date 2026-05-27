# -*- coding: utf-8 -*-
"""
fix_meanpool.py - Revert mean_pooling -> pooler_output for SigLIP
SigLIP duoc train voi CLS token (pooler_output), khong phai mean pooling.
Mean pooling cua last_hidden_state khong phu hop voi SigLIP architecture.
"""
import json, sys

NB_PATH = 'Tuan4_GiaVy_DINOv2.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def get_src(cell):
    return ''.join(cell['source'])

def set_src(cell, new_str):
    cell['source'] = new_str.splitlines(keepends=True)

changes = 0

for cell in nb['cells']:
    cid = cell.get('id', '')
    src = get_src(cell)

    # ============================================================
    # FIX 1: Revert mean_pooling -> pooler_output in extract cell
    # ============================================================
    if cid == 'cell-step2-extract':
        WRONG = (
            '            # Dung mean pooling cua last_hidden_state (phong phu hon pooler_output)\n'
            '            # pooler_output = 1 token [CLS]; mean pooling = trung binh ca sequence\n'
            '            img_features = output.last_hidden_state.mean(dim=1)  # (B, hidden_size)\n'
        )
        CORRECT = (
            '            # Dung pooler_output (CLS token) - dung voi SigLIP architecture\n'
            '            # SigLIP duoc train voi CLS-based contrastive loss -> pooler_output la feature tot nhat\n'
            '            img_features = output.pooler_output  # (B, hidden_size)\n'
        )
        if WRONG in src:
            src = src.replace(WRONG, CORRECT, 1)
            set_src(cell, src)
            sys.stdout.buffer.write(b'[FIX 1] Reverted: mean_pooling -> pooler_output (CLS)\n')
            changes += 1
        elif 'output.pooler_output' in src:
            sys.stdout.buffer.write(b'[FIX 1] pooler_output already correct - OK\n')
        else:
            sys.stdout.buffer.write(b'[FIX 1] WARNING: extraction line not found!\n')

    # ============================================================
    # FIX 2: Tat AQE mac dinh vi features cu (cache) con dung mean_pool
    # User phai xoa cache va chay lai tu buoc 2 thi AQE moi hieu qua
    # ============================================================
    if cid == 'cell-step5-aqe':
        if 'DO_AQE = True' in src:
            src = src.replace('DO_AQE = True', 'DO_AQE = True  # Xem ghi chu ben duoi')
            # Them warning comment
            OLD_COMMENT = 'AQE_K = 3   # So ket qua top de mo rong query\n'
            NEW_COMMENT = (
                'AQE_K = 3   # So ket qua top de mo rong query\n'
                '# LUU Y: Neu vua doi mean_pooling -> pooler_output, phai xoa cache\n'
                '# siglip_features.npy va chay lai tu Buoc 2 de AQE hieu qua!\n'
            )
            src = src.replace(OLD_COMMENT, NEW_COMMENT, 1)
            set_src(cell, src)
            sys.stdout.buffer.write(b'[FIX 2] Added AQE cache warning note\n')
            changes += 1

sys.stdout.buffer.write(f'\nTotal fixes: {changes}\n'.encode('ascii'))

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

sys.stdout.buffer.write(b'Notebook saved OK.\n')

# Verify
nb2 = json.load(open(NB_PATH, 'r', encoding='utf-8'))
for cell in nb2['cells']:
    if cell.get('id','') == 'cell-step2-extract':
        src2 = ''.join(cell['source'])
        has_pool = 'output.pooler_output' in src2
        has_mean = 'mean(dim=1)' in src2
        sys.stdout.buffer.write(f'Verify: pooler_output={has_pool}, mean_pooling={has_mean}\n'.encode('ascii'))
        if has_pool and not has_mean:
            sys.stdout.buffer.write(b'[OK] SigLIP extraction is now using pooler_output correctly!\n')
        break
