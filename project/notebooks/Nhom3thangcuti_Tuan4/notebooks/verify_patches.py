# -*- coding: utf-8 -*-
import json

NB_PATH = 'Tuan4_GiaVy_DINOv2.ipynb'

# Read with surrogate escape mode to recover the data despite encode errors
with open(NB_PATH, 'r', encoding='utf-8', errors='surrogatepass') as f:
    content = f.read()

# Encode back to bytes with surrogate escape, then decode cleanly (replacing bad chars)
content_bytes = content.encode('utf-8', errors='surrogatepass')
content_clean = content_bytes.decode('utf-8', errors='replace')

# Write cleaned content
with open(NB_PATH, 'w', encoding='utf-8') as f:
    f.write(content_clean)

# Verify it parses OK
with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

n_cells = len(nb['cells'])
print('Notebook OK:', n_cells, 'cells')

# Confirm each patch
for cell in nb['cells']:
    cid = cell.get('id', '')
    src = ''.join(cell['source'])

    if cid == 'cell-helpers':
        cnt = src.count('if n_relevant <= 0:')
        print('SUA1+2 n_relevant fix occurrences:', cnt, '(expect 2)')

    if cid == 'cell-config':
        if 'PHASH_THRESHOLD = 5' in src:
            print('SUA3 confirmed: PHASH_THRESHOLD = 5')
        else:
            print('SUA3 NOT FOUND')

    if cid == 'cell-step5-metrics':
        if 'LU' in src and 'Test Set' in src:
            print('SUA4 confirmed: baseline fairness note present')
        else:
            print('SUA4 NOT FOUND')

    if cid == 'cell-step4b-phash-grid':
        print('SUA5 confirmed: cell-step4b-phash-grid exists')

print('Done.')
