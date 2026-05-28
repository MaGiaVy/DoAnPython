import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open(r'd:\New folder (3)\project\notebooks\Nhom3thangcuti_Tuan4\notebooks\Tuan4_GiaVy_Dinov3+miniML.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']
print(f'Total cells: {len(cells)}')
for i, cell in enumerate(cells):
    src = ''.join(cell['source'])
    ctype = cell['cell_type']
    print(f'\n{"="*60}')
    print(f'Cell {i} ({ctype})')
    print(f'{"="*60}')
    print(src)
