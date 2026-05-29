import json
import sys

nb_path = r'd:\New folder (3)\project\notebooks\Nhom3thangcuti_Tuan4\notebooks\Tuan4_GiaVy_Test.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
output_lines = []
output_lines.append(f'Total cells: {len(cells)}\n\n')
for i, cell in enumerate(cells):
    src = ''.join(cell['source'])
    ctype = cell['cell_type']
    output_lines.append(f'=== Cell {i} [{ctype}] ===\n')
    output_lines.append(src + '\n\n')

with open(r'd:\New folder (3)\project\notebooks\Nhom3thangcuti_Tuan4\read_nb_full.txt', 'w', encoding='utf-8') as f:
    f.writelines(output_lines)

print('Done - Full content written')
