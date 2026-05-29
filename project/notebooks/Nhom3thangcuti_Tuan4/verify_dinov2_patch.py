import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

nb_path = r'd:\New folder (3)\project\notebooks\Nhom3thangcuti_Tuan4\notebooks\Tuan4_GiaVy_Test.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
result_lines = []
result_lines.append(f'Total cells after patch: {len(cells)}\n')
for i in range(25, len(cells)):
    src = ''.join(cells[i]['source'])
    ctype = cells[i]['cell_type']
    first_line = src.split('\n')[0][:100]
    result_lines.append(f'  Cell {i} [{ctype}]: {first_line}\n')

with open(r'd:\New folder (3)\project\notebooks\Nhom3thangcuti_Tuan4\verify_result.txt', 'w', encoding='utf-8') as f:
    f.writelines(result_lines)
print('Done')
