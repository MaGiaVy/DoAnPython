import json, sys
sys.stdout.reconfigure(encoding='utf-8')
nb_path = r'd:\New folder (3)\project\notebooks\Nhom3thangcuti_Tuan4\notebooks\Tuan4_GiaVy_Dinov3+miniML.ipynb'
with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']
# Print cells 12, 13, 16 fully
for i in [12, 13, 16]:
    src = ''.join(cells[i]['source'])
    ctype = cells[i]['cell_type']
    print(f'=== CELL {i} ({ctype}) ===')
    print(src)
    print()
