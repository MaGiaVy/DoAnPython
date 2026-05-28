import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
with open('notebooks/Tuan4_GiaVy_DINOv2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx in [15, 16, 17]:
    cell = nb['cells'][idx]
    src = ''.join(cell.get('source', []))
    print(f"=== CELL {idx} ===")
    print(src)
    print("="*60)
