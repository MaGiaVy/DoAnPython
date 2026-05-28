import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
with open('notebooks/Tuan4_GiaVy_DINOv2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb.get('cells', [])):
    src = ''.join(cell.get('source', []))
    if 'val_gallery_indices' in src or 'test_gallery_indices' in src:
        print(f"=== CELL {i} ===")
        print(src)
        print("="*60)
