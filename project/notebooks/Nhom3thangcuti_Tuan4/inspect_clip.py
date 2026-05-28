import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
with open('notebooks/Tuan4_GiaVy_CLIP.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb.get('cells', [])):
    src = ''.join(cell.get('source', []))
    print(f"Cell {i} ({cell.get('cell_type')}): {repr(src[:80])}...")
