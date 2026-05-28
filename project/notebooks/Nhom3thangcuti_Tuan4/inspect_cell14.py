import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
with open('notebooks/Tuan4_GiaVy_Dinov3+miniML - Copy.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("=== CELL 14 ===")
print(''.join(nb['cells'][14].get('source', [])))
