import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('notebooks/Tuan4_GiaVy_Dinov3+miniML - Copy.ipynb', 'r', encoding='utf-8') as f:
    nb_vy = json.load(f)

with open('notebooks/Tuan4_Hung_Dinov3+miniML.ipynb', 'r', encoding='utf-8') as f:
    nb_hung = json.load(f)

print(f"Vy cells: {len(nb_vy['cells'])}")
print(f"Hung cells: {len(nb_hung['cells'])}")

for i in range(min(len(nb_vy['cells']), len(nb_hung['cells']))):
    c_vy = nb_vy['cells'][i]
    c_hung = nb_hung['cells'][i]
    if c_vy['cell_type'] != c_hung['cell_type']:
        print(f"Cell {i} type mismatch: Vy={c_vy['cell_type']}, Hung={c_hung['cell_type']}")
        continue
    src_vy = ''.join(c_vy.get('source', []))
    src_hung = ''.join(c_hung.get('source', []))
    if src_vy != src_hung:
        print(f"Cell {i} code mismatch:")
        print(f"Vy:\n{src_vy[:150]}\nHung:\n{src_hung[:150]}")
        print("-" * 40)
