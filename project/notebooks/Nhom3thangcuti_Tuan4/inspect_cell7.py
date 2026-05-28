import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

for fn in ['notebooks/Tuan4_GiaVy_CLIP.ipynb', 'notebooks/Tuan4_GiaVy_DINOv2.ipynb', 'notebooks/Tuan4_GiaVy_SigLIP_DINOv2.ipynb']:
    with open(fn, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print(f"=== {fn} Cell 7 ===")
    print(''.join(nb['cells'][7].get('source', [])))
    print("-" * 50)
