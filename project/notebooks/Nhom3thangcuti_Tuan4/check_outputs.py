import json

for fn in ['notebooks/Tuan4_GiaVy_CLIP.ipynb', 'notebooks/Tuan4_GiaVy_DINOv2.ipynb', 'notebooks/Tuan4_GiaVy_SigLIP_DINOv2.ipynb']:
    with open(fn, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    cell = nb['cells'][7]
    print(f"=== {fn} Cell 7 outputs ===")
    print(f"Execution count: {cell.get('execution_count')}")
    outputs = cell.get('outputs', [])
    for out in outputs:
        if out.get('output_type') == 'stream':
            print(''.join(out.get('text', [])))
        elif out.get('output_type') == 'error':
            print(out.get('ename'), ':', out.get('evalue'))
    print("-" * 50)
