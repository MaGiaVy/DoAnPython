import json
import sys
import os

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')

nb_path = sys.argv[1]
out_path = sys.argv[2] if len(sys.argv) > 2 else 'nb_output.txt'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open(out_path, 'w', encoding='utf-8') as out:
    for i, cell in enumerate(nb['cells']):
        cell_type = cell['cell_type']
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        out.write(f'=== CELL {i} ({cell_type}) ===\n')
        out.write(source[:1500] + '\n')
        if len(source) > 1500:
            out.write(f'... [TRUNCATED, total {len(source)} chars]\n')
        out.write('\n')
    out.write(f'\nTOTAL CELLS: {len(nb["cells"])}\n')

print(f'Done. Output written to {out_path}')
