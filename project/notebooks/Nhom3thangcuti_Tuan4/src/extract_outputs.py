import json
import sys

nb_path = sys.argv[1]
out_path = sys.argv[2] if len(sys.argv) > 2 else 'nb_outputs.txt'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open(out_path, 'w', encoding='utf-8') as out:
    for i, cell in enumerate(nb['cells']):
        outputs = cell.get('outputs', [])
        if outputs:
            out.write(f'=== CELL {i} OUTPUTS ===\n')
            for j, output in enumerate(outputs):
                otype = output.get('output_type', '')
                if otype in ('stream', 'execute_result', 'display_data'):
                    text = output.get('text', output.get('data', {}).get('text/plain', []))
                    if isinstance(text, list):
                        text = ''.join(text)
                    # Only print first 2000 chars of output
                    out.write(text[:2000] + '\n')
                    if len(text) > 2000:
                        out.write(f'... [OUTPUT TRUNCATED, total {len(text)} chars]\n')
                elif otype == 'error':
                    out.write(f'ERROR: {output.get("ename","")}: {output.get("evalue","")}\n')
            out.write('\n')

print(f'Done. Output written to {out_path}')
