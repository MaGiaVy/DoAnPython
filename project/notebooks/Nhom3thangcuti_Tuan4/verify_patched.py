import json
import ast
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('notebooks/Tuan4_GiaVy_Dinov3+miniML.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
print("Verifying python syntax of all code cells...")

errors = 0
for idx, cell in enumerate(nb['cells']):
    cell_type = cell.get('cell_type')
    if cell_type == 'code':
        source = ''.join(cell.get('source', []))
        cleaned_source = []
        for line in source.split('\n'):
            if line.strip().startswith('!') or line.strip().startswith('%'):
                cleaned_source.append("# " + line)
            else:
                cleaned_source.append(line)
        cleaned_source = '\n'.join(cleaned_source)
        try:
            ast.parse(cleaned_source)
        except SyntaxError as e:
            print(f"Cell {idx} has Syntax Error:\n{e}\nSource:\n{cleaned_source}\n")
            errors += 1

if errors == 0:
    print("✅ All code cells have valid Python syntax!")
else:
    print(f"❌ Found {errors} syntax errors!")
