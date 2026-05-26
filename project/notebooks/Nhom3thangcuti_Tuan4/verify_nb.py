import json, os, sys
sys.stdout.reconfigure(encoding='utf-8')

nb_path = r'd:\New folder (3)\project\notebooks\Nhom3thangcuti_Tuan4\notebooks\Tuan4_GiaVy_DINOv2.ipynb'
file_size = os.path.getsize(nb_path)

with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

print("=" * 60)
print("KIEM TRA NOTEBOOK: Tuan4_GiaVy_DINOv2.ipynb")
print("=" * 60)
print(f"File size   : {file_size:,} bytes ({file_size/1024:.1f} KB)")
print(f"nbformat    : {nb['nbformat']}")
print(f"Tong so cell: {len(nb['cells'])}")
print()

code_count = 0
md_count   = 0
total_lines = 0

for i, cell in enumerate(nb['cells']):
    ctype = cell['cell_type']
    src   = ''.join(cell['source'])
    lines = src.strip().split('\n')
    first = lines[0][:70] if lines else ''
    n_lines = len(lines)
    total_lines += n_lines

    if ctype == 'code':
        code_count += 1
    else:
        md_count += 1

    first_safe = first.encode('ascii', errors='replace').decode('ascii')
    print(f"  [{i:02d}] {ctype:8s} ({n_lines:3d} dong) | {first_safe}")

print()
print(f"Code cells    : {code_count}")
print(f"Markdown cells: {md_count}")
print(f"Tong dong code: ~{total_lines}")
print()

# Kiem tra cac buoc quan trong
steps = {
    'BUOC 1 - Phan chia du lieu':      'train_test_split',
    'BUOC 2 - DINOv2':                 'torch.hub.load',
    'BUOC 3 - TF-IDF':                 'TfidfVectorizer',
    'BUOC 4 - Grid Search':            'np.arange(0.50',
    'BUOC 5 - pHash Boosting':         'PHASH_BOOST',
    'BUOC 5 - Reranking':              'TOP_RERANK',
    'BUOC 5 - Luu final_metrics.csv':  'final_metrics.csv',
    'L2 normalize':                    'l2_normalize',
    'compute_map_at_k':                'compute_map_at_k',
    'compute_precision_at_1':          'compute_precision_at_1',
    'compute_recall_at_k':             'compute_recall_at_k',
    'FAISS IndexFlatIP':               'IndexFlatIP',
    'Tieng Viet (comments)':           'Dang',
}

all_src = '\n'.join(''.join(c['source']) for c in nb['cells'])
print("KIEM TRA NOI DUNG:")
all_ok = True
for desc, keyword in steps.items():
    found = keyword in all_src
    status = "OK" if found else "MISSING"
    if not found:
        all_ok = False
    print(f"  [{status}] {desc}")

print()
if all_ok:
    print(">> TAT CA KIEM TRA THANH CONG! Notebook hop le.")
else:
    print(">> COT MOT SO BUOC CON THIEU. Kiem tra lai!")
