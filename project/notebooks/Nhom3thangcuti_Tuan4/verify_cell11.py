import json
nb_path = r'd:\New folder (3)\project\notebooks\Nhom3thangcuti_Tuan4\notebooks\Baseline2_MobileCLIP.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)
for cell in nb['cells']:
    if cell.get('metadata', {}).get('id') == '8dwqb17e8eeg':
        src = cell['source']
        print(f'Cell 11 - so dong: {len(src)}')
        outs = cell['outputs']
        print(f'Outputs count: {len(outs)}')
        for i, line in enumerate(src):
            print(f'  [{i}]', line.encode('ascii', errors='replace').decode('ascii').rstrip())
        break
