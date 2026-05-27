# -*- coding: utf-8 -*-
import json
import re

NB_PATH = 'Tuan4_GiaVy_DINOv2.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell.get('id') == 'cell-config':
        source = cell['source']
        for i, line in enumerate(source):
            if line.startswith("BASE_DIR"):
                source[i] = "BASE_DIR        = '/content/drive/MyDrive/DuLieuPython'\n"
            elif line.startswith("IMAGE_DIR"):
                source[i] = "IMAGE_DIR       = os.path.join(BASE_DIR, 'train_images.zip')\n"
            elif line.startswith("CANDIDATE_CSV"):
                source[i] = "CANDIDATE_CSV   = os.path.join(BASE_DIR, 'train.csv')\n"
        
        cell['source'] = source
        break

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Paths updated successfully.")
