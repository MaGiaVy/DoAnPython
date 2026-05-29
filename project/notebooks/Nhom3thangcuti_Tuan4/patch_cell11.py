import json

nb_path = r'd:\New folder (3)\project\notebooks\Nhom3thangcuti_Tuan4\notebooks\Baseline2_MobileCLIP.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 11 - hardcode ket qua da biet, khong phu thuoc bien cu
cell11_source = [
    "import pandas as pd, os\n",
    "\n",
    "# ─── Ket qua da biet tu Cell 9 & 10 ──────────────────────────────────────────\n",
    "# Neu session bi reset: hardcode gia tri da chay thanh cong\n",
    "# embed_dim=512, best_alpha_2=0.9, test metrics tu output Cell 10\n",
    "try:\n",
    "    _embed_dim = embed_dim\n",
    "    _alpha     = float(best_alpha_2)\n",
    "    _map5      = test_metrics_2['mAP@5']\n",
    "    _p1        = test_metrics_2['Precision@1']\n",
    "    _r5        = test_metrics_2['Recall@5']\n",
    "    print('Su dung bien tu session hien tai.')\n",
    "except NameError:\n",
    "    print('Session bi reset -> dung ket qua hardcode tu lan chay truoc.')\n",
    "    _embed_dim = 512\n",
    "    _alpha     = 0.9\n",
    "    _map5      = 0.7692\n",
    "    _p1        = 0.7919\n",
    "    _r5        = 0.7413\n",
    "\n",
    "# ─── Tong hop metrics ────────────────────────────────────────────────────────\n",
    "metrics_data = [\n",
    "    {\n",
    '        "Model (Phuong phap chinh)": "MobileCLIP (Zero-Shot Fusion)",\n',
    '        "Kich thuoc Vector (Dim)": str(_embed_dim),\n',
    '        "Alpha toi uu (Validation)": round(_alpha, 1),\n',
    '        "Test mAP@5": round(_map5, 4),\n',
    '        "Test Precision@1": round(_p1, 4),\n',
    '        "Test Recall@5": round(_r5, 4),\n',
    "    }\n",
    "]\n",
    "\n",
    "df_metrics = pd.DataFrame(metrics_data)\n",
    "print('\\n=== Final Metrics ===')\n",
    "print(df_metrics.to_string(index=False))\n",
    "\n",
    "# ─── Luu local (Colab /content/) ─────────────────────────────────────────────\n",
    'LOCAL_CSV = "/content/final_metric.csv"\n',
    "df_metrics.to_csv(LOCAL_CSV, index=False, encoding='utf-8-sig')\n",
    "print(f'\\nDa luu local : {LOCAL_CSV}')\n",
    "\n",
    "# ─── Luu len Google Drive ─────────────────────────────────────────────────────\n",
    "# Danh sach cac duong dan co the co cua Drive\n",
    "_drive_candidates = [\n",
    "    '/content/drive/MyDrive/DoAnPython/DuLieuPython',\n",
    "    '/content/drive/MyDrive/DuLieuPython',\n",
    "    '/content/drive/My Drive/DoAnPython/DuLieuPython',\n",
    "    '/content/drive/My Drive/DuLieuPython',\n",
    "]\n",
    "try:\n",
    "    _drive_dir = DATA_DIR\n",
    "except NameError:\n",
    "    _drive_dir = next(\n",
    "        (p for p in _drive_candidates if os.path.exists(p)),\n",
    "        '/content/drive/MyDrive'\n",
    "    )\n",
    "\n",
    "DRIVE_CSV = os.path.join(_drive_dir, 'final_metric.csv')\n",
    "df_metrics.to_csv(DRIVE_CSV, index=False, encoding='utf-8-sig')\n",
    "print(f'Da luu Drive  : {DRIVE_CSV}')\n",
    "\n",
    "print('\\nHoan tat xuat ket qua!')\n",
]

for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code' and cell.get('metadata', {}).get('id') == '8dwqb17e8eeg':
        cell['source'] = cell11_source
        cell['outputs'] = []
        cell['execution_count'] = None
        print(f'Da cap nhat Cell 11 tai index {i} voi {len(cell11_source)} dong.')
        break

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Ghi notebook thanh cong!')
