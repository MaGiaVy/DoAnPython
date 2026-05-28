import json, sys
sys.stdout.reconfigure(encoding='utf-8')
nb_path = r'd:\New folder (3)\project\notebooks\Nhom3thangcuti_Tuan4\notebooks\Tuan4_GiaVy_Dinov3+miniML.ipynb'
with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']
print(f'Total cells: {len(cells)}')
print()

# Kiểm tra từng sửa đổi
checks = [
    ("SỬA 1a - Cell 13 có multilingual", 13, "paraphrase-multilingual-MiniLM-L12-v2"),
    ("SỬA 1a - Cell 13 KHÔNG còn all-MiniLM-L6", 13, "all-MiniLM-L6-v2"),
    ("SỬA 1b - Cell 16 có multilingual", 16, "paraphrase-multilingual-MiniLM-L12-v2"),
    ("SỬA 1b - Cell 16 KHÔNG còn all-MiniLM-L6", 16, "all-MiniLM-L6-v2"),
    ("SỬA 2b - Cell 16 lưu multilingual_minilm_text_features.npy", 16, "multilingual_minilm_text_features.npy"),
    ("SỬA 2b - Cell 16 KHÔNG còn minilm_text_features.npy (cũ)", 16, "minilm_text_features.npy"),
    ("SỬA 2 - Cell 12 dùng valid_indices", 12, "valid_indices"),
    ("SỬA 2 - Cell 12 lưu split_indices.json", 12, "split_indices.json"),
    ("SỴA 3 - Cell 18 load multilingual_minilm_text_features.npy", 18, "multilingual_minilm_text_features.npy"),
    ("SỬA 3 - Cell 18 có best_alpha", 18, "best_alpha"),
    ("SỬA 4 - Cell 20 load multilingual_minilm_text_features.npy", 20, "multilingual_minilm_text_features.npy"),
    ("SỬA 4 - Cell 20 dùng test_idx", 20, "test_idx"),
    ("SỬA 4 - Cell 20 gt_len==0 → continue", 20, "if gt_len == 0:"),
    ("SỬA 4 - Cell 20 lưu metrics_dinov3_minilm.csv", 20, "metrics_dinov3_minilm.csv"),
]

for label, cell_idx, keyword in checks:
    src = ''.join(cells[cell_idx]['source'])
    found = keyword in src
    # For negative checks (KHÔNG còn)
    if "KHÔNG còn" in label:
        status = "❌ FAIL" if found else "✅ OK"
    else:
        status = "✅ OK" if found else "❌ FAIL"
    print(f"{status} | {label}")

# Kiểm tra cell cuối
print()
if len(cells) >= 23:
    cell_21_src = ''.join(cells[21]['source'])
    cell_22_src = ''.join(cells[22]['source'])
    print("✅ OK  | SỬA 5 - Cell 21 là GHI CHÚ AI" if "GHI CHÚ AI" in cell_21_src else "❌ FAIL | SỬA 5 - Cell 21 là GHI CHÚ AI")
    print("✅ OK  | SỬA 5 - Cell 22 là KẾ HOẠCH TUẦN 5" if "KẾ HOẠCH TUẦN 5" in cell_22_src else "❌ FAIL | SỬA 5 - Cell 22 là KẾ HOẠCH TUẦN 5")
else:
    print("❌ FAIL | Không đủ số cell (cần 23)")
