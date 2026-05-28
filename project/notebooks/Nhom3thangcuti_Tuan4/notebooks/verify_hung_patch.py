import json

with open('Tuan4_Hung_Dinov3+miniML.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb["cells"]
print(f"Total cells: {len(cells)}")
print()

# Kiểm tra các fix quan trọng
checks = {
    "minilm_multilingual": False,
    "gt_len_continue":     False,
    "grid_search_alpha":   False,
    "val_test_split":      False,
    "save_csv_metric":     False,
}

for i, c in enumerate(cells):
    src = "".join(c["source"])
    t = c["cell_type"]
    label = src[:80].replace("\n", " ")
    print(f"  [{i:02d}] {t:8s} | {label}")

    if "paraphrase-multilingual-MiniLM-L12-v2" in src:
        checks["minilm_multilingual"] = True
    if ("gt_len == 0" in src or "gt_n == 0" in src or "len(gt) == 0" in src) and "continue" in src:
        checks["gt_len_continue"] = True
    if "alpha_grid" in src or "grid search" in src.lower():
        checks["grid_search_alpha"] = True
    if "val_idx" in src and "test_idx" in src:
        checks["val_test_split"] = True
    if "metrics_dinov3_minilm" in src and ".csv" in src:
        checks["save_csv_metric"] = True

print()
print("=" * 50)
print("VERIFICATION CHECKLIST:")
for k, v in checks.items():
    status = "✅" if v else "❌"
    print(f"  {status} {k}")
