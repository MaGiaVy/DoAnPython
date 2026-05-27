"""
diagnose_hung.py  –  Chẩn đoán tại sao mAP thấp hơn sau patch
"""
import json, os, sys

NB = r"d:\New folder (3)\project\notebooks\Nhom3thangcuti_Tuan4\notebooks\Tuan4_Hung_Dinov3+miniML.ipynb"

with open(NB, encoding="utf-8") as f:
    nb = json.load(f)

print("=" * 60)
print("CHẨN ĐOÁN NOTEBOOK")
print("=" * 60)

# 1. Xem grid search cell chọn tham số gì
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    if "alpha_grid" in src and "threshold_grid" in src:
        print(f"\n[Cell {i}] GRID SEARCH parameters:")
        for line in c["source"]:
            if any(k in line for k in ["alpha_grid", "threshold_grid", "best_alpha", "best_threshold", "ALPHA", "PHASH_THRESH"]):
                print("  ", line.rstrip())

    if "ALPHA" in src and "PHASH_THRESH" in src and "sim_batch" in src:
        print(f"\n[Cell {i}] SIMILARITY cell:")
        for line in c["source"]:
            if any(k in line for k in ["ALPHA", "PHASH_THRESH", "best_alpha", "best_threshold"]):
                print("  ", line.rstrip())

# 2. Xem outputs hiện tại của metric cell
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    if "mAP@5" in src or "map_at_5" in src or "full_map5" in src:
        outputs = c.get("outputs", [])
        for out in outputs:
            if out.get("output_type") == "stream":
                txt = "".join(out.get("text", []))
                if txt.strip():
                    print(f"\n[Cell {i}] OUTPUT:")
                    print(txt[:500])

print()
print("=" * 60)
print("NGUYÊN NHÂN CÓ THỂ:")
print("  1. Grid search tìm ra tham số tệ hơn vì text features vẫn là English MiniLM cũ")
print("  2. pHash threshold 5-10 tạo quá nhiều false positives → precision giảm → mAP giảm")
print("  3. Đang so sánh val/test mAP (nhỏ hơn) với full dataset mAP gốc (0.7388)")
print()
print("GIẢI PHÁP:")
print("  → Thêm safety check: nếu grid search tệ hơn default thì revert về alpha=0.7, threshold=2")
print("  → Fix 1 (gt_len→continue) vẫn đúng và guaranteed tăng mAP")
print("  → Cần re-extract text features với multilingual model để Fix 2 có tác dụng")
