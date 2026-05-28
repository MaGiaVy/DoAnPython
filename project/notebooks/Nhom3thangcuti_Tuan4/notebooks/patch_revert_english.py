"""
patch_revert_english.py – Revert MiniLM to English version all-MiniLM-L6-v2
"""

import json

NB = r"d:\New folder (3)\project\notebooks\Nhom3thangcuti_Tuan4\notebooks\Tuan4_Hung_Dinov3+miniML.ipynb"

with open(NB, encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        new_source = []
        for line in cell["source"]:
            # Revert model name
            line = line.replace("paraphrase-multilingual-MiniLM-L12-v2", "all-MiniLM-L6-v2")
            # Revert feature file name
            line = line.replace("multilingual_minilm_text_features.npy", "minilm_text_features.npy")
            # Revert batch size back to 256 for text encode since all-MiniLM-L6-v2 is smaller
            if "batch_size=128" in line and "encode" in line:
                # check if it is part of minilm encoding
                line = line.replace("batch_size=128", "batch_size=256")
            new_source.append(line)
        cell["source"] = new_source

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Reverted to all-MiniLM-L6-v2 successfully!")
