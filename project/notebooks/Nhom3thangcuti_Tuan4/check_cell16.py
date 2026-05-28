import json, sys
sys.stdout.reconfigure(encoding='utf-8')
nb_path = r'd:\New folder (3)\project\notebooks\Nhom3thangcuti_Tuan4\notebooks\Tuan4_GiaVy_Dinov3+miniML.ipynb'
with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)
src16 = ''.join(nb['cells'][16]['source'])
# Check the exact old string is gone
old_str = 'np.save(os.path.join(output_dir, "minilm_text_features.npy"), text_features)'
new_str = 'np.save(os.path.join(output_dir, "multilingual_minilm_text_features.npy"), text_features)'
print("Old string present:", old_str in src16)
print("New string present:", new_str in src16)
print()
# Print the save line
for line in src16.splitlines():
    if 'minilm_text_features' in line:
        print(repr(line))
