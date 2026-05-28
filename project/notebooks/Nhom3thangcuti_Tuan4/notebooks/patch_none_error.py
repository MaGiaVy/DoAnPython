import json
import glob
import os

def patch_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                if "best_alpha_2, best_map5_2 = None, -1.0" in line:
                    new_source.append(line.replace("None", "0.5"))
                    modified = True
                elif "best_alpha_1, best_map5 = None, -1.0" in line:
                    new_source.append(line.replace("None", "0.5"))
                    modified = True
                elif "'mAP@5'      : float(np.mean(ap_list)),\n" in line:
                    new_source.append(line.replace("float(np.mean(ap_list))", "float(np.mean(ap_list)) if ap_list else 0.0"))
                    modified = True
                elif "'Precision@1': float(np.mean(p1_list)),\n" in line:
                    new_source.append(line.replace("float(np.mean(p1_list))", "float(np.mean(p1_list)) if p1_list else 0.0"))
                    modified = True
                elif "'Recall@5'   : float(np.mean(r5_list)),\n" in line:
                    new_source.append(line.replace("float(np.mean(r5_list))", "float(np.mean(r5_list)) if r5_list else 0.0"))
                    modified = True
                else:
                    new_source.append(line)
            cell['source'] = new_source
            
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Patched {os.path.basename(filepath)}")
    else:
        print(f"No changes needed for {os.path.basename(filepath)}")

base_dir = r"d:\New folder (3)\project\notebooks\Nhom3thangcuti_Tuan4\notebooks"
for nb_file in glob.glob(os.path.join(base_dir, "Baseline*.ipynb")):
    patch_notebook(nb_file)
