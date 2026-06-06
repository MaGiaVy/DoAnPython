import sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'd:\New folder (3)\project\notebooks\Nhom3thangcuti_Tuan4\Nhom3ThangCuTi_Tuan5_33.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("=== Mentions of 's1' ===")
for i, line in enumerate(lines, 1):
    if ' s1' in line or 'S1' in line or '_s1' in line:
        print(f'L{i}: {line.rstrip()[:150]}')

print("\n=== Mentions of L2 norm riêng/độc lập ===")
for i, line in enumerate(lines, 1):
    stripped = line.lower()
    if ('l2' in stripped and ('riêng' in stripped or 'độc lập' in stripped or 'norm' in stripped)) or 'q_img_norm' in stripped or 'q_txt_norm' in stripped:
        print(f'L{i}: {line.rstrip()[:150]}')
