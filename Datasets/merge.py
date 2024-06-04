import random

# ファイルを読み込む
with open('drd2_train_smiles_no_dot.smi', 'r') as f1, open('Data4InitD_neg_small.smi', 'r') as f2:
    lines1 = f1.readlines()
    lines2 = f2.readlines()

# drd2_train_smiles_no_dot.smiの各行を処理
processed_lines1 = []
for line in lines1:
    smiles, _ = line.split(' ', 1)  # CHEMBLの部分を削除
    processed_lines1.append(f"{smiles},1\n")

# data.smiの各行をそのまま使用
processed_lines2 = lines2

# 2つのリストをランダムに並べ替え
all_lines = processed_lines1 + processed_lines2
random.shuffle(all_lines)

# 結果を書き込む
with open('merged_data.smi', 'w') as f_out:
    for line in all_lines:
        f_out.write(line)