from rdkit import Chem

def check_smiles_validity(smiles_list):
    valid_smiles = []
    invalid_smiles = []
    
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                valid_smiles.append(smiles)
            else:
                invalid_smiles.append(smiles)
        except:
            invalid_smiles.append(smiles)
    
    return valid_smiles, invalid_smiles

# SMILESリスト
def read_smi_file(filepath):
    smiles_list = []
    with open(filepath, 'r') as file:
        for line in file:
            smiles = line.strip()  # 行末の改行文字や空白を取り除く
            if smiles:  # 空行を無視
                smiles_list.append(smiles)
    return smiles_list

# 例のSMIファイルを読み込む
filepath = 'generated_molecules_noseed.smi'
#filepath = 'generated_molecules.smi'
smiles_list = read_smi_file(filepath)

#

#drd2のもの
with open('Datasets/drd2_train_smiles_no_dot.smi', 'r') as f1:
    lines1 = f1.readlines()

# drd2_train_smiles_no_dot.smiの各行を処理
processed_lines1 = []
for line in lines1:
    smiles, _ = line.split(' ', 1)  # CHEMBLの部分を削除
    processed_lines1.append(f"{smiles}\n")


#smiles_list = processed_lines1


valid_smiles, invalid_smiles = check_smiles_validity(smiles_list)
print("Valid SMILES:", len(valid_smiles),"/",len(smiles_list))
print("Invalid SMILES:", len(invalid_smiles),"/",len(smiles_list))

# サンプリングした分子のSMILESをファイルに出力
output_file = 'generated_molecules_noseed_valid.smi'
with open(output_file, 'w') as f:
    for smiles in valid_smiles:
        f.write(smiles + '\n')

print(f'Valid smiles in {filepath} were saved to {output_file}')
#print(smiles_list[:10])