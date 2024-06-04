import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import math

def smiles_unique(smiles_list):
    unique_smiles = set()
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            unique_smiles.add(Chem.MolToSmiles(mol, isomericSmiles=True))
    unique_smiles = list(unique_smiles)
    return unique_smiles

def read_smi_file(filepath):
    smiles_list = []
    with open(filepath, 'r') as file:
        for line in file:
            smiles = line.strip()  # 行末の改行文字や空白を取り除く
            if smiles:  # 空行を無視
                smiles_list.append(smiles)
    return smiles_list

def analyze_smiles(smiles_list):
    results = {
        'Molecular Weight': [],
        'MolLogP': [],
        'QED': [],
        'Valid': []
    }

    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mol_weight = Descriptors.MolWt(mol)
                mol_logp = Descriptors.MolLogP(mol)
                mol_qed = QED.qed(mol)
                results['Molecular Weight'].append(mol_weight)
                results['MolLogP'].append(mol_logp)
                results['QED'].append(mol_qed)
                results['Valid'].append(True)
            else:
                results['Valid'].append(False)
        except Exception as e:
            results['Valid'].append(False)
    
    return results

def calculate_bins(data):
    return math.ceil(math.log2(len(data)) + 1)

def plot_histogram(data1, data2, xlabel, ylabel, title, filename):
    bins1 = calculate_bins(data1)
    bins2 = calculate_bins(data2)
    bins = max(bins1, bins2)  # どちらか大きい方のビン数を使用
    


    plt.figure(figsize=(10, 6))
    sns.histplot(data1, color='blue', label='Generated', kde=True, stat="density", bins=bins)
    sns.histplot(data2, color='red', label='IEV2', kde=True, stat="density", bins=bins, alpha=0.4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

def plot_combined_scatter_and_contour(x1, y1, x2, y2, xlabel, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    
    # 等高線プロットを作成
    sns.kdeplot(x=x2, y=y2, cmap="Reds", shade=True, bw_adjust=0.5, alpha=0.7, label='IEV2')

    # 散布図（点プロット）を作成
    plt.scatter(x1, y1, c='blue', label='Generated', alpha=0.55, s=6)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    # プロットをファイルに保存
    plt.savefig(filename)
    plt.show()

# SMIファイルを読み込む
#必ずパス１が生成したもの、パス２がIEV2のもの
#filepath1 = 'generated_molecules_noseed_valid.smi'
filepath1 = '6CM4_dock_ligands.smi'
smiles_list1 = read_smi_file(filepath1)
filepath2 = 'Datasets/drd2_train_smiles_no_dot.smi'
smiles_list2 = read_smi_file(filepath2)

# 重複を排除
smiles_list1 = smiles_unique(smiles_list1)
smiles_list2 = smiles_unique(smiles_list2)

# 分析
results1 = analyze_smiles(smiles_list1)
results2 = analyze_smiles(smiles_list2)

# プロット（散布図と等高線プロットの重ね表示）
plot_combined_scatter_and_contour(results1['Molecular Weight'], results1['MolLogP'], results2['Molecular Weight'], results2['MolLogP'], 'Molecular Weight', 'MolLogP', 'Molecular Weight vs MolLogP', 'plot/molecular_weight_vs_mollogp.png')
plot_combined_scatter_and_contour(results1['Molecular Weight'], results1['QED'], results2['Molecular Weight'], results2['QED'], 'Molecular Weight', 'QED', 'Molecular Weight vs QED', 'plot/molecular_weight_vs_qed.png')
plot_combined_scatter_and_contour(results1['MolLogP'], results1['QED'], results2['MolLogP'], results2['QED'], 'MolLogP', 'QED', 'MolLogP vs QED', 'plot/mollogp_vs_qed.png')

# プロット（ヒストグラム）
plot_histogram(results1['Molecular Weight'], results2['Molecular Weight'], 'Molecular Weight', 'Density', 'Histogram of Molecular Weight', 'plot/histogram_molecular_weight.png')
plot_histogram(results1['MolLogP'], results2['MolLogP'], 'MolLogP', 'Density', 'Histogram of MolLogP', 'plot/histogram_mollogp.png')
plot_histogram(results1['QED'], results2['QED'], 'QED', 'Density', 'Histogram of QED', 'plot/histogram_qed.png')
