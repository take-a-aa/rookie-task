import os
import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem
import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils import Vocabulary
from Dataset import MolData
from Model import Generator

class PG:
    def __init__(self, emb_size=128, hidden_size=512, num_layers=3, dropout=0.5, convs=None,
                 lr=0.001, load_dir_G=None, voc=None, device=None):
        self.voc = voc
        self.generator = Generator(self.voc, emb_size=emb_size, hidden_size=hidden_size,
                                   num_layers=num_layers, dropout=dropout)
        if device:
            self.device = torch.device(device)
            self.generator = self.generator.to(self.device)
        else:
            self.device = device
            
        # 学習済みモデルの読み込み
        if load_dir_G:
            checkpoint = torch.load(load_dir_G)
            self.generator.load_state_dict(checkpoint['generator_state_dict'])

    def _proceed_sequences(self, prevs, states, max_len):
        # 省略
        with torch.no_grad():
            n_sequences = prevs.shape[0]
            sequences = []
            lengths = torch.zeros(n_sequences, dtype=torch.long, device=prevs.device)
            one_lens = torch.ones(n_sequences, dtype=torch.long, device=prevs.device)
            is_end = prevs.eq(self.voc.vocab['EOS']).view(-1)
            for _ in range(max_len):
                outputs, _, states = self.generator(prevs, one_lens, states)
                probs = F.softmax(outputs, dim=-1).view(n_sequences, -1)
                currents = torch.multinomial(probs, 1)
                currents[is_end, :] = self.voc.vocab['PAD']
                sequences.append(currents)
                lengths[~is_end] += 1
                is_end[currents.view(-1) == self.voc.vocab['EOS']] = 1
                if is_end.sum() == n_sequences:
                    break
                prevs = currents
            sequences = torch.cat(sequences, dim=-1)
        return sequences, lengths

    def sample_tensor(self, n, max_len=100):
        prevs = torch.empty(n, 1, dtype=torch.long, device=self.device).fill_(self.voc.vocab['GO'])
        samples, lengths = self._proceed_sequences(prevs, None, max_len)
        samples = torch.cat([prevs, samples], dim=-1)
        lengths += 1
        return samples, lengths

# 学習済みファイルのパス
load_dir_G = 'test_AD_save_DRD2_merged/G_990.ckpt'
output_file = 'generated_molecules_noseed.smi'

# モデルの読み込み
voc = Vocabulary(init_from_file='Datasets/Voc')
generator = PG(load_dir_G=load_dir_G, voc=voc, device='cuda')

# 新しい分子の生成
num_samples = 1000
max_length = 100
generator.generator.eval() 
samples, lengths = generator.sample_tensor(num_samples, max_length)

"""
# サンプリングした分子のSMILESを復元
smiles_list = []

#seedの読み込み
seed_smiles = []
with open('Datasets/drd2_train_smiles_no_dot.smi', 'r') as f:
    f1 = f.readlines()
    counter = 0
    for line in f1:
        counter += 1
        if counter > 12:
            break
        smiles, _ = line.split(' ', 1)  # CHEMBLの部分を削除
        seed_smiles.append(smiles)
#seedから生成
for i in range(10):
    print("seed", (i,seed_smiles[i]))
    # シード化合物のSMILESをインデックスに変換
    seed_indices = voc.encode(seed_smiles[i])
    torch.manual_seed(seed_value)

    # 生成器をevalモードに設定
    generator.generator.eval()

    # シード化合物のインデックスからサンプリング開始
    prevs = torch.tensor(seed_indices, dtype=torch.long, device=generator.device).unsqueeze(0)
    samples, lengths = generator._proceed_sequences(prevs, None, max_length)

    for i in range(num_samples):
        smiles = voc.decode(samples[i,:lengths[i]].tolist())
        smiles_list.append(smiles)
"""


# サンプリングした分子のSMILESをファイルに出力
with open(output_file, 'w') as f:
    for i in range(num_samples):
        smiles = voc.decode(samples[i,:lengths[i]].tolist())
        f.write(smiles[2:] + '\n')

print(f'Generated molecules were saved to {output_file}')