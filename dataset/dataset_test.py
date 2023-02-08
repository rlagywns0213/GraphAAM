import os
import csv
import math
import time
import random
import numpy as np
import json

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler

from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, DataLoader

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import RDLogger                                                                                                                                                               
RDLogger.DisableLog('rdApp.*')  


ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]

def read_smiles(data_path, target, task):
    smiles_data, labels = [], []
    with open(data_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i != 0:
                smiles = row['smiles']
                label = row[target]
                mol = Chem.MolFromSmiles(smiles)
                if mol != None and label != '':
                    smiles_data.append(smiles)
                    if task == 'classification':
                        labels.append(int(label))
                    elif task == 'regression':
                        labels.append(float(label))
                    else:
                        ValueError('task must be either regression or classification')
    print(len(smiles_data))
    return smiles_data, labels

#### for atommapping ####

class AtomMappingDatasetWrapper(object):
    
    def __init__(
        self,
        dataset_name : str = 'train',
        train_path: str = '',
        val_path: str = '',
        test_path: str = '',
        train_batch_size: int = 1024,
        eval_batch_size: int = 1024,
        num_workers: int = 0):
        super().__init__()
        if dataset_name =='train':
            self.data = json.load(open(train_path, 'r'))
            # self.data = self.data[:10000] #train 50000개만
        elif dataset_name =='val':
            self.data = json.load(open(val_path, 'r'))
        elif dataset_name =='test':
            self.data = json.load(open(test_path, 'r'))

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers

    def get_data_loaders(self):
        dataset_format = AtommappingDataset(data=self.data)
        sampler = RandomSampler(dataset_format)

        data_loader = DataLoader(
            dataset_format, batch_size=self.train_batch_size, sampler=sampler,
            num_workers=self.num_workers, drop_last=False
        )
        return data_loader


class AtommappingDataset(Dataset):
    def __init__(self, data):
        super(Dataset, self).__init__()
        self.smiles_data = read_atoms(data)

    def __getitem__(self, index):
        reactant, product, label = self.smiles_data[index]
        reactant_data = atom_to_adj_mapping(reactant, label)
        product_data = atom_to_adj_mapping(product, label)
        sample = {"reactant": reactant_data, "product": product_data}
        return sample

    def __len__(self):
        return len(self.smiles_data)

    
def atom_to_adj_mapping(atom_data, y):
    mol = Chem.MolFromSmiles(atom_data)
    # mol = Chem.AddHs(mol)

    N = mol.GetNumAtoms()
    M = mol.GetNumBonds()

    type_idx = []
    chirality_idx = []
    atomic_number = []
    for atom in mol.GetAtoms():
        type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
        chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
        atomic_number.append(atom.GetAtomicNum())

    x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
    x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
    x = torch.cat([x1, x2], dim=-1)

    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_feat.append([
            BOND_LIST.index(bond.GetBondType()),
            BONDDIR_LIST.index(bond.GetBondDir())
        ])
        edge_feat.append([
            BOND_LIST.index(bond.GetBondType()),
            BONDDIR_LIST.index(bond.GetBondDir())
        ])

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr) # torch_geometric.utils.to_dense_adj(edge_index)
    return data

    
def read_atoms(data):
    smiles_data = []
    for i, row in enumerate(data):
        reactants, products = row['unmapped_rxn'].split('>>')
        reactants_mol = Chem.MolFromSmiles(reactants)
        products_mol = Chem.MolFromSmiles(products)
        if (reactants_mol != None) & (products_mol != None):
            smiles_data.append((reactants,products, row['correct_map']))
    print(len(smiles_data))
    return smiles_data

# class MolTestDataset(Dataset):
#     def __init__(self, data_path, target, task):
#         super(Dataset, self).__init__()
#         self.smiles_data, self.labels = read_smiles(data_path, target, task)
#         self.task = task

#         self.conversion = 1
#         if 'qm9' in data_path and target in ['homo', 'lumo', 'gap', 'zpve', 'u0']:
#             self.conversion = 27.211386246
#             print(target, 'Unit conversion needed!')

#     def __getitem__(self, index):
#         mol = Chem.MolFromSmiles(self.smiles_data[index])
#         mol = Chem.AddHs(mol)

#         N = mol.GetNumAtoms()
#         M = mol.GetNumBonds()

#         type_idx = []
#         chirality_idx = []
#         atomic_number = []
#         for atom in mol.GetAtoms():
#             type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
#             chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
#             atomic_number.append(atom.GetAtomicNum())

#         x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
#         x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
#         x = torch.cat([x1, x2], dim=-1)

#         row, col, edge_feat = [], [], []
#         for bond in mol.GetBonds():
#             start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
#             row += [start, end]
#             col += [end, start]
#             edge_feat.append([
#                 BOND_LIST.index(bond.GetBondType()),
#                 BONDDIR_LIST.index(bond.GetBondDir())
#             ])
#             edge_feat.append([
#                 BOND_LIST.index(bond.GetBondType()),
#                 BONDDIR_LIST.index(bond.GetBondDir())
#             ])

#         edge_index = torch.tensor([row, col], dtype=torch.long)
#         edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
#         if self.task == 'classification':
#             y = torch.tensor(self.labels[index], dtype=torch.long).view(1,-1)
#         elif self.task == 'regression':
#             y = torch.tensor(self.labels[index] * self.conversion, dtype=torch.float).view(1,-1)
#         data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
#         return data

#     def __len__(self):
#         return len(self.smiles_data)