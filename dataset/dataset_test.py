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

#### Graphormer

from torch import IntTensor
from numpy import minimum, nan_to_num, ones
from scipy.sparse.csgraph import shortest_path

ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC, BT.UNSPECIFIED]
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
        num_workers: int = 0,
        multiple_solution = False,
        bert_input = False,
        ):
        super().__init__()
        if dataset_name =='train':
            self.data = json.load(open(train_path, 'r'))
            # self.data = self.data[:1000] #train 10000개만
            print("train dataset len:", len(self.data))
        elif dataset_name =='val':
            self.data = json.load(open(val_path, 'r'))
            # self.data = self.data[:100] #train 10000개만
            print("valid dataset len:", len(self.data))
        elif dataset_name =='test':
            self.data = json.load(open(test_path, 'r'))
            # self.data = self.data[:500] #train 10000개만

            # if 'nat' in test_path:
            #     data_list = []
            #     for i in range(len(self.data['data_set'])):
            #         if self.data['correct_maps'][str(i)] == []:
            #             continue
            #         else:
            #             d = {'mapped_rxn': self.data['CORRECT MAPPING'][str(i)], 'unmapped_rxn': self.data['rxn'][str(i)], 'correct_map': self.data['correct_maps'][str(i)][0]}
            #             data_list.append(d)
            #     print("human curated dataset len:", len(data_list))
            #     self.data = data_list

        self.dataset_name = dataset_name
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.multiple_solution = multiple_solution
        self.bert_input = bert_input

    def get_data_loaders(self):
        dataset_format = AtommappingDataset(data=self.data, multiple_solution = self.multiple_solution,
                                            bert_input = self.bert_input)
        if 'test' != self.dataset_name:
            sampler = RandomSampler(dataset_format)

            data_loader = DataLoader(
                dataset_format, batch_size=self.train_batch_size, sampler=sampler,
                num_workers=self.num_workers, drop_last=False, pin_memory=True, 
            )
        else:
            data_loader = DataLoader(
                dataset_format, batch_size=self.eval_batch_size,
                num_workers=self.num_workers, drop_last=False, pin_memory=True
            )
        return data_loader



class AtommappingDataset(Dataset):
    def __init__(self, data, multiple_solution, bert_input):
        super(Dataset, self).__init__()
        self.multiple_solution = multiple_solution
        self.bert_input = bert_input
        self.smiles_data = self.read_atoms(data)
        # self.model_path = 'rxnmapper/models/transformers/albert_heads_8_uspto_all_1310k'
        # self.model = AlbertModel.from_pretrained(
        #     self.model_path,
        #     output_attentions=True,
        #     output_past=False,
        #     output_hidden_states=False,
        #     return_dict=True,
        # )
        # vocab_path = os.path.join(self.model_path, "vocab.txt")
        # self.tokenizer = SmilesTokenizer(vocab_path, max_len=self.model.config.max_position_embeddings)
        

    def __getitem__(self, index):
        reactant, product, label, multiple_label = self.smiles_data[index]
        if self.bert_input:
            reactant_data = self.atom_to_adj_mapping_bert(reactant, label, multiple_label)
            product_data = self.atom_to_adj_mapping_bert(product, label, multiple_label)

        else:
            reactant_data = self.full_atom_to_graphormer(reactant, product,label, multiple_label)
            # reactant_data = self.atom_to_graphormer(reactant, label, multiple_label)
            # product_data = self.atom_to_graphormer(product, label, multiple_label)
        # sample = {"reactant": reactant_data, "product": product_data}
        sample = {"reactions": reactant_data}
        return sample

    def __len__(self):
        return len(self.smiles_data)

    def get_features(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        N = mol.GetNumAtoms()
        type_idx = []
        formal_charges=[]
        degrees = []
        implicit_hs = []
        explicit_hs = []
        idx = 0
        for idx, atom in enumerate(mol.GetAtoms()):
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum() +1)) #1~
            degrees.append(atom.GetDegree()+1) # 1~
            implicit_hs.append(atom.GetNumImplicitHs()+1) # 1~
            explicit_hs.append(atom.GetNumExplicitHs()+1) # 1~
            formal_charges.append(atom.GetFormalCharge()+7) # -5~6 -> -6~6 => 1~13
        adjacency_matrix = Chem.GetAdjacencyMatrix(mol)
        assert adjacency_matrix.shape[0] == len(type_idx)
        # sp = shortest_path(adjacency_matrix, method='FW', directed=False, unweighted=True)
        # nan_to_num(sp, copy=False, posinf=(1)) # treat components as isolated molecules 하고 싶으면 0으로
        # minimum(sp, 10 + 2, out=sp) # max_distance: set distances greater than cutoff to cutoff value : 10
        type_idx = torch.tensor(type_idx, dtype=torch.long)
        degrees = torch.tensor(degrees, dtype=torch.long).view(1,-1)
        imp_hs = torch.tensor(implicit_hs, dtype=torch.long).view(1,-1)
        exp_hs = torch.tensor(explicit_hs, dtype=torch.long).view(1,-1)
        fcs = torch.tensor(formal_charges, dtype=torch.long).view(1,-1)
        x = torch.cat((degrees, imp_hs, exp_hs, fcs), dim=0)

        edge_type_matrix = np.ones_like(adjacency_matrix)
        ### edge encoding ###
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_type_matrix[start][end] = BOND_LIST.index(bond.GetBondType()) +2 # 2부터 시작, 1: bond X!, 0은 패딩을 위함
            edge_type_matrix[end][start] = BOND_LIST.index(bond.GetBondType()) +2 # 2부터 시작, 1: bond X!, 0은 패딩을 위함
            
        return type_idx, x, adjacency_matrix, edge_type_matrix
        


    def full_atom_to_graphormer(self, reactant, product, y, multiple_label=None):
        react_type, react_x, react_adj, react_edge_type  = self.get_features(reactant)
        product_type, product_x, product_adj, product_edge_type = self.get_features(product)
        reaction_atom_type = torch.cat((react_type, product_type))
        reaction_fts = torch.cat((react_x, product_x), dim=-1)
        N_r,N_p = react_adj.shape[0], product_adj.shape[0]
        reaction_adj = np.zeros((N_r+N_p,N_r+N_p)) # adj for SP
        reaction_adj[:N_r, :N_r] = react_adj
        reaction_adj[-N_p:,-N_p:] = product_adj

        reaction_edge_types = np.ones((N_r+N_p,N_r+N_p)) # edgetype adj matrix
        reaction_edge_types[:N_r, :N_r] = react_edge_type
        reaction_edge_types[-N_p:,-N_p:] = product_edge_type
        
        # ################ TODO : lape ###################
        # reaction_degrees = np.sum(reaction_adj, axis=1)
        # laplacian = np.diag(reaction_degrees) - reaction_degrees
        # eigenvalues, eigenvectors = np.linalg.eig(laplacian)
        # pe_dim = 15
        # lape = eigenvectors[:, 1:pe_dim+1]
        # ################ TODO : lape ###################


        sp = shortest_path(reaction_adj, method='FW', directed=False, unweighted=True) +2 # 자기 자신 0, 연결 X : -inf
        nan_to_num(sp, copy=False, posinf=(1)) # treat components as isolated molecules 하고 싶으면 0으로
        minimum(sp, 10 + 2, out=sp) # max_distance: set distances greater than cutoff to cutoff value : 10
        
        data =Data(x = reaction_atom_type, react_fts = reaction_fts.tolist(), devide = N_r, 
                   reaction_smiles = reactant+">>"+product, edge_type = reaction_edge_types, 
                   y = y, sp=sp, multiple_y = multiple_label) 
        return data


    def atom_to_adj_mapping_bert(self, atom_data, y, multiple_label=None):
        
        mol = Chem.MolFromSmiles(atom_data)
        mol_hs = Chem.AddHs(mol)

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        hs_atomic_degree = []
        bert_input = []
        for atom in mol_hs.GetAtoms():
            type_id = ATOM_LIST.index(atom.GetAtomicNum())
            if type_id == 0:
                # hydrogen - pass
                continue
            else:
                bert_input.append(atom.GetSmarts())
                type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
                chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
                atomic_number.append(atom.GetAtomicNum())
                hs_atomic_degree.append(atom.GetDegree())

        atomic_degree = []
        for atom in mol.GetAtoms():
            atomic_degree.append(atom.GetDegree())
        
        only_hs_degree =[a-b for a,b in zip(hs_atomic_degree, atomic_degree)]
        # x1 = torch.tensor(tokenizer.convert_tokens_to_ids(bert_input)).view(-1,1)
        # x1 = torch.tensor(tokenizer.convert_tokens_to_ids(bert_input)).view(-1,1)

        # x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(atomic_degree, dtype=torch.long).view(-1,1)
        x3 = torch.tensor(only_hs_degree, dtype = torch.long).view(-1,1)
        x = torch.cat([x2, x3], dim=-1)

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
        edge_csv_inf = []
        for a, b, c in zip(row, col, edge_attr[:,0]):
            edge_csv_inf.append((a,b,int(c)))

        # bert_input= [" ".join(i) for i in bert_input]
        # data = Data(x=x, y=y, multiple_y = multiple_label, edge_index=edge_index, edge_attr=edge_attr, inference_edge_csv =edge_csv_inf , smiles = atom_data) # torch_geometric.utils.to_dense_adj(edge_index)
        
        data = Data(x=x, y=y, multiple_y = multiple_label, edge_index=edge_index, edge_attr=edge_attr,bert_input=bert_input, inference_edge_csv =edge_csv_inf , smiles = atom_data) # torch_geometric.utils.to_dense_adj(edge_index)
        return data
    
    
    def read_atoms(self, data):
        smiles_data = []
        for i, row in enumerate(data):
            reactants, products = row['unmapped_rxn'].split('>>')
            reactants_mol = Chem.MolFromSmiles(reactants)
            products_mol = Chem.MolFromSmiles(products)
            if (reactants_mol != None) & (products_mol != None):
                # if self.multiple_solution:
                smiles_data.append((reactants,products, row['ori_correct_map'], row['correct_map']))
                # else:
                    # smiles_data.append((reactants,products, row['ori_correct_map']))

        print(len(smiles_data))
        return smiles_data

# def atom_to_adj_mapping(atom_data, y):
#     mol = Chem.MolFromSmiles(atom_data)
#     # mol = Chem.AddHs(mol)

#     N = mol.GetNumAtoms()
#     M = mol.GetNumBonds()

#     type_idx = []
#     chirality_idx = []
#     for atom in mol.GetAtoms():
#         type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
#         chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))

#     x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
#     x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
#     x = torch.cat([x1, x2], dim=-1)

#     row, col, edge_feat = [], [], []
#     for bond in mol.GetBonds():
#         start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
#         row += [start, end]
#         col += [end, start]
#         edge_feat.append([
#             BOND_LIST.index(bond.GetBondType()),
#             BONDDIR_LIST.index(bond.GetBondDir())
#         ])
#         edge_feat.append([
#             BOND_LIST.index(bond.GetBondType()),
#             BONDDIR_LIST.index(bond.GetBondDir())
#         ])

#     edge_index = torch.tensor([row, col], dtype=torch.long)
#     edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
#     edge_csv_inf = []
#     for a, b, c in zip(row, col, edge_attr[:,0]):
#         edge_csv_inf.append((a,b,int(c)))

#     data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, inference_edge_csv =edge_csv_inf , smiles = atom_data) # torch_geometric.utils.to_dense_adj(edge_index)
#     return data


# def atom_to_adj_mapping_others(atom_data, y):
#     mol = Chem.MolFromSmiles(atom_data)
#     mol_hs = Chem.AddHs(mol)

#     N = mol.GetNumAtoms()
#     M = mol.GetNumBonds()

#     features=[]
#     for atom in mol.GetAtoms():
#         feature = atom_features(atom)
#         features.append(feature / sum(feature))

#     row, col, edge_feat = [], [], []
#     for bond in mol.GetBonds():
#         start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
#         row += [start, end]
#         col += [end, start]
#         edge_feat.append([
#             BOND_LIST.index(bond.GetBondType()),
#             BONDDIR_LIST.index(bond.GetBondDir())
#         ])
#         edge_feat.append([
#             BOND_LIST.index(bond.GetBondType()),
#             BONDDIR_LIST.index(bond.GetBondDir())
#         ])

#     edge_index = torch.tensor([row, col], dtype=torch.long)
#     edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
#     data = Data(x=torch.tensor(features, dtype = torch.float32), y=y, edge_index=edge_index, edge_attr=edge_attr) # torch_geometric.utils.to_dense_adj(edge_index)
#     return data

# def atom_features(atom):
#     # from https://github.com/moen-hyb/ATMOL/blob/b98cb5857d72c738d4756dc97cbd124bd366acf5/utils_gat_pretrain.py#L14
#     # 아직 sp1, sp2, sp3 는 반영X (torch_geometric)
#     hybridization = atom.GetHybridization()
#     return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
#                                           ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
#                                            'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
#                                            'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
#                                            'Pt', 'Hg', 'Pb', 'Unknown']) +
#                     one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
#                     one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
#                     one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
#                     [atom.GetIsAromatic()])
#                     # [1 if hybridization == HybridizationType.SP else 0]+
#         # sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
#         # sp3.append(1 if hybridization == HybridizationType.SP3 else 0))

# def one_of_k_encoding_unk(x, allowable_set):
#     """Maps inputs not in the allowable set to the last element."""
#     if x not in allowable_set:
#         x = allowable_set[-1]
#     return list(map(lambda s: x == s, allowable_set))


# def one_of_k_encoding(x, allowable_set):
#     if x not in allowable_set:
#         raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
#     return list(map(lambda s: x == s, allowable_set))
