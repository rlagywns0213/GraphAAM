import torch
import pytorch_lightning as pl
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import torch.nn.functional as F
# import evaluate
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, OneCycleLR
import numpy as np
from pytorch_lightning.callbacks import model_checkpoint, early_stopping, LearningRateMonitor, ModelSummary
from datetime import datetime, timezone, timedelta
import os
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import rdkit.Chem as Chem

from torch.nn.utils.rnn import pad_sequence
from torch import IntTensor, Size, int32, ones as t_ones, zeros
from graph_transformer import MoleculeEncoder, RXNMoleculeEncoder

class FG(pl.LightningModule):
    def __init__(
        self,
        model_params: Optional = None,
        train_params: Optional = None,
        config: Optional = None,
        logger: Optional = None,
        mode : Optional = False,
        multiple_solution : Optional = False,
        test_atom : Optional =False, 
    ):
        """
        Main pytorch lightning module for wMetric Learning task. Configuration files are read from "./configs'
        :param model_params: Model parameters configuration. [dict]
        :param train_params:  Training parameters configuration. [dict]
        :param dataset_params: Dataset parameters configuration. [dict]
        :param model_name: Name of the current model. [str]
        """
        super().__init__()
        self.save_hyperparameters()

        self.model_params = model_params
        self.train_params = train_params
        self.multiple_solution = multiple_solution

        self.feat_dim = self.model_params['feat_dim']

        self.max_neighbors = self.model_params['max_neighbors']
        self.max_distance = self.model_params['max_distance']
        self.d_model = self.model_params['d_model']
        self.n_head = self.model_params['n_head']
        self.num_layers = self.model_params['num_layers']
        self.dim_feedforward = self.model_params['dim_feedforward']
        self.dropout = self.model_params['dropout']
        self.test_atom = True #True #test_atom
        print(config)

        self.network = MoleculeEncoder(max_neighbors = self.max_neighbors, 
                                       max_distance = self.max_distance, 
                                       shared_weights=True,
                                       d_model = self.d_model, 
                                       nhead = self.n_head,
                                       num_layers = self.num_layers,
                                       dim_feedforward = self.dim_feedforward,
                                       dropout = self.dropout)

        self.fc1 = nn.Linear(self.d_model, self.feat_dim)
        self.fc2 = nn.Linear(self.feat_dim, self.d_model)
        self.loss_fct = CrossEntropyLoss()
        # self.loss_fct = nn.MultiLabelSoftMarginLoss()
        self.f = lambda x: torch.exp(x / 0.5) # self.tau == 0.5

        if mode :
            self.dirpath = './checkpoints/' + datetime.now(timezone(offset=timedelta(hours=9))).strftime(
                "%y%b%d") + '_' + ''.join(
                str(datetime.now(timezone(offset=timedelta(hours=9)))).split()[1].split('.')[0].split(':'))
            print("Model save path:", self.dirpath)
        else :
            self.dirpath=None
        print(config)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1) #row 정규화
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def _update_matrix(
        self, between_sim, product_atom, reactant_atom
    ):
        '''
        뽑힌 reactant에 대해 -inf 값'''
        between_sim[int(product_atom),:] = -np.inf
        between_sim[:,int(reactant_atom)] = -np.inf
        return between_sim

    def _new_update_matrix(
        self, attention_multiplier_matrix, product_atom, reactant_atom, product_adj, reactant_adj, sim_matrix
    ):
        """Perform the "neighbor multiplier" step of the atom mapping similar to RXNMapper """
        product_neighbors = product_adj[int(product_atom)] == 1
        reactant_neighbors = reactant_adj[int(reactant_atom)] == 1
        
        attention_multiplier_matrix[np.ix_(product_neighbors,reactant_neighbors)] *= float(30)
        
        attention_multiplier_matrix[product_atom] = np.zeros(attention_multiplier_matrix.shape[1])
        attention_multiplier_matrix[:,reactant_atom] = np.zeros(attention_multiplier_matrix.shape[0])
        
        sim_matrix = np.multiply(
                    sim_matrix.cpu(),attention_multiplier_matrix #sim matrix 업데이트!
                )
        # row_sums = sim_matrix.sum(axis=1)
        # sim_matrix = np.divide(
        #     sim_matrix,
        #     row_sums[:, np.newaxis],
        #     out=np.zeros_like(sim_matrix),
        #     where=row_sums[:, np.newaxis] != 0,
        # )
        # sim_matrix = torch.tensor(sim_matrix)
        return sim_matrix

    def padding_tensor(self, batch_data):
        batch_size = batch_data.batch.max().item()+1
        atom_t = []
        # neighbor_t = []
        degrees = []
        im_hs = []
        ex_hs = []
        charges = []
        other_feats = []
        sp_t = []
        # charges=[]
        edge_types = []
        for i in range(batch_size):
            reactant_b_mask = batch_data.batch== i
            atom_t.append(batch_data['x'][reactant_b_mask])
            degrees.append(torch.tensor(batch_data['react_fts'][i][0]))
            im_hs.append(torch.tensor(batch_data['react_fts'][i][1]))
            ex_hs.append(torch.tensor(batch_data['react_fts'][i][2]))
            charges.append(torch.tensor(batch_data['react_fts'][i][3]))
            # neighbor_t.append(batch_data['neighbors'][reactant_b_mask])
            sp_t.append(IntTensor(batch_data['sp'][i]))
            edge_types.append(IntTensor(batch_data['edge_type'][i]))

        pad_atoms = pad_sequence(atom_t, True).unsqueeze(dim=0)
        device = pad_atoms.device
        pad_degrees = pad_sequence(degrees, True).to(device).unsqueeze(dim=0)
        pad_im_hs = pad_sequence(im_hs, True).to(device).unsqueeze(dim=0)
        pad_ex_hs = pad_sequence(ex_hs, True).to(device).unsqueeze(dim=0)
        pad_charges = pad_sequence(charges, True).to(device).unsqueeze(dim=0)

        x_feats = torch.cat((pad_atoms, pad_degrees, pad_im_hs, pad_ex_hs, pad_charges))
        # pad_bond = pad_sequence(neighbor_t, True)
        _, b, s = pad_atoms.shape
        tmp = zeros(b, s, s, dtype=torch.long)
        tmp2 = zeros(b, s, s, dtype=torch.long)
        
        tmp[:, :, 0] = 1  # prevent nan in MHA softmax on padding
        tmp2[:, :, 0] = 1  # prevent nan in MHA softmax on padding
        
        # 패딩 마스크 생성
        mask = (pad_atoms.squeeze() != 0)

        for i, d in enumerate(zip(sp_t,edge_types)):
            sp, et = d[0], d[1]
            r_size = sp.size(0)
            tmp[i, :r_size, :r_size] = sp
            tmp2[i, :r_size, :r_size] = et

        return x_feats, tmp.to(pad_atoms.device), tmp2.to(pad_atoms.device), mask
    
    def forward(self,**inputs):
        reactions = inputs['reactions']
        
        x_feats, reactant_tmp, reactant_et, reactant_mask = self.padding_tensor(reactions)
        # get Graphormer Encoder

        reactant_z = self.network(x_feats, reactant_tmp, reactant_et)
        # return reactant_z, product_z
        h1 = self.projection(reactant_z) #13681, 300 => 13681. 300

        return h1,reactant_mask

    def training_step(self, batch, batch_idx):
        h, padding_mask = self(**batch)
        
        batch_size = batch['reactions'].batch.max().item()+1
        device = h.device
        losses = []
        for b in range(batch_size):
            devide_index = int(batch['reactions'].devide[b])
            reaction = h[b][padding_mask[b]] #패딩 제외
            react_emb, product_emb = reaction[:devide_index], reaction[devide_index:] # devide token 단위로 나눔
            per_gt = batch['reactions'].y[b]
            assert len(per_gt) == product_emb.shape[0]
            sim_matrix = self.f(self.sim(product_emb, react_emb))
            if -1 in per_gt:
                indices = []
                for i,x in enumerate(per_gt):
                    if x != -1:
                        indices.append(i)
                per_gt = list(filter(lambda x: x != -1, per_gt))
                sim_matrix = sim_matrix[indices]
                assert sim_matrix.shape[0] == len(per_gt)
            per_gt = torch.tensor(per_gt).to(device)
            # sim_matrix = F.softmax(self.sim(product_h[batch_product_mask], reactant_h[batch_react_mask]), dim=1)

            loss = self.loss_fct(sim_matrix, per_gt)
            losses.append(loss)
        return {"loss": sum(losses) / len(losses)}

    def training_step_end(self, batch_parts):
        """
        Calculate loss and metrics after one train epoch ends.
        :param outputs: List of concatenated outputs of each batch. list[dict]
        """
        loss = batch_parts['loss']
   
        if self.logger:
            # log metrics
            self.log('train_loss', loss)
          
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        h, padding_mask = self(**batch)
        batch_size = batch['reactions'].batch.max().item()+1
        device = h.device
        losses = []
        preds = []
        gts = []
        confidences_list = []
        for b in range(batch_size):
            devide_index = int(batch['reactions'].devide[b])
            reaction = h[b][padding_mask[b]] #패딩 제외
            react_emb, product_emb = reaction[:devide_index], reaction[devide_index:] # devide token 단위로 나눔
            per_gt = batch['reactions'].y[b]
            assert len(per_gt) == product_emb.shape[0]
            sim_matrix = self.f(self.sim(product_emb, react_emb))
            if -1 in per_gt:
                indices = []
                for i,x in enumerate(per_gt):
                    if x != -1:
                        indices.append(i)
                loss_per_gt = list(filter(lambda x: x != -1, per_gt))
                loss_sim_matrix = sim_matrix[indices]
                loss_per_gt = torch.tensor(loss_per_gt).to(device)
                loss = self.loss_fct(loss_sim_matrix, loss_per_gt)
                per_gt = torch.tensor(per_gt).to(device)
                assert loss_sim_matrix.shape[0] == len(loss_per_gt)
            else:
                per_gt = torch.tensor(per_gt).to(device)
            # sim_matrix = F.softmax(self.sim(product_h[batch_product_mask], reactant_h[batch_react_mask]), dim=1)
                loss = self.loss_fct(sim_matrix, per_gt)
            
            losses.append(loss)
            
            with torch.no_grad():
                len_per_product_atom_list = product_emb.shape[0]
                pxr_mapping_vector = (np.ones(len_per_product_atom_list) * -1).astype(int) # (25,) 인 vector -1 
                confidences = np.ones(len_per_product_atom_list)
                
                sim_for_evaluation = sim_matrix.detach().cpu()
                for i in range(len_per_product_atom_list):
                    product_atom_to_map = torch.argmax(torch.max(sim_for_evaluation, dim=1)[0])
                    corresponding_reactant_atom = torch.argmax(sim_for_evaluation, axis=1)[product_atom_to_map]
                    confidence = float(torch.max(sim_for_evaluation))

                    pxr_mapping_vector[product_atom_to_map] = corresponding_reactant_atom
                    confidences[product_atom_to_map] = round(confidence,2)
                    sim_for_evaluation = self._update_matrix(
                        sim_for_evaluation, product_atom_to_map, corresponding_reactant_atom
                    )

                preds.append(pxr_mapping_vector)
                gts.append(per_gt)
                confidences_list.append(confidences)

    
        return {"loss": sum(losses) / len(losses),
                "preds": preds,
                "gts": gts,
                'confidences' : confidences_list}
            
    def validation_step_end(self, batch_parts):
        """
        Calculate loss and metrics after one train epoch ends.
        :param outputs: List of concatenated outputs of each batch. list[dict]
        """
        loss = batch_parts['loss']
        preds_total = batch_parts['preds']
        # confidences_list = batch_parts['confidences']

        gts_total = batch_parts['gts']

        c = 0
        total = 0
        ori_percents = []
        for p, g in zip(preds_total, gts_total):
            total+=1

            g = np.array(g.cpu())
            prediction = (p == g)
            #one solution
            percents = sum(prediction)/p.shape[0]
            ori_percents.append(percents)
            if percents == 1:
                c+=1

        correctness = c / total

        ori_percents = np.array(ori_percents)
        ori_percents = np.mean(ori_percents)

        if self.logger:
            # log metrics
            self.log('val_loss', loss)
            self.log('val_correctness', correctness)
            self.log('val_same_mapping', c)
            self.log('val_total', total)
            self.log('val_percent', percents)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        h, padding_mask = self(**batch)
        batch_size = batch['reactions'].batch.max().item()+1
        device = h.device
        preds = []
        gts = []
        confidences_list = []
        multiple_gts = []
        reactant_info=[]
        product_info=[]
        reactant_smiles = []
        product_smiles = []
        reactant_edge_info = []
        product_edge_info = []
        for b in range(batch_size):
            devide_index = int(batch['reactions'].devide[b])
            reaction = h[b][padding_mask[b]] #패딩 제외
            react_emb, product_emb = reaction[:devide_index], reaction[devide_index:] # devide token 단위로 나눔
            sim_matrix = self.f(self.sim(product_emb, react_emb))
            per_gt = batch['reactions'].y[b]
            per_gt = torch.tensor(per_gt).to(device)

            if len(per_gt) != product_emb.shape[0]:
                target_reactant,target_product = batch['reactions']['reaction_smiles'][b].split(">>")
                reactant_smiles.append(target_reactant)
                product_smiles.append(target_product)

                reactant_edge_info.append(None)
                product_edge_info.append(None)
            
                reactant_info.append(None)
                product_info.append(None)
                preds.append(None)
                gts.append(per_gt.tolist())
                confidences_list.append(None)
                multiple_gts.append(None)
            else:
                # reactant atom 별로 reactant 단위 group mapping
                target_reactant,target_product = batch['reactions']['reaction_smiles'][b].split(">>")
                product_adjaency = Chem.GetAdjacencyMatrix(Chem.MolFromSmiles(target_product))
                reactant_adjaency = Chem.GetAdjacencyMatrix(Chem.MolFromSmiles(target_reactant))
                attention_multiplier_matrix = np.ones_like(
                                            sim_matrix.cpu()
                                            ).astype(float)
                len_per_product_atom_list = product_emb.shape[0]
                pxr_mapping_vector = (np.ones(len_per_product_atom_list) * -1).astype(int) # (25,) 인 vector -1 
                confidences = np.ones(len_per_product_atom_list)
                
                sim_for_evaluation = sim_matrix.detach().cpu()
                #### 같은 원소는 무조건 매핑되게 ###
                batch_mask = batch['reactions'].batch == b
                types = batch['reactions'].x[batch_mask]
                react_types, product_types = types[:devide_index], types[devide_index:]
                atom_type_mask = np.zeros(sim_for_evaluation.shape)
                product_types = product_types.cpu()
                react_types = react_types.cpu()

                for j, atom_type in enumerate(product_types):
                    atom_type_mask[j, :] = (
                        np.array(react_types) == int(atom_type)
                    ).astype(int)
                sim_for_evaluation = np.multiply(sim_for_evaluation, atom_type_mask)
                ### #####################
                for i in range(len_per_product_atom_list):

                    product_atom_to_map = torch.argmax(torch.max(sim_for_evaluation, dim=1)[0])
                    corresponding_reactant_atom = torch.argmax(sim_for_evaluation, axis=1)[product_atom_to_map]
                    if product_types[product_atom_to_map] not in react_types: # product type이 reactant에 없는 경우 바로 break
                        corresponding_reactant_atom = -1
                        pxr_mapping_vector[product_atom_to_map] = corresponding_reactant_atom
                        confidences[product_atom_to_map] = 1.0
                        sim_for_evaluation[product_atom_to_map] = 0
                        continue
                    confidence = float(torch.max(sim_for_evaluation))
                    if np.isclose(confidence, 0.0):
                        confidence = 1.0
                        corresponding_reactant_atom = pxr_mapping_vector[
                            product_atom_to_map
                        ]  # either -1 or already mapped
                        break
                    pxr_mapping_vector[product_atom_to_map] = corresponding_reactant_atom
                    confidences[product_atom_to_map] = round(confidence,2)
                    # sim_for_evaluation = self._update_matrix(
                    #     sim_for_evaluation, product_atom_to_map, corresponding_reactant_atom
                    # )
                    sim_for_evaluation = self._new_update_matrix(
                        attention_multiplier_matrix, product_atom_to_map, 
                        corresponding_reactant_atom, product_adjaency, reactant_adjaency, sim_for_evaluation)
                
                reactant_smiles.append(target_reactant)
                product_smiles.append(target_product)

                reactant_edge_info.append(None)
                product_edge_info.append(None)
                
                reactant_info.append(None)
                product_info.append(None)
                preds.append(list(pxr_mapping_vector))
                gts.append(per_gt.tolist())
                confidences_list.append(list(confidences))
                multiple_gts.append(None)
   
        return {"pred" : preds,
                "gt": gts,
                'confidences_list' : confidences_list,
                'reactant_info' : reactant_info,
                'product_info' :product_info,
                'reactant_smiles':reactant_smiles,
                'product_smiles':product_smiles,
                'reactant_edge_info':reactant_edge_info,
                'product_edge_info':product_edge_info,
                'multiple_gts':multiple_gts
                }

    def configure_optimizers(self):
        """
        Configure optimizers and schedulers.
        Scheduler configuration can be changed in the config files (config/[model_name]), over train_params.
        """
        # optimizers
        optimizer_name = self.train_params['optimizer']['name']
        if optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.train_params['optimizer']['lr'])
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.train_params['optimizer']['lr'])  # , weight_decay=0.5)

        # schedulers
        # TODO: add schuler parameters restrictions to only implemented schdulers.
        scheduler_name = self.train_params['scheduler']['name']
        if scheduler_name == 'ExponentialLR':
            scheduler = ExponentialLR(optimizer, **self.train_params['scheduler']['params'])
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}

        elif scheduler_name == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, **self.train_params['scheduler']['params'])
            return {'optimizer': optimizer, 'lr_scheduler': scheduler, "monitor": "val_loss"}

        elif scheduler_name == 'OneCycleLR':
            scheduler = OneCycleLR(optimizer, **self.train_params['scheduler']['params'])
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}

        else:
            return optimizer

    def configure_callbacks(self):
        """
        Configure callbacks for torch lightning module. Refer to https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html.
        """
        checkpoint = model_checkpoint.ModelCheckpoint(dirpath=self.dirpath,
                                                      filename="{epoch:02d}-{val_correctness:.3f}",
                                                      monitor=self.train_params['monitor'],
                                                      mode=self.train_params['monitor_mode'],
                                                      verbose=True,
                                                      save_top_k=self.train_params['save_top_k'],
                                                      save_on_train_epoch_end=True,
                                                      save_last=True)
        
        earlystop = early_stopping.EarlyStopping(monitor=self.train_params['monitor'],
                                                 verbose=True,
                                                 mode=self.train_params['monitor_mode'],
                                                 patience=self.train_params['patience'])

        modelsummary = ModelSummary(max_depth=-1)

        callbacks = [checkpoint, earlystop, modelsummary]
        if self.logger:
            lr_monitor = LearningRateMonitor(logging_interval='step')
            callbacks.append(lr_monitor)
        return callbacks

    def load_my_state_dict(self, state_dict): #나중에 업데이트안되는 부분 찾아보기
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)



class RXN_div_Graphormer(pl.LightningModule):
    def __init__(
        self,
        model_params: Optional = None,
        train_params: Optional = None,
        config: Optional = None,
        logger: Optional = None,
        mode : Optional = False,
        multiple_solution : Optional = False,
    ):
        """
        Main pytorch lightning module for wMetric Learning task. Configuration files are read from "./configs'
        :param model_params: Model parameters configuration. [dict]
        :param train_params:  Training parameters configuration. [dict]
        :param dataset_params: Dataset parameters configuration. [dict]
        :param model_name: Name of the current model. [str]
        """
        super().__init__()
        self.save_hyperparameters()

        ## rxnmapper ##
        from transformers import AlbertModel
        import pkg_resources
        from rxnmapper.tokenization_smiles  import SmilesTokenizer

        self.model_path = config.get(
            "model_path",
            pkg_resources.resource_filename(
                "rxnmapper", "models/transformers/albert_heads_8_uspto_all_1310k"
            ),
        )
        self.model = AlbertModel.from_pretrained(
            self.model_path,
            output_attentions=True,
            output_past=False,
            output_hidden_states=False,
            return_dict=True,
        )
        vocab_path = os.path.join(self.model_path, "vocab.txt")
        tokenizer = SmilesTokenizer(vocab_path, max_len=self.model.config.max_position_embeddings)
        self.tokenizer = tokenizer
        ##################

        self.model_params = model_params
        self.train_params = train_params
        self.multiple_solution = multiple_solution

        self.num_layer = self.model_params['num_layers']
        self.feat_dim = self.model_params['feat_dim']
        self.drop_ratio = self.model_params['dropout']

        self.max_neighbors = self.model_params['max_neighbors']
        self.max_distance = self.model_params['max_distance']
        self.d_model = self.model_params['d_model']
        self.n_head = self.model_params['n_head']
        self.num_layers = self.model_params['num_layers']
        self.dim_feedforward = self.model_params['dim_feedforward']
        self.dropout = self.model_params['dropout']

        print(config)

        self.network1 = RXNMoleculeEncoder(max_neighbors = self.max_neighbors, 
                                       max_distance = self.max_distance, 
                                       shared_weights=True,
                                       d_model = 256, 
                                       nhead = self.n_head,
                                       num_layers = self.num_layers,
                                       dim_feedforward = self.dim_feedforward,
                                       dropout = self.dropout)

        self.network2 = RXNMoleculeEncoder(max_neighbors = self.max_neighbors, 
                                            max_distance = self.max_distance, 
                                            shared_weights=True,
                                            d_model = 256, 
                                            nhead = self.n_head,
                                            num_layers = self.num_layers,
                                            dim_feedforward = self.dim_feedforward,
                                            dropout = self.dropout)

        self.fc1 = nn.Linear(self.d_model, self.feat_dim)
        self.fc2 = nn.Linear(self.feat_dim, self.d_model)
        self.loss_fct = CrossEntropyLoss()
        # self.loss_fct = nn.MultiLabelSoftMarginLoss()
        self.f = lambda x: torch.exp(x / 0.5) # self.tbert_outputau == 0.5

        if mode :
            self.dirpath = './checkpoints/' + datetime.now(timezone(offset=timedelta(hours=9))).strftime(
                "%y%b%d") + '_' + ''.join(
                str(datetime.now(timezone(offset=timedelta(hours=9)))).split()[1].split('.')[0].split(':'))
            print("Model save path:", self.dirpath)
        else :
            self.dirpath=None
        print(config)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1) #row 정규화
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def _update_matrix(
        self, between_sim, product_atom, reactant_atom
    ):
        '''
        뽑힌 reactant에 대해 -inf 값을 주자'''
        between_sim[int(product_atom),:] = -np.inf
        between_sim[:,int(reactant_atom)] = -np.inf
        return between_sim

    def _new_update_matrix(
        self, attention_multiplier_matrix, product_atom, reactant_atom, product_adj, reactant_adj, sim_matrix
    ):
        """Perform the "neighbor multiplier" step of the atom mapping similar to RXNMapper """
        product_neighbors = product_adj[int(product_atom)] == 1
        reactant_neighbors = reactant_adj[int(reactant_atom)] == 1
        
        attention_multiplier_matrix[np.ix_(product_neighbors,reactant_neighbors)] *= float(30)
        
        attention_multiplier_matrix[product_atom] = np.zeros(attention_multiplier_matrix.shape[1])
        attention_multiplier_matrix[:,reactant_atom] = np.zeros(attention_multiplier_matrix.shape[0])
        
        sim_matrix = np.multiply(
                    sim_matrix.cpu(),attention_multiplier_matrix #sim matrix 업데이트!
                )
        row_sums = sim_matrix.sum(axis=1)
        sim_matrix = np.divide(
            sim_matrix,
            row_sums[:, np.newaxis],
            out=np.zeros_like(sim_matrix),
            where=row_sums[:, np.newaxis] != 0,
        )
        sim_matrix = torch.tensor(sim_matrix)
        return sim_matrix
    
    def padding_tensor(self, batch_data):
        batch_size = batch_data.batch.max().item()+1
        atom_t = []
        neighbor_t = []
        sp_t = []
        for i in range(batch_size):
            reactant_b_mask = batch_data.batch== i
            atom_t.append(batch_data['x'][reactant_b_mask])
            neighbor_t.append(batch_data['neighbors'][reactant_b_mask])
            sp_t.append(IntTensor(batch_data['sp'][i]))
        
        pad_atoms = pad_sequence(atom_t, True)
        pad_neigh = pad_sequence(neighbor_t, True)
        b, s = pad_atoms.shape
        tmp = zeros(b, s, s, dtype=torch.long)
        tmp[:, :, 0] = 1 

        # 패딩 마스크 생성
        mask = (pad_atoms != 0)

        for i, d in enumerate(sp_t):
            r_size = d.size(0)
            tmp[i, :r_size, :r_size] = d

        return pad_atoms, pad_neigh, tmp.to(pad_atoms.device), mask
    
    def RXN_embedding(self, batch_data):
        bert_input = batch_data.bert_input
        # full_bert_input = batch_data.full_bert_input
        bert_input = ["".join(i) for i in bert_input]
        
        bert_input = self.tokenizer.batch_encode_plus(
            bert_input,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        bert_input.to('cuda')
        output = self.model(**bert_input)
        return output[0]
    
    def forward(self,**inputs):
        reactant = inputs['reactant']
        product = inputs['product']
        
        _, reactant_neigh, reactant_tmp, reactant_mask = self.padding_tensor(reactant)
        _, product_neigh, product_tmp, product_mask = self.padding_tensor(product)

        RXN_reactant = self.RXN_embedding(reactant)
        RXN_product = self.RXN_embedding(product)        

        # get Graphormer Encoder
        reactant_z = self.network1(RXN_reactant, reactant_neigh, reactant_tmp)
        product_z = self.network2(RXN_product, product_neigh, product_tmp)

        # return reactant_z, product_z
        h1 = self.projection(reactant_z) #13681, 300 => 13681. 300
        h2 = self.projection(product_z)

        return h1, h2, reactant_mask, product_mask

    def training_step(self, batch, batch_idx):
        reactant_h, product_h, reactant_mask, product_mask = self(**batch)
        
        batch_size = batch['product'].batch.max().item()+1
        
        device = reactant_h.device
        losses = []
        preds = []
        gts = []
        multiple_gts = []
        confidences_list = []
        if self.multiple_solution == False:
            for b in range(batch_size):
                batch_react_mask = (batch['reactant']['batch'] == b)
                batch_product_mask = (batch['product']['batch'] == b)
                per_gt = batch['product'].y[batch_product_mask]
                                
                sim_matrix = self.f(self.sim(product_h[b][product_mask[b]], reactant_h[b][reactant_mask[b]]))
                # sim_matrix = F.softmax(self.sim(product_h[batch_product_mask], reactant_h[batch_react_mask]), dim=1)
                loss = self.loss_fct(sim_matrix, per_gt)
                losses.append(loss)
            return {"loss": sum(losses) / len(losses)}
        else:
            #TODO not yet multiple solution
            for b in range(batch_size):
                batch_react_mask = (batch['reactant']['batch'] == b)
                batch_product_mask = (batch['product']['batch'] == b)
                
                per_gt = batch['product'].multiple_y[b]
                per_gt = torch.tensor(per_gt, device=device)
                sim_matrix = self.f(self.sim(product_h[batch_product_mask], reactant_h[batch_react_mask]))
                # sim_matrix = F.softmax(self.sim(product_h[batch_product_mask], reactant_h[batch_react_mask]), dim=1)
                # import pdb; pdb.set_trace()
                loss = self.loss_fct(sim_matrix, per_gt) #cross entropy loss

                losses.append(loss)

        
            return {"loss": sum(losses) / len(losses)}

    def training_step_end(self, batch_parts):
        """
        Calculate loss and metrics after one train epoch ends.
        :param outputs: List of concatenated outputs of each batch. list[dict]
        """
        loss = batch_parts['loss']
   
        if self.logger:
            # log metrics
            self.log('train_loss', loss)
          
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        reactant_h, product_h, reactant_mask, product_mask = self(**batch)
        batch_size = batch['product'].batch.max().item()+1
        # f = lambda x: torch.exp(x / 0.5) # self.tau == 0.5

        device = reactant_h.device
        losses = []
        preds = []
        gts = []
        confidences_list = []
        multiple_gts = []
        for b in range(batch_size):
            batch_react_mask = (batch['reactant']['batch'] == b)
            batch_product_mask = (batch['product']['batch'] == b)
            
            if self.multiple_solution == False:
                per_gt = batch['product'].y[batch_product_mask]
                per_multiple_gt = batch['product'].multiple_y[b]
                sim_matrix = self.f(self.sim(product_h[b][product_mask[b]], reactant_h[b][reactant_mask[b]]))
                # sim_matrix = F.softmax(self.sim(product_h[batch_product_mask], reactant_h[batch_react_mask]))
                loss = self.loss_fct(sim_matrix, torch.tensor(per_gt, device=device))

                losses.append(loss)
            else:
                #TODO : not yet
                per_multiple_gt = batch['product'].multiple_y[b]

                sim_matrix = self.f(self.sim(product_h[batch_product_mask], reactant_h[batch_react_mask]))
                # sim_matrix = F.softmax(self.sim(product_h[batch_product_mask], reactant_h[batch_react_mask]))
                # sim_matrix = F.softmax(self.sim(product_h[batch_product_mask], reactant_h[batch_react_mask]))
                loss = self.loss_fct(sim_matrix, torch.tensor(per_multiple_gt, device=device))

                losses.append(loss)
                
            with torch.no_grad():
                len_per_product_atom_list = batch['product'].x[batch_product_mask].shape[0]
                pxr_mapping_vector = (np.ones(len_per_product_atom_list) * -1).astype(int) # (25,) 인 vector -1 
                confidences = np.ones(len_per_product_atom_list)
                
                sim_for_evaluation = sim_matrix.detach().cpu()
                for i in range(len_per_product_atom_list):
                    product_atom_to_map = torch.argmax(torch.max(sim_for_evaluation, dim=1)[0])
                    corresponding_reactant_atom = torch.argmax(sim_for_evaluation, axis=1)[product_atom_to_map]
                    confidence = float(torch.max(sim_for_evaluation))

                    pxr_mapping_vector[product_atom_to_map] = corresponding_reactant_atom
                    confidences[product_atom_to_map] = round(confidence,2)
                    sim_for_evaluation = self._update_matrix(
                        sim_for_evaluation, product_atom_to_map, corresponding_reactant_atom
                    )

                preds.append(pxr_mapping_vector)
                gts.append(per_gt)
                multiple_gts.append(per_multiple_gt)
                confidences_list.append(confidences)

    
        return {"loss": sum(losses) / len(losses),
                "preds": preds,
                "gts": gts,
                "multiple_gts": multiple_gts,
                'confidences' : confidences_list}
            
    def validation_step_end(self, batch_parts):
        """
        Calculate loss and metrics after one train epoch ends.
        :param outputs: List of concatenated outputs of each batch. list[dict]
        """
        loss = batch_parts['loss']
        preds_total = batch_parts['preds']
        # confidences_list = batch_parts['confidences']

        gts_total = batch_parts['gts']
        multiple_gts = batch_parts['multiple_gts']
        
        c = 0
        iso_c = 0
        total = 0
        ori_percents = []
        iso_percents = []
        for p, g, multiple in zip(preds_total, gts_total, multiple_gts):
            total+=1
            iso_g_arr = list()
            multiple = torch.tensor(multiple)
            iso_list = multiple.sum(dim=1)>1
            iso_list = iso_list.tolist()
            g = np.array(g.cpu())
            prediction = (p == g)
            #one solution
            percents = sum(prediction)/p.shape[0]
            ori_percents.append(percents)
            if percents == 1:
                c+=1
            #multiple solution
            for index,bool in enumerate(prediction): # True, True, ... False, ..True (다른 값들 return)
                if (bool == False) & (iso_list[index] ==True): # 잘못 예측한 값중에 iso한 값들이 존재한다면,
                    if p[index] in torch.where(multiple[index]==1)[0]: # 실제 예측 값이 iso 한 값들 중 존재한다면
                        symmetric = True
                        iso_g_arr.append(symmetric)
                        # print('Prediction:',p_arr[index], "GT:",g_arr[index],"iso_map", iso[p_arr[index]])
                    else:
                        iso_g_arr.append(bool) #isomorphic 안 되는 경우
                else:
                    iso_g_arr.append(bool)
            if sum(iso_g_arr) == len(iso_g_arr):
                iso_c +=1
            iso_percents.append(sum(iso_g_arr) / len(iso_g_arr))
            # sum(iso_g_arr) / len(iso_g_arr)
        correctness = c / total
        iso_correctness = iso_c / total
        ori_percents = np.array(ori_percents)
        ori_percents = np.mean(ori_percents)
        iso_percents = np.array(iso_percents)
        iso_percents = np.mean(iso_percents)
        if self.logger:
            # log metrics
            self.log('val_loss', loss)
            self.log('val_correctness', correctness)
            self.log('val_same_mapping', c)
            self.log('val_total', total)
            self.log('val_percent', percents)
            self.log('val_iso_correctness', iso_correctness)
            self.log('val_iso_percents', iso_percents)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        reactant_h, product_h, reactant_mask, product_mask = self(**batch)
        batch_size = batch['product'].batch.max().item()+1
        f = lambda x: torch.exp(x / 0.5) # self.tau == 0.5

        preds = []
        gts = []
        confidences_list = []
        multiple_gts = []
        reactant_info=[]
        product_info=[]
        reactant_smiles = []
        product_smiles = []
        reactant_edge_info = []
        product_edge_info = []
        for b in range(batch_size):
            batch_react_mask = (batch['reactant']['batch'] == b)
            batch_product_mask = (batch['product']['batch'] == b)
            per_gt = batch['product'].y[batch_product_mask]
            # per_multiple_gt = batch['product'].multiple_y[b]
            sim_matrix = self.f(self.sim(product_h[b][product_mask[b]], reactant_h[b][reactant_mask[b]]))

            # reactant atom 별로 reactant 단위 group mapping
            target_reactant = batch['reactant']['smiles'][b]
            target_product = batch['product']['smiles'][b]
            product_adjaency = Chem.GetAdjacencyMatrix(Chem.MolFromSmiles(target_product))
            reactant_adjaency = Chem.GetAdjacencyMatrix(Chem.MolFromSmiles(target_reactant))
            attention_multiplier_matrix = np.ones_like(
                                        sim_matrix.cpu()
                                        ).astype(float)
            
            
            len_per_product_atom_list = batch['product'].x[batch_product_mask].shape[0]
            pxr_mapping_vector = (np.ones(len_per_product_atom_list) * -1).astype(int) # (25,) 인 vector -1 
            confidences = np.ones(len_per_product_atom_list)
            
            sim_for_evaluation = sim_matrix.detach().cpu()
            
            for i in range(len_per_product_atom_list):
                product_atom_to_map = torch.argmax(torch.max(sim_for_evaluation, dim=1)[0])
                corresponding_reactant_atom = torch.argmax(sim_for_evaluation, axis=1)[product_atom_to_map]
                confidence = float(torch.max(sim_for_evaluation))

                pxr_mapping_vector[product_atom_to_map] = corresponding_reactant_atom
                confidences[product_atom_to_map] = round(confidence,2)
                # sim_for_evaluation = self._new_update_matrix(
                #     attention_multiplier_matrix, product_atom_to_map, 
                #     corresponding_reactant_atom, product_adjaency, reactant_adjaency, sim_for_evaluation)
                sim_for_evaluation = self._update_matrix(
                        sim_for_evaluation, product_atom_to_map, corresponding_reactant_atom
                    )

            reactant_smiles.append(batch['reactant']['smiles'][b])
            product_smiles.append(batch['product']['smiles'][b])

            reactant_edge_info.append(batch['reactant']['inference_edge_csv'][b])
            product_edge_info.append(batch['product']['inference_edge_csv'][b])
            
            reactant_info.append(batch['reactant'].x[batch_react_mask].tolist())
            product_info.append(batch['product'].x[batch_product_mask].tolist())
            preds.append(list(pxr_mapping_vector))
            gts.append(per_gt.tolist())
            confidences_list.append(list(confidences))
            multiple_gts.append(None)
   
        return {"pred" : preds,
                "gt": gts,
                'confidences_list' : confidences_list,
                'reactant_info' : reactant_info,
                'product_info' :product_info,
                'reactant_smiles':reactant_smiles,
                'product_smiles':product_smiles,
                'reactant_edge_info':reactant_edge_info,
                'product_edge_info':product_edge_info,
                'multiple_gts':multiple_gts
                }

    def configure_optimizers(self):
        """
        Configure optimizers and schedulers.
        Scheduler configuration can be changed in the config files (config/[model_name]), over train_params.
        """
        # optimizers
        optimizer_name = self.train_params['optimizer']['name']
        if optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.train_params['optimizer']['lr'])
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.train_params['optimizer']['lr'])  # , weight_decay=0.5)

        # schedulers
        # TODO: add schuler parameters restrictions to only implemented schdulers.
        scheduler_name = self.train_params['scheduler']['name']
        if scheduler_name == 'ExponentialLR':
            scheduler = ExponentialLR(optimizer, **self.train_params['scheduler']['params'])
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}

        elif scheduler_name == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, **self.train_params['scheduler']['params'])
            return {'optimizer': optimizer, 'lr_scheduler': scheduler, "monitor": "val_loss"}

        elif scheduler_name == 'OneCycleLR':
            scheduler = OneCycleLR(optimizer, **self.train_params['scheduler']['params'])
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}

        else:
            return optimizer

    def configure_callbacks(self):
        """
        Configure callbacks for torch lightning module. Refer to https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html.
        """
        checkpoint = model_checkpoint.ModelCheckpoint(dirpath=self.dirpath,
                                                      filename="{epoch:02d}-{val_correctness:.3f}",
                                                      monitor=self.train_params['monitor'],
                                                      mode=self.train_params['monitor_mode'],
                                                      verbose=True,
                                                      save_top_k=self.train_params['save_top_k'],
                                                      save_on_train_epoch_end=False,
                                                      save_last=True)

        earlystop = early_stopping.EarlyStopping(monitor=self.train_params['monitor'],
                                                 verbose=True,
                                                 mode=self.train_params['monitor_mode'],
                                                 patience=self.train_params['patience'])

        modelsummary = ModelSummary(max_depth=-1)

        callbacks = [checkpoint, earlystop, modelsummary]
        if self.logger:
            lr_monitor = LearningRateMonitor(logging_interval='step')
            callbacks.append(lr_monitor)
        return callbacks

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./ckpt/pretrained_gcn/checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            # model.load_state_dict(state_dict)
            model.load_my_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model


    def load_my_state_dict(self, state_dict): #나중에 업데이트안되는 부분 찾아보기
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)


class Full_Graphormer_loss(pl.LightningModule):
    def __init__(
        self,
        model_params: Optional = None,
        train_params: Optional = None,
        config: Optional = None,
        logger: Optional = None,
        mode : Optional = False,
        multiple_solution : Optional = False,
    ):
        """
        Main pytorch lightning module for wMetric Learning task. Configuration files are read from "./configs'
        :param model_params: Model parameters configuration. [dict]
        :param train_params:  Training parameters configuration. [dict]
        :param dataset_params: Dataset parameters configuration. [dict]
        :param model_name: Name of the current model. [str]
        """
        super().__init__()
        self.save_hyperparameters()

        self.model_params = model_params
        self.train_params = train_params
        self.multiple_solution = multiple_solution

        self.feat_dim = self.model_params['feat_dim']

        self.max_neighbors = self.model_params['max_neighbors']
        self.max_distance = self.model_params['max_distance']
        self.d_model = self.model_params['d_model']
        self.n_head = self.model_params['n_head']
        self.num_layers = self.model_params['num_layers']
        self.dim_feedforward = self.model_params['dim_feedforward']
        self.dropout = self.model_params['dropout']

        print(config)

        self.network = MoleculeEncoder(max_neighbors = self.max_neighbors, 
                                       max_distance = self.max_distance, 
                                       shared_weights=True,
                                       d_model = self.d_model, 
                                       nhead = self.n_head,
                                       num_layers = self.num_layers,
                                       dim_feedforward = self.dim_feedforward,
                                       dropout = self.dropout)

        self.fc1 = nn.Linear(self.d_model, self.feat_dim)
        self.fc2 = nn.Linear(self.feat_dim, self.d_model)
        self.loss_fct = CrossEntropyLoss()
        # self.loss_fct = nn.MultiLabelSoftMarginLoss()
        self.f = lambda x: torch.exp(x / 0.5) # self.tau == 0.5

        if mode :
            self.dirpath = './checkpoints/' + datetime.now(timezone(offset=timedelta(hours=9))).strftime(
                "%y%b%d") + '_' + ''.join(
                str(datetime.now(timezone(offset=timedelta(hours=9)))).split()[1].split('.')[0].split(':'))
            print("Model save path:", self.dirpath)
        else :
            self.dirpath=None
        print(config)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1) #row 정규화
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def _update_matrix(
        self, between_sim, product_atom, reactant_atom
    ):
        '''
        뽑힌 reactant에 대해 -inf 값'''
        between_sim[int(product_atom),:] = -np.inf
        between_sim[:,int(reactant_atom)] = -np.inf
        return between_sim

    def _new_update_matrix(
        self, attention_multiplier_matrix, product_atom, reactant_atom, product_adj, reactant_adj, sim_matrix
    ):
        """Perform the "neighbor multiplier" step of the atom mapping similar to RXNMapper """
        product_neighbors = product_adj[int(product_atom)] == 1
        reactant_neighbors = reactant_adj[int(reactant_atom)] == 1
        
        attention_multiplier_matrix[np.ix_(product_neighbors,reactant_neighbors)] *= float(30)
        
        attention_multiplier_matrix[product_atom] = np.zeros(attention_multiplier_matrix.shape[1])
        attention_multiplier_matrix[:,reactant_atom] = np.zeros(attention_multiplier_matrix.shape[0])
        
        sim_matrix = np.multiply(
                    sim_matrix.cpu(),attention_multiplier_matrix #sim matrix 업데이트!
                )
        row_sums = sim_matrix.sum(axis=1)
        sim_matrix = np.divide(
            sim_matrix,
            row_sums[:, np.newaxis],
            out=np.zeros_like(sim_matrix),
            where=row_sums[:, np.newaxis] != 0,
        )
        sim_matrix = torch.tensor(sim_matrix)
        return sim_matrix

    def padding_tensor(self, batch_data):
        batch_size = batch_data.batch.max().item()+1
        atom_t = []
        neighbor_t = []
        sp_t = []
        charges=[]
        for i in range(batch_size):
            reactant_b_mask = batch_data.batch== i
            atom_t.append(batch_data['x'][reactant_b_mask])
            neighbor_t.append(batch_data['neighbors'][reactant_b_mask])
            sp_t.append(IntTensor(batch_data['sp'][i]))
            charges.append(batch_data['x_charges'][reactant_b_mask])
        
        pad_atoms = pad_sequence(atom_t, True)
        pad_neigh = pad_sequence(neighbor_t, True)
        pad_charges = pad_sequence(charges, True)
        
        # pad_bond = pad_sequence(neighbor_t, True)
        b, s = pad_atoms.shape
        tmp = zeros(b, s, s, dtype=torch.long)
        tmp[:, :, 0] = 1  # prevent nan in MHA softmax on padding

        # 패딩 마스크 생성
        mask = (pad_atoms != 0)

        for i, d in enumerate(sp_t):
            r_size = d.size(0)
            tmp[i, :r_size, :r_size] = d

        return pad_atoms, pad_neigh, pad_charges, tmp.to(pad_atoms.device), mask
    
    def forward(self,**inputs):
        reactions = inputs['reactions']
        
        reactant_atoms, reactant_neigh, pad_charges, reactant_tmp, reactant_mask = self.padding_tensor(reactions)
        # get Graphormer Encoder
        reactant_z = self.network(reactant_atoms, reactant_neigh, pad_charges, reactant_tmp)
        # return reactant_z, product_z
        h1 = self.projection(reactant_z) #13681, 300 => 13681. 300

        return h1,reactant_mask

    def training_step(self, batch, batch_idx):
        h, padding_mask = self(**batch)
        
        batch_size = batch['reactions'].batch.max().item()+1
        device = h.device
        losses = []
        preds = []
        gts = []
        multiple_gts = []
        confidences_list = []
        for b in range(batch_size):
            devide_index = int(batch['reactions'].devide[b])
            reaction = h[b][padding_mask[b]] #패딩 제외
            react_emb, product_emb = reaction[:devide_index], reaction[devide_index:] # devide token 단위로 나눔
            per_gt = batch['reactions'].y[b]
            per_multiples = batch['reactions'].multiple_y[b]
            positive_mask = torch.tensor(np.array(per_multiples).astype(bool)).to(device)
            negative_mask = ~positive_mask
            
            if len(per_gt) != product_emb.shape[0]:
                print(len(per_gt), product_emb.shape[0])
            sim_matrix = self.f(self.sim(product_emb, react_emb))
            per_gt = torch.tensor(per_gt).to(device)
            # sim_matrix = F.softmax(self.sim(product_h[batch_product_mask], reactant_h[batch_react_mask]), dim=1)
            neg_output = (sim_matrix*negative_mask).sum(1)
            pos_output = (sim_matrix*positive_mask).sum(1)
            # loss = -torch.log((pos_output / (pos_output+neg_output))).mean()
            losses.append(-torch.log((pos_output / (pos_output+neg_output))))

        l1 = torch.cat(losses)
        return {"loss": l1.mean()}

    def training_step_end(self, batch_parts):
        """
        Calculate loss and metrics after one train epoch ends.
        :param outputs: List of concatenated outputs of each batch. list[dict]
        """
        loss = batch_parts['loss']
   
        if self.logger:
            # log metrics
            self.log('train_loss', loss)
          
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        h, padding_mask = self(**batch)
        batch_size = batch['reactions'].batch.max().item()+1
        device = h.device
        losses = []
        preds = []
        gts = []
        confidences_list = []
        multiple_gts = []
        for b in range(batch_size):
            devide_index = int(batch['reactions'].devide[b])
            reaction = h[b][padding_mask[b]] #패딩 제외
            react_emb, product_emb = reaction[:devide_index], reaction[devide_index:] # devide token 단위로 나눔
            per_gt = batch['reactions'].y[b]
            per_multiples = batch['reactions'].multiple_y[b]
            positive_mask = torch.tensor(np.array(per_multiples).astype(bool)).to(device)
            negative_mask = ~positive_mask
            
            assert len(per_gt) == product_emb.shape[0]
            sim_matrix = self.f(self.sim(product_emb, react_emb))
            per_gt = torch.tensor(per_gt).to(device)
            # sim_matrix = F.softmax(self.sim(product_h[batch_product_mask], reactant_h[batch_react_mask]), dim=1)
            neg_output = (sim_matrix*negative_mask).sum(1)
            pos_output = (sim_matrix*positive_mask).sum(1)
            # loss = -torch.log((pos_output / (pos_output+neg_output))).mean()
            losses.append(-torch.log((pos_output / (pos_output+neg_output))))
            
            with torch.no_grad():
                len_per_product_atom_list = product_emb.shape[0]
                pxr_mapping_vector = (np.ones(len_per_product_atom_list) * -1).astype(int) # (25,) 인 vector -1 
                confidences = np.ones(len_per_product_atom_list)
                
                sim_for_evaluation = sim_matrix.detach().cpu()
                for i in range(len_per_product_atom_list):
                    product_atom_to_map = torch.argmax(torch.max(sim_for_evaluation, dim=1)[0])
                    corresponding_reactant_atom = torch.argmax(sim_for_evaluation, axis=1)[product_atom_to_map]
                    confidence = float(torch.max(sim_for_evaluation))

                    pxr_mapping_vector[product_atom_to_map] = corresponding_reactant_atom
                    confidences[product_atom_to_map] = round(confidence,2)
                    sim_for_evaluation = self._update_matrix(
                        sim_for_evaluation, product_atom_to_map, corresponding_reactant_atom
                    )

                preds.append(pxr_mapping_vector)
                gts.append(per_gt)
                multiple_gts.append(batch['reactions'].multiple_y[b])
                confidences_list.append(confidences)

        l1 = torch.cat(losses)
        return {"loss": l1.mean(),
                "preds": preds,
                "gts": gts,
                "multiple_gts": multiple_gts,
                'confidences' : confidences_list}
            
    def validation_step_end(self, batch_parts):
        """
        Calculate loss and metrics after one train epoch ends.
        :param outputs: List of concatenated outputs of each batch. list[dict]
        """
        loss = batch_parts['loss']
        preds_total = batch_parts['preds']
        # confidences_list = batch_parts['confidences']

        gts_total = batch_parts['gts']
        multiple_gts = batch_parts['multiple_gts']
        
        c = 0
        iso_c = 0
        total = 0
        ori_percents = []
        iso_percents = []
        for p, g, multiple in zip(preds_total, gts_total, multiple_gts):
            total+=1
            iso_g_arr = list()
            multiple = torch.tensor(multiple)
            iso_list = multiple.sum(dim=1)>1
            iso_list = iso_list.tolist()
            g = np.array(g.cpu())
            prediction = (p == g)
            #one solution
            percents = sum(prediction)/p.shape[0]
            ori_percents.append(percents)
            if percents == 1:
                c+=1
            #multiple solution
            for index,bool in enumerate(prediction): # True, True, ... False, ..True (다른 값들 return)
                if (bool == False) & (iso_list[index] ==True): # 잘못 예측한 값중에 iso한 값들이 존재한다면,
                    if p[index] in torch.where(multiple[index]==1)[0]: # 실제 예측 값이 iso 한 값들 중 존재한다면
                        symmetric = True
                        iso_g_arr.append(symmetric)
                        # print('Prediction:',p_arr[index], "GT:",g_arr[index],"iso_map", iso[p_arr[index]])
                    else:
                        iso_g_arr.append(bool) #isomorphic 안 되는 경우
                else:
                    iso_g_arr.append(bool)
            if sum(iso_g_arr) == len(iso_g_arr):
                iso_c +=1
            iso_percents.append(sum(iso_g_arr) / len(iso_g_arr))
            # sum(iso_g_arr) / len(iso_g_arr)
        correctness = c / total
        iso_correctness = iso_c / total
        ori_percents = np.array(ori_percents)
        ori_percents = np.mean(ori_percents)
        iso_percents = np.array(iso_percents)
        iso_percents = np.mean(iso_percents)
        if self.logger:
            # log metrics
            self.log('val_loss', loss)
            self.log('val_correctness', correctness)
            self.log('val_same_mapping', c)
            self.log('val_total', total)
            self.log('val_percent', percents)
            self.log('val_iso_correctness', iso_correctness)
            self.log('val_iso_percents', iso_percents)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        h, padding_mask = self(**batch)
        batch_size = batch['reactions'].batch.max().item()+1
        device = h.device
        preds = []
        gts = []
        confidences_list = []
        multiple_gts = []
        reactant_info=[]
        product_info=[]
        reactant_smiles = []
        product_smiles = []
        reactant_edge_info = []
        product_edge_info = []
        for b in range(batch_size):
            devide_index = int(batch['reactions'].devide[b])
            reaction = h[b][padding_mask[b]] #패딩 제외
            react_emb, product_emb = reaction[:devide_index], reaction[devide_index:] # devide token 단위로 나눔
            sim_matrix = self.f(self.sim(product_emb, react_emb))
            per_gt = batch['reactions'].y[b]
            assert len(per_gt) == product_emb.shape[0]
            sim_matrix = self.f(self.sim(product_emb, react_emb))
            per_gt = torch.tensor(per_gt).to(device)

            # reactant atom 별로 reactant 단위 group mapping
            target_reactant,target_product = batch['reactions']['reaction_smiles'][b].split(">>")
            product_adjaency = Chem.GetAdjacencyMatrix(Chem.MolFromSmiles(target_product))
            reactant_adjaency = Chem.GetAdjacencyMatrix(Chem.MolFromSmiles(target_reactant))
            attention_multiplier_matrix = np.ones_like(
                                        sim_matrix.cpu()
                                        ).astype(float)
            len_per_product_atom_list = product_emb.shape[0]
            pxr_mapping_vector = (np.ones(len_per_product_atom_list) * -1).astype(int) # (25,) 인 vector -1 
            confidences = np.ones(len_per_product_atom_list)
            
            sim_for_evaluation = sim_matrix.detach().cpu()
            for i in range(len_per_product_atom_list):
                product_atom_to_map = torch.argmax(torch.max(sim_for_evaluation, dim=1)[0])
                corresponding_reactant_atom = torch.argmax(sim_for_evaluation, axis=1)[product_atom_to_map]
                confidence = float(torch.max(sim_for_evaluation))

                pxr_mapping_vector[product_atom_to_map] = corresponding_reactant_atom
                confidences[product_atom_to_map] = round(confidence,2)
                # sim_for_evaluation = self._update_matrix(
                #     sim_for_evaluation, product_atom_to_map, corresponding_reactant_atom
                # )
                sim_for_evaluation = self._new_update_matrix(
                    attention_multiplier_matrix, product_atom_to_map, 
                    corresponding_reactant_atom, product_adjaency, reactant_adjaency, sim_for_evaluation)
            
            reactant_smiles.append(target_reactant)
            product_smiles.append(target_product)

            reactant_edge_info.append(None)
            product_edge_info.append(None)
            
            reactant_info.append(None)
            product_info.append(None)
            preds.append(list(pxr_mapping_vector))
            gts.append(per_gt.tolist())
            confidences_list.append(list(confidences))
            multiple_gts.append(None)
   
        return {"pred" : preds,
                "gt": gts,
                'confidences_list' : confidences_list,
                'reactant_info' : reactant_info,
                'product_info' :product_info,
                'reactant_smiles':reactant_smiles,
                'product_smiles':product_smiles,
                'reactant_edge_info':reactant_edge_info,
                'product_edge_info':product_edge_info,
                'multiple_gts':multiple_gts
                }

    def configure_optimizers(self):
        """
        Configure optimizers and schedulers.
        Scheduler configuration can be changed in the config files (config/[model_name]), over train_params.
        """
        # optimizers
        optimizer_name = self.train_params['optimizer']['name']
        if optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.train_params['optimizer']['lr'])
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.train_params['optimizer']['lr'])  # , weight_decay=0.5)

        # schedulers
        # TODO: add schuler parameters restrictions to only implemented schdulers.
        scheduler_name = self.train_params['scheduler']['name']
        if scheduler_name == 'ExponentialLR':
            scheduler = ExponentialLR(optimizer, **self.train_params['scheduler']['params'])
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}

        elif scheduler_name == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, **self.train_params['scheduler']['params'])
            return {'optimizer': optimizer, 'lr_scheduler': scheduler, "monitor": "val_loss"}

        elif scheduler_name == 'OneCycleLR':
            scheduler = OneCycleLR(optimizer, **self.train_params['scheduler']['params'])
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}

        else:
            return optimizer

    def configure_callbacks(self):
        """
        Configure callbacks for torch lightning module. Refer to https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html.
        """
        checkpoint = model_checkpoint.ModelCheckpoint(dirpath=self.dirpath,
                                                      filename="{epoch:02d}-{val_correctness:.3f}",
                                                      monitor=self.train_params['monitor'],
                                                      mode=self.train_params['monitor_mode'],
                                                      verbose=True,
                                                      save_top_k=self.train_params['save_top_k'],
                                                      save_on_train_epoch_end=False,
                                                      save_last=True)
    
        
        earlystop = early_stopping.EarlyStopping(monitor=self.train_params['monitor'],
                                                 verbose=True,
                                                 mode=self.train_params['monitor_mode'],
                                                 patience=self.train_params['patience'])

        modelsummary = ModelSummary(max_depth=-1)

        callbacks = [checkpoint, earlystop, modelsummary]
        if self.logger:
            lr_monitor = LearningRateMonitor(logging_interval='step')
            callbacks.append(lr_monitor)
        return callbacks

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./ckpt/pretrained_gcn/checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            # model.load_state_dict(state_dict)
            model.load_my_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model


    def load_my_state_dict(self, state_dict): #나중에 업데이트안되는 부분 찾아보기
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

