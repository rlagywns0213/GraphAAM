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

# hj
from torch import nn
from torch_geometric.nn import MessagePassing, GCNConv
from torch.nn import Parameter
import math
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.utils.num_nodes import maybe_num_nodes

from torch_scatter import scatter_add
import torch_sparse
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F



_logger = None

num_atom_type = 119 # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 5 # including aromatic and self-loop edge
num_bond_direction = 3 

class GCNConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super(GCNConv, self).__init__()
        self.emb_dim = emb_dim
        self.aggr = aggr

        self.weight = Parameter(torch.Tensor(emb_dim, emb_dim))
        self.bias = Parameter(torch.Tensor(emb_dim))
        self.reset_parameters()

        self.edge_embedding1 = nn.Embedding(num_bond_type, 1)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, 1)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def reset_parameters(self):
        # glorot(self.weight)
        # zeros(self.bias)
        stdv = math.sqrt(6.0 / (self.weight.size(-2) + self.weight.size(-1)))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        
        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        edge_index, __ = self.gcn_norm(edge_index)

        x = x @ self.weight

        # propagate_type: (x: Tensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_embeddings, size=None)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, edge_attr):
        # return x_j if edge_attr is None else edge_attr.view(-1, 1) * x_j
        return x_j if edge_attr is None else edge_attr + x_j

    def message_and_aggregate(self, adj_t, x):
        return torch_sparse.matmul(adj_t, x, reduce=self.aggr)

    def gcn_norm(self, edge_index, num_nodes=None):
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

class Atom_supervised_GCN(pl.LightningModule):
    def __init__(
        self,
        model_params: Optional = None,
        train_params: Optional = None,
        config: Optional = None,
        logger: Optional = None,
        mode : Optional = False,
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
        print(config)

        '''Load model'''
        if config is None:
            config = {}

        self.emb_dim = self.model_params['embed_dim']
        self.num_layer = self.model_params['num_layer']
        self.feat_dim = self.model_params['feat_dim']
        self.drop_ratio = self.model_params['drop_ratio']
        
        self.x_embedding1 = nn.Embedding(num_atom_type, self.emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, self.emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        
        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(self.num_layer):
            self.gnns.append(GCNConv(self.emb_dim, aggr="add"))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(self.num_layer):
            self.batch_norms.append(nn.BatchNorm1d(self.emb_dim))

        self.fc1 = nn.Linear(self.emb_dim, self.feat_dim)
        self.fc2 = nn.Linear(self.feat_dim, self.emb_dim)
       
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
        z1 = F.normalize(z1)
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
    
    def gcn_forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        h = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        return h

    def forward(self,**inputs):
        reactant = inputs['reactant']
        product = inputs['product']
       
        # get GCN encoder
        reactant_z = self.gcn_forward(reactant)
        product_z = self.gcn_forward(product)

        h1 = self.projection(reactant_z) #13681, 300 => 13681. 300
        h2 = self.projection(product_z)

        return h1, h2

    def training_step(self, batch, batch_idx):
        reactant_h, product_h = self(**batch)
        
        batch_size = batch['product'].batch.max().item()+1

        f = lambda x: torch.exp(x / 0.5) # self.tau == 0.5

        device = reactant_h.device
        losses = []
        preds = []
        gts = []
        confidences_list = []
        for b in range(batch_size):
            batch_react_mask = (batch['reactant']['batch'] == b)
            batch_product_mask = (batch['product']['batch'] == b)
            
            per_gt = batch['product'].y[b]

            sim_matrix = f(self.sim(product_h[batch_product_mask], reactant_h[batch_react_mask]))
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(sim_matrix, torch.tensor(per_gt, device=device))

            losses.append(loss)

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
            confidences_list.append(confidences)

        return {"loss": sum(losses) / len(losses),
                "preds": preds,
                "gts": gts,
                'confidences' : confidences_list}

    def training_step_end(self, batch_parts):
        """
        Calculate loss and metrics after one train epoch ends.
        :param outputs: List of concatenated outputs of each batch. list[dict]
        """
        loss = batch_parts['loss']
        preds_total = batch_parts['preds']
        gts_total = batch_parts['gts']
        confidences_list = batch_parts['confidences']

        
        c = 0
        total = 0
        Ds = []
        percents = []

        for p, g in zip(preds_total, gts_total):
            total+=1
            list_p = list(p)
            g = [int(i) for i in g]
            Ds.append(list_p==g)
            # print(p)
            if list_p ==g:
                c+=1
            if p is None:
                percents.append(0)
                continue
            else:
                p_arr = np.array(p)
                g_arr = np.array(g)
                if p_arr.shape != g_arr.shape:
                    percents.append(0)
                    continue
                percents.append((p_arr == g_arr).sum()/p_arr.shape[0])

        correctness = c / total
        percents = np.array(percents)
        percents = np.mean(percents)


        if self.logger:
            # log metrics
            self.log('train_loss', loss)
            self.log('train_correctness', correctness)
            self.log('train_same_mapping', c)
            self.log('train_total', total)
            self.log('train_percent', percents)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        reactant_h, product_h = self(**batch)
        batch_size = batch['product'].batch.max().item()+1
        f = lambda x: torch.exp(x / 0.5) # self.tau == 0.5

        device = reactant_h.device
        losses = []
        preds = []
        gts = []
        confidences_list = []
        for b in range(batch_size):
            batch_react_mask = (batch['reactant']['batch'] == b)
            batch_product_mask = (batch['product']['batch'] == b)
            
            per_gt = batch['product'].y[b]

            sim_matrix = f(self.sim(product_h[batch_product_mask], reactant_h[batch_react_mask]))
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(sim_matrix, torch.tensor(per_gt, device=device))

            losses.append(loss)

            len_per_product_atom_list = batch['product'].x[batch_product_mask].shape[0]
            pxr_mapping_vector = (np.ones(len_per_product_atom_list) * -1).astype(int) # (25,) 인 vector -1 
            confidences = np.ones(len_per_product_atom_list)

            for i in range(len_per_product_atom_list):
                product_atom_to_map = torch.argmax(torch.max(sim_matrix, dim=1)[0])
                corresponding_reactant_atom = torch.argmax(sim_matrix, axis=1)[product_atom_to_map]
                confidence = float(torch.max(sim_matrix))

                pxr_mapping_vector[product_atom_to_map] = corresponding_reactant_atom
                confidences[product_atom_to_map] = round(confidence,2)
                sim_matrix = self._update_matrix(
                    sim_matrix, product_atom_to_map, corresponding_reactant_atom
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
        gts_total = batch_parts['gts']
        confidences_list = batch_parts['confidences']

        
        c = 0
        total = 0
        Ds = []
        percents = []

        for p, g in zip(preds_total, gts_total):
            total+=1
            list_p = list(p)
            g = [int(i) for i in g]
            Ds.append(list_p==g)
            # print(p)
            if list_p ==g:
                c+=1
            if p is None:
                percents.append(0)
                continue
            else:
                p_arr = np.array(p)
                g_arr = np.array(g)
                if p_arr.shape != g_arr.shape:
                    percents.append(0)
                    continue
                percents.append((p_arr == g_arr).sum()/p_arr.shape[0])

        correctness = c / total
        percents = np.array(percents)
        percents = np.mean(percents)


        if self.logger:
            # log metrics
            self.log('val_loss', loss)
            self.log('val_correctness', correctness)
            self.log('val_same_mapping', c)
            self.log('val_total', total)
            self.log('val_percent', percents)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        reactant_h, product_h = self(**batch)
        batch_size = batch['product'].batch.max().item()+1
        f = lambda x: torch.exp(x / 0.5) # self.tau == 0.5

        device = reactant_h.device
        losses = []
        preds = []
        gts = []
        confidences_list = []
        reactant_info=[]
        product_info=[]
        for b in range(batch_size):
            batch_react_mask = (batch['reactant']['batch'] == b)
            batch_product_mask = (batch['product']['batch'] == b)
            
            per_gt = batch['product'].y[b]

            sim_matrix = f(self.sim(product_h[batch_product_mask], reactant_h[batch_react_mask]))

            len_per_product_atom_list = batch['product'].x[batch_product_mask].shape[0]
            pxr_mapping_vector = (np.ones(len_per_product_atom_list) * -1).astype(int) # (25,) 인 vector -1 
            confidences = np.ones(len_per_product_atom_list)

            for i in range(len_per_product_atom_list):
                product_atom_to_map = torch.argmax(torch.max(sim_matrix, dim=1)[0])
                corresponding_reactant_atom = torch.argmax(sim_matrix, axis=1)[product_atom_to_map]
                confidence = float(torch.max(sim_matrix))

                pxr_mapping_vector[product_atom_to_map] = corresponding_reactant_atom
                confidences[product_atom_to_map] = round(confidence,2)
                sim_matrix = self._update_matrix(
                    sim_matrix, product_atom_to_map, corresponding_reactant_atom
                )

            reactant_info.append(batch['reactant'].x[:,0][batch_react_mask].tolist())
            product_info.append(batch['product'].x[:,0][batch_product_mask].tolist())
            preds.append(list(pxr_mapping_vector))
            gts.append(per_gt)
            confidences_list.append(list(confidences))

        return {"pred" : preds,
                "gt": gts,
                'confidences_list' : confidences_list,
                'reactant_info' : reactant_info,
                'product_info' :product_info}


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

### contraistve            

class Atom_pretrain_contrastive_GCN(pl.LightningModule):
    def __init__(
        self,
        model_params: Optional = None,
        train_params: Optional = None,
        config: Optional = None,
        logger: Optional = None,
        mode : Optional = False,
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
        print(config)

        '''Load model'''
        if config is None:
            config = {}

        self.emb_dim = self.model_params['embed_dim']
        self.num_layer = self.model_params['num_layer']
        self.feat_dim = self.model_params['feat_dim']
        self.drop_ratio = self.model_params['drop_ratio']
        
        self.x_embedding1 = nn.Embedding(num_atom_type, self.emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, self.emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        
        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(self.num_layer):
            self.gnns.append(GCNConv(self.emb_dim, aggr="add"))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(self.num_layer):
            self.batch_norms.append(nn.BatchNorm1d(self.emb_dim))

        self.fc1 = nn.Linear(self.emb_dim, self.feat_dim)
        self.fc2 = nn.Linear(self.feat_dim, self.emb_dim)
       
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
        z1 = F.normalize(z1)
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
    
    def gcn_forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        h = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        return h

    def forward(self,**inputs):
        #gt = inputs["correct_map"]
        #atom_masking = inputs["atom_numbers"]
        reactant = inputs['reactant']
        product = inputs['product']
       
        # get GCN encoder
        reactant_z = self.gcn_forward(reactant)
        product_z = self.gcn_forward(product)

        h1 = self.projection(reactant_z) #13681, 300 => 13681. 300
        h2 = self.projection(product_z)

        return h1, h2

    def training_step(self, batch, batch_idx):
        reactant_h, product_h = self(**batch)
        device = product_h.device
        batch_size = batch['product'].batch.max().item()+1

        f = lambda x: torch.exp(x / 0.5) # self.tau == 0.5

        losses = []
        preds = []
        gts = []
        confidences_list = []
        for b in range(batch_size):
            batch_react_mask = (batch['reactant']['batch'] == b)
            batch_product_mask = (batch['product']['batch'] == b)
            
            per_reactant_atom_list = batch['reactant'].x[batch_react_mask][:,0]
            per_product_atom_list = batch['product'].x[batch_product_mask][:,0]

            negative_mask = np.full((len(per_product_atom_list), len(per_reactant_atom_list)), True)
            ground_truth_atom_mask = np.full((len(per_product_atom_list), len(per_reactant_atom_list)), False)
            per_gt = batch['product'].y[b]

            assert len(per_product_atom_list) == len(per_gt)

            # create masking for negative/positive 
            for product_index, p in enumerate(per_product_atom_list.cpu()):
                for react_index, r in enumerate(per_reactant_atom_list.cpu()):
                    if p == r:
                        negative_mask[product_index, react_index] = False  # negative mask for not same atoms
                        ground_truth_atom_mask[product_index, react_index] = True 
                    else:
                        continue
                # ground_truth_atom_mask[product_index, per_gt[product_index]] = True # postive mask for ground truth (atom-to-atom mapping)
                
            negative_mask = torch.tensor(negative_mask).to(device)
            ground_truth_atom_mask = torch.tensor(ground_truth_atom_mask).to(device)
            between_sim = f(self.sim(product_h[batch_product_mask], reactant_h[batch_react_mask])) #product * reactant
            neg_output = (between_sim*negative_mask).sum(1)
            pos_output = (between_sim*ground_truth_atom_mask).sum(1)
            debug = [int(i) for i in neg_output.cpu() if int(i) == 0.0]

            if len(debug) == len(per_gt):
                # 모든 원소들이 같은 경우 => 분모가 0이 나오므로, 
                continue
            else:
                losses.append(-torch.log((pos_output / (pos_output+neg_output))))

            len_per_product_atom_list = batch['product'].x[batch_product_mask].shape[0]
            pxr_mapping_vector = (np.ones(len_per_product_atom_list) * -1).astype(int) # (25,) 인 vector -1 
            confidences = np.ones(len_per_product_atom_list)
            
            sim_for_evaluation = between_sim.detach().cpu()
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

        l1 = torch.cat(losses)
        
        return {"loss": l1.mean(),
                "preds": preds,
                "gts": gts,
                'confidences' : confidences_list}

    def training_step_end(self, batch_parts):
        """
        Calculate loss and metrics after one train epoch ends.
        :param outputs: List of concatenated outputs of each batch. list[dict]
        """
        loss = batch_parts['loss']
        preds_total = batch_parts['preds']
        gts_total = batch_parts['gts']
        confidences_list = batch_parts['confidences']

        
        c = 0
        total = 0
        Ds = []
        percents = []

        for p, g in zip(preds_total, gts_total):
            total+=1
            list_p = list(p)
            g = [int(i) for i in g]
            Ds.append(list_p==g)
            # print(p)
            if list_p ==g:
                c+=1
            if p is None:
                percents.append(0)
                continue
            else:
                p_arr = np.array(p)
                g_arr = np.array(g)
                if p_arr.shape != g_arr.shape:
                    percents.append(0)
                    continue
                percents.append((p_arr == g_arr).sum()/p_arr.shape[0])

        correctness = c / total
        percents = np.array(percents)
        percents = np.mean(percents)


        if self.logger:
            # log metrics
            self.log('train_loss', loss)
            self.log('train_correctness', correctness)
            self.log('train_same_mapping', c)
            self.log('train_total', total)
            self.log('train_percent', percents)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        reactant_h, product_h = self(**batch)
        device = product_h.device
        batch_size = batch['product'].batch.max().item()+1
        f = lambda x: torch.exp(x / 0.5) # self.tau == 0.5

        losses = []
        preds = []
        gts = []
        confidences_list = []
        for b in range(batch_size):
            batch_react_mask = (batch['reactant']['batch'] == b)
            batch_product_mask = (batch['product']['batch'] == b)
            
            per_reactant_atom_list = batch['reactant'].x[batch_react_mask][:,0]
            per_product_atom_list = batch['product'].x[batch_product_mask][:,0]

            negative_mask = np.full((len(per_product_atom_list), len(per_reactant_atom_list)), True)
            ground_truth_atom_mask = np.full((len(per_product_atom_list), len(per_reactant_atom_list)), False)
            per_gt = batch['product'].y[b]

            assert len(per_product_atom_list) == len(per_gt)

            # create masking for negative/positive 
            for product_index, p in enumerate(per_product_atom_list.cpu()):
                for react_index, r in enumerate(per_reactant_atom_list.cpu()):
                    if p == r:
                        negative_mask[product_index, react_index] = False  # negative mask for not same atoms
                        ground_truth_atom_mask[product_index, react_index] = True 
                    else:
                        continue
                # ground_truth_atom_mask[product_index, per_gt[product_index]] = True # postive mask for ground truth (atom-to-atom mapping)
                

            negative_mask = torch.tensor(negative_mask).to(device)
            ground_truth_atom_mask = torch.tensor(ground_truth_atom_mask).to(device)
            between_sim = f(self.sim(product_h[batch_product_mask], reactant_h[batch_react_mask])) #product * reactant
            neg_output = (between_sim*negative_mask).sum(1)
            pos_output = (between_sim*ground_truth_atom_mask).sum(1)
            debug = [int(i) for i in neg_output.cpu() if int(i) == 0.0]

            if len(debug) == len(per_gt):
                # 모든 원소들이 같은 경우 => 분모가 0이 나오므로, 
                continue
            else:
                losses.append(-torch.log((pos_output / (pos_output+neg_output))))

            len_per_product_atom_list = batch['product'].x[batch_product_mask].shape[0]
            pxr_mapping_vector = (np.ones(len_per_product_atom_list) * -1).astype(int) # (25,) 인 vector -1 
            confidences = np.ones(len_per_product_atom_list)

            sim_for_evaluation = between_sim.detach().cpu()
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

        l1 = torch.cat(losses)
        
        return {"loss": l1.mean(),
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
        gts_total = batch_parts['gts']
        confidences_list = batch_parts['confidences']

        
        c = 0
        total = 0
        Ds = []
        percents = []

        for p, g in zip(preds_total, gts_total):
            total+=1
            list_p = list(p)
            g = [int(i) for i in g]
            Ds.append(list_p==g)
            # print(p)
            if list_p ==g:
                c+=1
            if p is None:
                percents.append(0)
                continue
            else:
                p_arr = np.array(p)
                g_arr = np.array(g)
                if p_arr.shape != g_arr.shape:
                    percents.append(0)
                    continue
                percents.append((p_arr == g_arr).sum()/p_arr.shape[0])

        correctness = c / total
        percents = np.array(percents)
        percents = np.mean(percents)


        if self.logger:
            # log metrics
            self.log('val_loss', loss)
            self.log('val_correctness', correctness)
            self.log('val_same_mapping', c)
            self.log('val_total', total)
            self.log('val_percent', percents)

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        attn_outputs, attn_weights = self(**batch)
        reactant_h, product_h = self(**batch)
        attn_weights = attn_weights.detach().cpu().numpy()
        selected_max = np.argmax(attn_weights, -1)
        selected_max = np.uint32(selected_max).tolist()
        remove_set = {0}
        selected_max = [[k for k in j if k not in remove_set] for j in selected_max]
        ranks = [get_rank(i) for i in selected_max]
        return {"pred": ranks}

    def test_epoch_end(self, outputs):
        """
        Calculate loss and metrics after one train epoch ends.
        :param outputs: List of concatenated outputs of each batch. list[dict]
        """
        predictions = []
        for o in outputs:
            predictions.append(o['pred'])
        return predictions

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        attn_outputs, attn_weights = self(**batch)
        gts = batch['correct_map']


        reaction_formats = (determine_format(rxn) for rxn in batch['unmapped_rxn'])
        reactions = [parse_any_reaction_smiles(rxn) for rxn in batch['unmapped_rxn']]
        rxns = [to_reaction_smiles(reaction, reaction_format=ReactionFormat.STANDARD_WITH_TILDE) for reaction in
                reactions]
        attns = [a[mask][:, mask] for a, mask in zip(attn_weights, batch['attention_mask'].to(torch.bool))]


        return {"pred" : attn_weights,
                "gt": gts,
                'attn' : attns,
                'rxn' : rxns,
                'unmapped_rxn' : batch['unmapped_rxn'],
                'mapped_rxn': batch['mapped_rxn']}


        '''mask_p = batch['p_numbers']
        mask_p = mask_p < 0
        pred_seq = attn_weights.argmax(-1)
        pred_seq = torch.where(~mask_p, pred_seq, torch.zeros_like(pred_seq))
        seq_acc = pred_seq.eq(gts).sum(-1) - mask_p.sum(-1)

        mask_p = mask_p.unsqueeze(-1).expand(-1, -1, mask_p.shape[-1])
        attn_weights = torch.where(~mask_p, attn_weights, torch.zeros_like(attn_weights))
        selected_max = attn_weights.argmax(-1)
        selected_max = selected_max.detach().cpu().numpy()
        selected_max = np.uint32(selected_max).tolist()
        remove_set = {0}
        selected_max = [[k for k in j if k not in remove_set] for j in selected_max]
        ranks = [get_rank(i) for i in selected_max]
        return {"pred" : ranks,
                'pred_seq' : pred_seq,
                'seq_acc' : seq_acc,
                'num_atoms': (batch['p_numbers'] >= 0).sum(-1)
                }'''

    ''' predict_epoch_end doesn't work 
    def predict_epoch_end(self, outputs):
        predictions = []
        for o in outputs:
            predictions.append(o['pred'])
        return predictions
    '''
    '''def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            #{
            #    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            #    "weight_decay": self.hparams.weight_decay,
            #},
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.train_params['optimizer']['lr'])
        return optimizer'''

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

