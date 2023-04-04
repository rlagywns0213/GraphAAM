import os
import shutil
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F
from dataset.dataset_test import AtomMappingDatasetWrapper
import pytorch_lightning as pl
from models_GT import FG

def main(config):
    random_seed = config['seed']
    pl.seed_everything(random_seed,workers=True)
    torch.manual_seed(random_seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU

    if config['train']:
        # logger = setup_neptune_logger(config) if config['logger'] else None
        from pytorch_lightning.loggers import WandbLogger
        wandb_logger = WandbLogger(project="GraphAAM") if config['logger'] else None
        train_dataset = AtomMappingDatasetWrapper('train',config['dataset']['train_path'], config['dataset']['valid_path'], config['dataset']['test_path'], 
                                                  train_batch_size= config['batch_size'], eval_batch_size = config['batch_size'], 
                                                  multiple_solution = config['multiple_solution'], bert_input = config['bert_input'])
        valid_dataset = AtomMappingDatasetWrapper('val',config['dataset']['train_path'], config['dataset']['valid_path'], config['dataset']['test_path'],
                                                  train_batch_size= config['batch_size'], eval_batch_size = config['batch_size'],
                                                  multiple_solution = config['multiple_solution'], bert_input = config['bert_input'])

        train_loader = train_dataset.get_data_loaders()
        valid_loader = valid_dataset.get_data_loaders()

        pl_model = FG(
                                    config=config,
                                    model_params=config['model_params']['FN'],
                                    train_params=config['train_params'],
                                    mode = config['train'],
                                    multiple_solution= config['multiple_solution'],
                                    )
        
        if config['train_from_pretrained']:
            pl_model = pl_model._load_pre_trained_weights(pl_model)
        
        trainer = pl.Trainer(accelerator='gpu',
                                    devices=[2],
                                    # strategy='ddp',
                                    precision=16,
                                    check_val_every_n_epoch=3,
                                    amp_backend='native',
                                    max_epochs=200,
                                    num_sanity_val_steps=0,
                                    fast_dev_run=False,
                                    enable_checkpointing=True,
                                    logger=wandb_logger,
                                    overfit_batches=0,
                                    log_every_n_steps=1)
                                    # resume_from_checkpoint=args.resume_from_checkpoint)
        trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    else:
        #inference
        test_dataset = AtomMappingDatasetWrapper('test',config['dataset']['train_path'], config['dataset']['valid_path'], config['dataset']['test_path'], train_batch_size= config['batch_size'], eval_batch_size = config['batch_size'], multiple_solution = config['multiple_solution'], bert_input = config['bert_input'])

        test_loader = test_dataset.get_data_loaders()

        pl_model = FG(
                                    config=config,
                                    model_params=config['model_params']['FN'],
                                    train_params=config['train_params'],
                                    mode = config['train'],
                                    multiple_solution = config['multiple_solution'],
                                    test_atom = config['test_atom_type']
                                    )

        # model = pl_model
        model = pl_model.load_from_checkpoint(config['checkpoint_path'])
        
        trainer = pl.Trainer(accelerator='gpu',
                             devices=config['gpu'])
        predictions = trainer.predict(model, dataloaders=test_loader)
        total_prediction = []
        total_gt = []
        total_confidence = []
        total_reactant_info = []
        total_product_info = []
        total_reactant_smiles = []
        total_product_smiles = []
        total_reactant_edge = []
        total_product_edge = []
        total_multiple_gts = []
        for p in predictions:
            prediction = p['pred']
            gt = p['gt']
            confidence = p['confidences_list']
            reactant_info = p['reactant_info']
            product_info = p['product_info']
            total_prediction.extend(prediction)
            total_gt.extend(gt)
            total_confidence.extend(confidence)
            total_reactant_info.extend(reactant_info)
            total_product_info.extend(product_info)
            total_reactant_smiles.extend(p['reactant_smiles'])
            total_product_smiles.extend(p['product_smiles'])
            total_reactant_edge.extend(p['reactant_edge_info'])
            total_product_edge.extend(p['product_edge_info'])
            total_multiple_gts.extend(p['multiple_gts'])

        import pandas as pd
        aa = pd.DataFrame()
        aa['prediction'] = total_prediction
        aa['ground_truth'] = total_gt
        aa['confidence'] = total_confidence
        aa['reactant_info'] = total_reactant_info
        aa['product_info'] = total_product_info
        aa['reactant_smiles'] = total_reactant_smiles
        aa['product_smiles'] = total_product_smiles
        aa['reactant_edge_info'] = total_reactant_edge
        aa['product_edge_info'] = total_product_edge
        aa['multiple_gts'] = total_multiple_gts

        aa.to_csv('final_results/test_atoms_no_norm_'+str(config['test_atom_type'])+str(model.__class__.__name__)+str(config['checkpoint_path'].split('/')[-1])+'_'+str(config['dataset']['test_path'].split('/')[-1])+'.csv', index=False)
        # aa.to_csv('no_initialize.csv')
          
    return pl_model 



if __name__ == "__main__":
    config = yaml.load(open("config_finetune_GT.yaml", "r"), Loader=yaml.FullLoader)
    # config = yaml.load(open("config_finetune.yaml", "r"), Loader=yaml.FullLoader)
    config['seed'] = 41
    config['train'] = False
    config['test_atom_type'] = True
    config['dataset_params']['seed'] = 41
    config['dataset_params']['num_workers'] = 48
    config['dataset_params']['dataset_name'] = 'Brics'
    config['batch_size'] = 32
    config['logger'] = True #args.logger
    config['multiple_solution']=False
    config['bert_input']=False
    config['skip_connection']= False

    main(config)