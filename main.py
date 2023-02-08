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
from models import Atom_supervised_GCN, Atom_pretrain_contrastive_GCN

def main(config):
    pl.seed_everything(config['seed'],workers=True)
    if config['train']:
        # logger = setup_neptune_logger(config) if config['logger'] else None
        from pytorch_lightning.loggers import WandbLogger
        wandb_logger = WandbLogger(project="test-project") if config['logger'] else None
        train_dataset = AtomMappingDatasetWrapper('train',config['dataset']['train_path'], config['dataset']['valid_path'], config['dataset']['test_path'])
        valid_dataset = AtomMappingDatasetWrapper('val',config['dataset']['train_path'], config['dataset']['valid_path'], config['dataset']['test_path'])

        train_loader = train_dataset.get_data_loaders()
        valid_loader = valid_dataset.get_data_loaders()

        pl_model = Atom_supervised_GCN(
                                    config=config,
                                    model_params=config['model_params']['FN'],
                                    train_params=config['train_params'],
                                    mode = config['train'],
                                    )
        
        if config['train_from_pretrained']:
            pl_model = pl_model._load_pre_trained_weights(pl_model)

        trainer = pl.Trainer(accelerator='gpu',
                                    devices=config['gpu'],
                                    #  strategy='dp',
                                    precision=16,
                                    check_val_every_n_epoch=1,
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
        test_dataset = AtomMappingDatasetWrapper('test',config['dataset']['train_path'], config['dataset']['valid_path'], config['dataset']['test_path'])

        test_loader = test_dataset.get_data_loaders()

        pl_model = Atom_supervised_GCN(
                                    config=config,
                                    model_params=config['model_params']['FN'],
                                    train_params=config['train_params'],
                                    mode = config['train'],
                                    )

        model = pl_model.load_from_checkpoint(config['checkpoint_path'])
        
        trainer = pl.Trainer(accelerator='gpu',
                             devices=config['gpu'])
        predictions = trainer.predict(model, dataloaders=test_loader)
        total_prediction = []
        total_gt = []
        total_confidence = []
        total_reactant_info = []
        total_product_info = []
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


        import pandas as pd
        aa = pd.DataFrame()
        aa['prediction'] = total_prediction
        aa['gt'] = total_gt
        aa['confidence'] = total_confidence
        aa['reactant_info'] = total_reactant_info
        aa['product_info'] = total_product_info
        aa.to_csv('results.csv', index=False)

    return pl_model 



if __name__ == "__main__":
    config = yaml.load(open("config_finetune.yaml", "r"), Loader=yaml.FullLoader)
    config['seed'] = 41
    config['train'] = True
    config['dataset_params']['seed'] = 41
    config['dataset_params']['num_workers'] = 48
    config['dataset_params']['dataset_name'] = 'Brics'
    config['batch_size'] = '1024'
    config['logger'] = True #args.logger
    main(config)    