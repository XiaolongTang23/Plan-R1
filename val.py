from argparse import ArgumentParser

import pytorch_lightning as pl

from datasets import NuplanDataset
from transforms import TokenBuilder
from torch_geometric.loader import DataLoader
from model import PlanR1
from utils import load_config

if __name__ == '__main__':
    pl.seed_everything(1024, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config/val/pred.yaml')
    args = parser.parse_args()
    config = load_config(args.config)

    if config['trainer']['ckpt_path']:
        print(f"Loading model from checkpoint: {config['trainer']['ckpt_path']}")
        model = PlanR1.load_from_checkpoint(config['trainer']['ckpt_path'], **config['model'])
    else:
        raise ValueError("No checkpoint path provided, please provide a valid checkpoint path.")
        
    trainer = pl.Trainer(
        devices=config['trainer']['devices'],
        accelerator=config['trainer']['accelerator'],
    )
    dataset = NuplanDataset(config['dataset']['root'], 
                            config['dataset']['dir'], 
                            'val', 
                            config['dataset']['mode'],
                            transform=TokenBuilder(config['dataset']['token_dict_path'], 
                                                   config['dataset']['interval'], 
                                                   config['dataset']['num_historical_steps'], 
                                                   mode="plan"
                            )
    )
    dataloader = DataLoader(dataset, 
                            batch_size=config['dataloader']['batch_size'], 
                            shuffle=False, 
                            num_workers=config['dataloader']['num_workers'], 
                            pin_memory=config['dataloader']['pin_memory'], 
                            persistent_workers=config['dataloader']['persistent_workers']
    )
    
    trainer.validate(model, dataloader)