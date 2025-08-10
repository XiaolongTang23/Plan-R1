from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from datamodules import NuplanDataModule
from model import PlanR1
from utils import load_config

import os

CURRENT_FILE_PATH = os.path.realpath(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_FILE_PATH)
os.environ["PROJECT_ROOT"] = PROJECT_ROOT

if __name__ == '__main__':
    pl.seed_everything(1024, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config/train/pred.yaml')
    args = parser.parse_args()
    config = load_config(args.config)

    # Load the model from a checkpoint if provided
    if config['trainer']['ckpt_path']:
        print(f"Loading model from checkpoint: {config['trainer']['ckpt_path']}")
        model = PlanR1.load_from_checkpoint(config['trainer']['ckpt_path'], **config['model'])
    else:
        print("No checkpoint path provided, initializing a new model.")
        model = PlanR1(**config['model'])
    
    datamodule = NuplanDataModule(**config['datamodule'])
    model_checkpoint = ModelCheckpoint(**config['trainer']['ckpt'])
    lr_monitor = LearningRateMonitor(**config['trainer']['lr_monitor'])
    csv_logger = CSVLogger(**config['trainer']['csv_logger'])
    trainer = pl.Trainer(
        strategy=config['trainer']['strategy'],
        devices=config['trainer']['devices'],
        accelerator=config['trainer']['accelerator'],
        callbacks=[model_checkpoint, lr_monitor],
        max_epochs=config['trainer']['max_epochs'],
        logger=csv_logger
    )

    trainer.fit(model, datamodule)