from typing import Optional
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
import os

from datasets import NuplanDataset
from transforms import TokenBuilder


class NuplanDataModule(pl.LightningDataModule):

    def __init__(self,
                 root: str,
                 dir:str,
                 train_batch_size: int,
                 val_batch_size: int,
                 shuffle: bool = True,
                 num_workers: int = 8,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 token_dict_path: str = None,
                 num_tokens: int = 1024,
                 interval: int = 5,
                 mode: str = 'pred',
                 num_historical_steps: int = 20,
                 num_future_steps: int = 80,
                 num_samples_per_second: int = 10,
                 num_total_scenarios: int = 1000000,
                 ratio: float = 0.1,
                 parallel: bool = True,
                 **kwargs) -> None:
        super(NuplanDataModule, self).__init__()
        self.root = root
        self.dir = dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.num_tokens = num_tokens
        self.interval = interval
        self.mode = mode
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_samples_per_second = num_samples_per_second
        self.num_total_scenarios = num_total_scenarios
        self.ratio = ratio
        self.parallel = parallel
        self.train_transform = TokenBuilder(token_dict_path, self.interval, num_historical_steps, self.mode)
        self.val_transform = TokenBuilder(token_dict_path, self.interval, num_historical_steps, self.mode)

    def prepare_data(self) -> None:
        NuplanDataset(self.root, self.dir, 'train', self.mode, self.train_transform, num_total_scenarios=self.num_total_scenarios, ratio=self.ratio, parallel=self.parallel)
        NuplanDataset(self.root, self.dir, 'val', self.mode, self.val_transform, num_total_scenarios=self.num_total_scenarios, ratio=self.ratio, parallel=self.parallel)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = NuplanDataset(self.root, self.dir, 'train', self.mode, self.train_transform, num_total_scenarios=self.num_total_scenarios, ratio=self.ratio, parallel=self.parallel)
        self.val_dataset = NuplanDataset(self.root, self.dir, 'val', self.mode, self.val_transform, num_total_scenarios=self.num_total_scenarios, ratio=self.ratio, parallel=self.parallel)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)
