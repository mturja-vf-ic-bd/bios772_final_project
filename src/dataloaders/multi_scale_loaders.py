import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import random

from src.dataloaders.loader_utils import cut_templates_and_join
from src.utils.data_utils import template_to_idx_mapping

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


class schaeferMultiScaleLoader(pl.LightningDataModule):
    def __init__(
            self,
            input_template,
            target_template_list,
            batch_size=1,
            split=0.8
    ):
        super(schaeferMultiScaleLoader, self).__init__()
        self.batch_size = batch_size
        self.split = split
        # x = torch.from_numpy(np.load('../../data/train_x.npy')).to(torch.float32)
        # y = torch.from_numpy(np.load('../../data/train_y.npy'))
        x = torch.from_numpy(np.load('data/train_x.npy')).to(torch.float32)
        y = torch.from_numpy(np.load('data/train_y.npy'))
        input_ranges = [template_to_idx_mapping[t] for t in input_template]
        input = cut_templates_and_join(x, input_ranges)
        template_ranges = [template_to_idx_mapping[t] for t in target_template_list]
        target = cut_templates_and_join(x, template_ranges)
        self.input_train, self.input_val, \
        self.target_train, self.target_val, \
        self.y_train, self.y_val = train_test_split(
            input, target, y, test_size=1 - split, random_state=SEED
        )

    def normalize(self, x, eps=1e-4):
        mu = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        return (x - mu) / (std + eps)

    def train_dataloader(self):
        x_train = self.normalize(self.input_train)
        target_train = self.normalize(self.target_train)
        return DataLoader(
            TensorDataset(x_train, target_train, self.y_train),
            batch_size=self.batch_size
        )

    def val_dataloader(self):
        x_val = self.normalize(self.input_val)
        target_val = self.normalize(self.target_val)
        return DataLoader(
            TensorDataset(x_val, target_val, self.y_val),
            batch_size=self.batch_size
        )
