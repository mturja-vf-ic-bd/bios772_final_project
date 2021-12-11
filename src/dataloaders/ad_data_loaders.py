from abc import ABC

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from src.dataloaders.loader_utils import *

SEED = 42


class adDataLoader(pl.LightningDataModule):
    def __init__(
            self,
            batch_size=1,
            split=0.8,
            template_list=None,
            oversample=False
    ):
        super(adDataLoader, self).__init__()
        self.batch_size = batch_size
        self.split = split
        self.smote = SMOTE(random_state=SEED)
        # x = torch.from_numpy(np.load('../../data/train_x.npy')).to(torch.float32)
        # y = torch.from_numpy(np.load('../../data/train_y.npy'))
        x = torch.from_numpy(np.load('data/train_x.npy')).to(torch.float32)
        y = torch.from_numpy(np.load('data/train_y.npy'))
        if template_list is not None:
            range_list = [template_to_idx_mapping[template] for template in template_list]
            x = cut_templates_and_join(x, range_list)
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x, y, test_size=1-split, random_state=SEED)
        if oversample:
            self.x_train, self.y_train = self.smote.fit_resample(self.x_train, self.y_train)
            self.x_train = torch.from_numpy(self.x_train)
            self.y_train = torch.from_numpy(self.y_train)

    def normalize(self, x, eps=1e-4):
        mu = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        return (x - mu) / (std + eps)

    def train_dataloader(self):
        x_train = self.normalize(self.x_train)
        return DataLoader(TensorDataset(x_train, self.y_train), batch_size=self.batch_size)

    def val_dataloader(self):
        x_val = self.normalize(self.x_val)
        return DataLoader(TensorDataset(x_val, self.y_val), batch_size=self.batch_size)


class EnsembleDataLoader(pl.LightningDataModule):
    def __init__(
            self,
            batch_size=1,
            split=0.8,
            template_list=None,
            pair_name="NC_AD"
    ):
        super(EnsembleDataLoader, self).__init__()
        self.batch_size = batch_size
        self.split = split
        self.smote = SMOTE(random_state=SEED)
        # x = torch.from_numpy(np.load(f'../../data/ensemble/train_x_{pair_name}.npy')).to(torch.float32)
        # y = torch.from_numpy(np.load(f'../../data/ensemble/train_y_{pair_name}.npy'))
        x = torch.from_numpy(np.load(f'data/ensemble/train_x_{pair_name}.npy')).to(torch.float32)
        y = torch.from_numpy(np.load(f'data/ensemble/train_y_{pair_name}.npy'))
        if template_list is not None:
            range_list = [template_to_idx_mapping[template] for template in template_list]
            x = cut_templates_and_join(x, range_list)
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x, y, test_size=1 - split,
                                                                              random_state=SEED)
        self.x_train = self.normalize(self.x_train)
        self.x_val = self.normalize(self.x_val)

    def normalize(self, x, eps=1e-4):
        mu = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        return (x - mu) / (std + eps)

    def train_dataloader(self):
        return DataLoader(TensorDataset(self.x_train, self.y_train), batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(TensorDataset(self.x_val, self.y_val), batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(TensorDataset(self.x_val, self.y_val), batch_size=self.batch_size)
