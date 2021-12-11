import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from src.dataloaders.loader_utils import *

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


class vanillaDataLoader(pl.LightningDataModule):
    def __init__(
            self,
            batch_size=1,
            split=0.8
    ):
        super(vanillaDataLoader, self).__init__()
        self.batch_size = batch_size
        self.split = split
        # x = torch.from_numpy(np.load('../../data/train_x.npy')).to(torch.float32)
        # y = torch.from_numpy(np.load('../../data/train_y.npy'))
        x = torch.from_numpy(np.load('data/train_x.npy')).to(torch.float32)
        y = torch.from_numpy(np.load('data/train_y.npy'))
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(x, y, test_size=1-split, random_state=SEED)

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
