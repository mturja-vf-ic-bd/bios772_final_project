import torch
import numpy as np
from src.dataloaders.loader_utils import cut_templates


def normalize(x, eps=1e-4):
    mu = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True)
    return (x - mu) / (std + eps)


def get_val_x(template_list):
    x = cut_templates(torch.from_numpy(np.load("../../data/ensemble/val_x.npy")), template_list).to(torch.float32)
    y = torch.from_numpy(np.load("../../data/ensemble/val_y.npy"))
    x = normalize(x)
    return x, y


def get_pred_x():
    x = torch.from_numpy(np.load("../../data/test_x.npy")).to(torch.float32)
    sub_ids = np.load("../../data/test_sub_ids.npy")
    return x, sub_ids