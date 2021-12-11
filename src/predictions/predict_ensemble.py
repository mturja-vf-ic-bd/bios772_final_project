import torch

from src.dataloaders.loader_utils import cut_templates
from src.trainers.ensemble_trainer import ensembleTrainer
import numpy as np
import pandas as pd


def predict_ensemble(x, ckpt_name):
    model = ensembleTrainer.load_from_checkpoint(checkpoint_path=ckpt_name)
    y_pred = model.predict_step(x)
    return y_pred


def entropy(y_pred):
    return -torch.sum(y_pred * torch.log(y_pred), dim=-1)


def predict(x):
    y_pred_nc_ad = predict_ensemble(x, ckpt_name="nc_ad_2.ckpt")
    y_pred_nc_mci = predict_ensemble(x, ckpt_name="nc_mci_2.ckpt")
    y_pred_mci_ad = predict_ensemble(x, ckpt_name="mci_ad_2.ckpt")
    entropy_list = torch.stack([entropy(y_pred_nc_ad), entropy(y_pred_nc_mci), entropy(y_pred_mci_ad)], dim=-1)
    y_pred = torch.stack([torch.argmax(y_pred_nc_ad, dim=1) * 2, torch.argmax(y_pred_nc_mci, dim=1), torch.argmax(y_pred_mci_ad, dim=1) + 1], dim=-1)
    y_pred = y_pred.gather(1, torch.argmin(entropy_list, dim=-1).view(-1, 1)).squeeze(1)
    return y_pred


from utils import normalize, get_pred_x


if __name__ == '__main__':
    template_list = ["vol_mist_444", "thick_mist_444"]
    x, sub_ids = get_pred_x()
    x = cut_templates(x, template_list)
    x = normalize(x)
    y_pred = predict(x)
    output = np.stack((sub_ids, y_pred), axis=1)
    output = pd.DataFrame(output, columns=["Subject_ids", "Predicted Label"])
    output.to_csv("ensemble_prediction_labels.csv")