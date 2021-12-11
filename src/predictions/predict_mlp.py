from src.dataloaders.loader_utils import cut_templates
from src.trainers.mlp_trainer import mlpTrainer
from utils import  get_pred_x, normalize
import torch
import numpy as np
import pandas as pd


if __name__ == '__main__':
    template_list = ["vol_mist_444", "thick_mist_444"]
    x, sub_ids = get_pred_x()
    x = cut_templates(x, template_list)
    x = normalize(x)
    model = mlpTrainer.load_from_checkpoint(
        checkpoint_path="mlp-mist-444.ckpt"
    ).model
    y_pred = torch.argmax(model(x), dim=-1).detach().cpu().numpy()
    output = np.stack((sub_ids, y_pred), axis=1)
    output = pd.DataFrame(output, columns=["Subject_ids", "Predicted Label"])
    output.to_csv("mlp_prediction_labels.csv")
