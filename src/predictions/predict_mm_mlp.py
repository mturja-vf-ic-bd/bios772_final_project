from src.trainers.multi_modal_trainer import MultiModalTrainer
from src.predictions.utils import get_pred_x, normalize
import numpy as np
import torch
import pandas as pd


if __name__ == '__main__':
    x, sub_ids = get_pred_x()
    x = normalize(x)
    model = MultiModalTrainer.load_from_checkpoint(
        checkpoint_path="src/predictions/mm-mlp-schaefer-200-mist-444.ckpt"
    ).model
    y_pred = torch.argmax(model(x), dim=-1).detach().cpu().numpy()
    output = np.stack((sub_ids, y_pred), axis=1)
    output = pd.DataFrame(output, columns=["Subject_ids", "Predicted Label"])
    output.to_csv("mm-mlp_prediction_labels.csv")
