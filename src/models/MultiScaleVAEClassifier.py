import torch
import torch.nn as nn

from src.layers.MLP import MLP
from src.layers.VAE import MultiTaskVAE


class MultiScaleVAEClassifier(nn.Module):
    def __init__(
            self,
            enc_layers,
            cls_layers,
            decoder_layer_list,
            activation: nn.modules.activation = nn.ELU(),
            batch_norm: bool = False,
            dropout: float = 0.01
    ):
        super(MultiScaleVAEClassifier, self).__init__()
        self.vae = MultiTaskVAE(
            enc_layers=enc_layers,
            decoder_layer_list=decoder_layer_list,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )
        self.classifier = MLP(
            [enc_layers[-1]] + cls_layers,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm
        )

    def forward(self, input):
        out = self.vae(input)
        out["y_pred"] = self.classifier(out["z"])
        return out
