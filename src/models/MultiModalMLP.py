import torch
import torch.nn as nn

from src.layers.MLP import MLP
from src.dataloaders.loader_utils import get_lengths_of_template, make_dict_from_template_names


class MultiModalMLP(nn.Module):
    def __init__(
            self,
            enc_layer_dict,
            latent_dim,
            cls_layers,
            activation: nn.modules.activation = nn.ELU(),
            batch_norm: bool = False,
            dropout: float = 0.01
    ):
        super(MultiModalMLP, self).__init__()
        self.modality = len(enc_layer_dict)
        self.template_list = list(enc_layer_dict.keys())
        self.encoder_modules = nn.ModuleDict({
            enc_names:
                MLP(
                    [get_lengths_of_template(enc_names)] + enc_hidden_layers + [latent_dim],
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    prefix=enc_names
                ) for enc_names, enc_hidden_layers in enc_layer_dict.items()
            }
        )
        self.classifier = MLP(
            [self.modality * latent_dim] + cls_layers,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm
        )

    def forward(self, x):
        x_dict = make_dict_from_template_names(x, self.template_list)
        embedding = []
        for t_name, x_cut in x_dict.items():
            embedding.append(self.encoder_modules[t_name](x_cut))
        embedding = torch.cat(embedding, dim=1)
        y_pred = self.classifier(embedding)
        return y_pred
