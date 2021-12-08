import torch.nn as nn
from torch.nn.init import xavier_uniform_
import torch

from src.layers.MLP import MLP


class MultiTaskVAE(nn.Module):
    def __init__(
            self,
            enc_layers,
            decoder_layer_list,
            activation: nn.modules.activation = nn.ELU(),
            batch_norm: bool = False,
            dropout: float = 0.01
    ):
        """
        Implementation of a variation autoencoder with muliple decoders from the latent space.
        First the input is projected into the latent space by the encoder and then
        it is fed into a set of decoders whose layer sizes are depicted in decoder_layer_list.

        :param enc_layers: number of units in the layer of encoders. Last value is the dimension of the latent space.
        :param decoder_layer_list: a list of layers for each of the decoders.
        :param batch_norm: whether to add batch norm in the mlp layers
        :param dropout: dropout rate
        """

        super(MultiTaskVAE, self).__init__()
        self.latent_dim = enc_layers[-1]
        # Construct encoder and decoder
        self.encoder = MLP(
            enc_layers,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )
        self.decoder_list = nn.ModuleList()
        for decoder_layer in decoder_layer_list:
            self.decoder_list.append(
                MLP(
                    decoder_layer,
                    activation=activation,
                    dropout=dropout,
                    batch_norm=batch_norm
                )
            )
        # Create mu and sigma layers of VAE
        self.hidden2mean = nn.Linear(self.latent_dim,
                                     self.latent_dim)
        self.hidden2std = nn.Linear(self.latent_dim,
                                    self.latent_dim)

    def reset_params(self):
        xavier_uniform_(self.hidden2mean.weight.data, gain=nn.init.calculate_gain('linear'))
        self.hidden2std.weight.data.fill_(0.0)

    def forward(self, x):
        hidden = self.encode(x)
        mu, logvar = self.hidden2mean(hidden), \
                     self.hidden2std(hidden)
        z_gauss = self.reparameterize(mu, logvar)
        x_rec = []
        for decoder in self.decoder_list:
            x_rec.append(decoder(z_gauss))
        kld_loss_gauss = self.KLD_loss(mu, logvar)
        output = {
            "input": x,
            "recon": torch.cat(x_rec, dim=-1),
            "z": z_gauss,
            "kld_loss": kld_loss_gauss
        }
        return output

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        if self.training:
            z = torch.randn_like(mu).mul(torch.exp(0.5 * logvar)).add_(mu)
        else:
            z = mu
        return z

    @staticmethod
    def KLD_loss(mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())