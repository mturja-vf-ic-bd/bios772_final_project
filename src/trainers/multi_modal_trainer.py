import os
import time

import torch.nn as nn
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import torchmetrics

from src.dataloaders.vanilla_data_loader import vanillaDataLoader
from src.utils.data_utils import template_to_idx_mapping
from src.utils.folder_manager import create_version_dir
from src.models.MultiModalMLP import MultiModalMLP


class MultiModalTrainer(pl.LightningModule):
    def __init__(
            self,
            enc_hidden_layers,
            cls_layers,
            latent_dim,
            template_list,
            lr=1e-4,
            activation=nn.ELU(),
            batch_norm=False,
            dropout=0.01
    ):
        super(MultiModalTrainer, self).__init__()
        enc_layer_dict = {}
        num_classes = cls_layers[-1]
        self.learning_rate = lr
        for t in template_list:
            enc_layer_dict[t] = enc_hidden_layers
        self.model = MultiModalMLP(
            enc_layer_dict=enc_layer_dict,
            cls_layers=cls_layers,
            latent_dim=latent_dim,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm
        )
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.train_f1 = torchmetrics.F1(
            num_classes=num_classes,
            average='macro'
        )
        self.val_f1 = torchmetrics.F1(
            num_classes=num_classes,
            average='macro'
        )
        self.save_hyperparameters()

    def compute_loss(self, y_pred, y_true, weight):
        loss_fn = nn.CrossEntropyLoss(weight=weight)
        cls_loss = loss_fn(y_pred, y_true)
        return {
            "cls_loss": cls_loss
        }

    def forward(self, input):
        return self.model(input)

    def add_log(self, losses, category="train"):
        for k, v in losses.items():
            self.log(category + "/" + k, v)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def training_epoch_end(self, outputs):
        self.log('train/acc_epoch', self.train_accuracy.compute())
        self.log('train/f1_epoch', self.train_f1.compute())

    def validation_epoch_end(self, outputs):
        self.log('val/acc_epoch', self.val_accuracy.compute())
        self.log('val/f1_epoch', self.val_f1.compute())

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.learning_rate,
                                weight_decay=0.01)

    def _common_step(self, batch, batch_idx, category="train"):
        x, y = batch
        y_pred = self.model(x)

        # Compute weight for each class
        weight = torch.FloatTensor(
            [len(y) * 1.0 / (y == 0).sum(), len(y) * 1.0 / (y == 1).sum(),
             len(y) * 1.0 / (y == 2).sum()]
        ).to(y.device)
        losses = self.compute_loss(y_pred, y, weight)
        loss = losses["cls_loss"]
        if category == "train":
            self.train_accuracy(torch.argmax(y_pred, dim=-1), y)
            self.train_f1(torch.softmax(y_pred, dim=-1), y)
        elif category == "val":
            self.val_accuracy(torch.argmax(y_pred, dim=-1), y)
            self.val_f1(torch.softmax(y_pred, dim=-1), y)
        self.add_log(losses, category)
        return loss


def cli_main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--enc_hidden_layers",
                        nargs="+", type=int,
                        help="Embedding module layers")
    parser.add_argument("-b", "--batch_size", nargs="?", type=int, default=1)
    parser.add_argument("-g", "--gpus", nargs="?", type=int, default=0)
    parser.add_argument("-m", "--max_epochs", nargs="?", type=int, default=100)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--cls_layers", type=int, nargs="+",
                        help="Classification module layers")
    parser.add_argument("-d", "--dropout", nargs="?",
                        type=float, default=0.01)
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--mode", choices=["train", "test"], default="test")
    parser.add_argument("--ckpt", type=str, help="Checkpoint file for prediction")
    parser.add_argument("--batch_norm", type=int, default=0,
                        help="add batch norm to neural network")
    parser.add_argument("--exp_name", type=str, default="default",
                        help="Name you experiment")
    parser.add_argument("--load_from_ckpt", type=int, default=0,
                        help="Start from a checkpoint for training")
    parser.add_argument("--write_dir", type=str, default="lightning_logs")
    parser.add_argument("--split", type=float, default=0.8,
                        help="Fraction of data for training and rest for validation")
    parser.add_argument("--template_list", type=str, nargs="+",
                        choices=list(template_to_idx_mapping.keys()),
                        default=list(template_to_idx_mapping.keys()),
                        help="Choose a list of atlases to train on")
    args = parser.parse_args()
    print("Hyper-parameters:")
    for k, v in vars(args).items():
        print("{} -> {}".format(k, v))

    if args.mode == "train":
        data_loader = vanillaDataLoader(
            batch_size=args.batch_size,
            split=args.split
        )
        # Create write dir
        write_dir = create_version_dir(
            os.path.join(args.write_dir, args.exp_name),
            prefix="run")
        args.write_dir = write_dir
        ckpt = ModelCheckpoint(
            dirpath=os.path.join(write_dir, "checkpoints"),
            monitor="val/f1_epoch",
            mode="max",
            every_n_epochs=100,
            save_top_k=1,
            auto_insert_metric_name=False,
            filename='epoch-{epoch:02d}-loss-{val/cls_loss:.3f}-f1={val/f1_epoch:.3f}-acc={val/acc_epoch:.3f}'
        )
        tb_logger = pl_loggers.TensorBoardLogger(write_dir, name="tb_logs")
        trainer = pl.Trainer(
            gpus=args.gpus,
            max_epochs=args.max_epochs,
            logger=tb_logger,
            log_every_n_steps=10,
            callbacks=[ckpt]
        )
        if args.load_from_ckpt == 1:
            model = MultiModalTrainer.load_from_checkpoint(checkpoint_path=args.ckpt)
        else:
            model = MultiModalTrainer(
                enc_hidden_layers=args.enc_hidden_layers,
                cls_layers=args.cls_layers,
                latent_dim=args.latent_dim,
                template_list=args.template_list,
                lr=args.lr,
                batch_norm=True if args.batch_norm == 0 else False,
                dropout=args.dropout
            )
        trainer.fit(
            model,
            train_dataloaders=data_loader.train_dataloader(),
            val_dataloaders=data_loader.val_dataloader()
        )


if __name__ == '__main__':
    start_time = time.time()
    cli_main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time elapsed {elapsed_time // 3600}h "
          f"{(elapsed_time % 3600) // 60}m "
          f"{((elapsed_time % 3600) % 60)}s")