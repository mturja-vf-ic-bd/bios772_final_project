import os
import time

import torch.nn as nn
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import torchmetrics

from src.dataloaders.ad_data_loaders import adDataLoader, get_length_of_cuts
from src.utils.data_utils import template_to_idx_mapping
from src.layers.MLP import MLP
from src.utils.folder_manager import create_version_dir


class mlpTrainer(pl.LightningModule):
    def __init__(self,
                 layers,
                 lr=1e-4,
                 activation=nn.ELU(),
                 batch_norm=False,
                 num_classes=3,
                 dropout=0.01):
        super(mlpTrainer, self).__init__()
        self.model = MLP(
            layers,
            prefix="mlp-net",
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )
        self.learning_rate = lr
        self.dropout=dropout
        self.activation=activation
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
    parser.add_argument("-l", "--layers",
                        nargs="+", type=int,
                        help="Hidden layer dimensions")
    parser.add_argument("-b", "--batch_size", nargs="?", type=int, default=1)
    parser.add_argument("-g", "--gpus", nargs="?", type=int, default=0)
    parser.add_argument("-m", "--max_epochs", nargs="?", type=int, default=100)
    parser.add_argument("-d", "--dropout", nargs="?",
                        type=float, default=0.01)
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--mode", choices=["train", "test"], default="test")
    parser.add_argument("--ckpt", type=str, help="Checkpoint file for prediction")
    parser.add_argument("--oversample", type=int, default=0,
                        help="Oversample using SMOTE")
    parser.add_argument("--batch_norm", type=int, default=0,
                        help="add batch norm to neural network")
    parser.add_argument("--exp_name", type=str, default="default",
                        help="Name you experiment")
    parser.add_argument("--write_dir", type=str, default="lightning_logs")
    parser.add_argument("--split", type=float, default=0.8,
                        help="Fraction of data for training and rest for validation")
    parser.add_argument("--template_list", type=str, nargs="+",
                        choices=list(template_to_idx_mapping.keys()),
                        default=list(template_to_idx_mapping.keys()),
                        help="Choose a list of templates to train on")
    args = parser.parse_args()
    print("Hyper-parameters:")
    for k, v in vars(args).items():
        print("{} -> {}".format(k, v))

    if args.mode == "train":
        data_loader = adDataLoader(
            batch_size=args.batch_size,
            template_list=args.template_list,
            oversample=True if args.oversample == 1 else False
        )
        # Create write dir
        write_dir = create_version_dir(
            os.path.join(args.write_dir, args.exp_name),
            prefix="run")
        args.write_dir = write_dir
        ckpt = ModelCheckpoint(
            dirpath=os.path.join(write_dir, "checkpoints"),
            monitor="val/cls_loss",
            every_n_epochs=20,
            save_top_k=10,
            auto_insert_metric_name=False,
            filename='epoch-{epoch:02d}-loss-{val/cls_loss:.3f}')
        tb_logger = pl_loggers.TensorBoardLogger(write_dir, name="tb_logs")
        trainer = pl.Trainer(
            gpus=args.gpus,
            max_epochs=args.max_epochs,
            logger=tb_logger,
            log_every_n_steps=10,
            callbacks=[ckpt]
        )
        input_dim = get_length_of_cuts(args.template_list)
        model = mlpTrainer(
            layers=[input_dim] + args.layers,
            lr=args.lr,
            batch_norm=True if args.batch_norm == 0 else False,
            dropout=args.dropout
        )
        trainer.fit(model,
                    train_dataloaders=data_loader.train_dataloader(),
                    val_dataloaders=data_loader.val_dataloader())


if __name__ == '__main__':
    start_time = time.time()
    cli_main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time elapsed {elapsed_time // 3600}h "
          f"{(elapsed_time % 3600) // 60}m "
          f"{((elapsed_time % 3600) % 60)}s")