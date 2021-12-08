import os
import time

import torch.nn as nn
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import torchmetrics

from src.dataloaders.loader_utils import get_length_of_cuts, get_lengths_of_template
from src.dataloaders.multi_scale_loaders import schaeferMultiScaleLoader
from src.models.MultiScaleVAEClassifier import MultiScaleVAEClassifier
from src.utils.data_utils import template_to_idx_mapping
from src.utils.folder_manager import create_version_dir


class multiScaleVAETrainer(pl.LightningModule):
    def __init__(
            self,
            enc_layers,
            cls_layers,
            decoder_layer_list,
            activation: nn.modules.activation = nn.ELU(),
            batch_norm: bool = False,
            dropout: float = 0.01,
            lr=1e-4,
            num_classes=3,
            loss_w=None,
    ):
        super(multiScaleVAETrainer, self).__init__()
        self.model = MultiScaleVAEClassifier(
            enc_layers=enc_layers,
            cls_layers=cls_layers,
            decoder_layer_list=decoder_layer_list,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )
        self.learning_rate = lr
        self.dropout = dropout
        self.activation = activation
        self.loss_w = loss_w
        self.decoder_layer_list = decoder_layer_list
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

    def compute_loss(self, y_true, output, recon_true, weight, loss_weights, eps=1e-5):
        assert len(output["recon"]) == len(recon_true), "Size Mismatch in reconstruction"
        y_pred = output["y_pred"]
        loss_fn_cls = nn.CrossEntropyLoss(weight=weight)
        loss = {"cls_loss": loss_fn_cls(y_pred, y_true), "kld_loss": output["kld_loss"]}
        loss["total_loss"] = loss["cls_loss"] + loss_weights[0] * loss["kld_loss"]
        for i in range(len(self.decoder_layer_list)):
            mse_percent = ((y_pred[i] - y_true[i]) ** 2 / (y_true[i] + eps)).mean()
            loss[f"recon_{i}"] = mse_percent
            loss["total_loss"] += mse_percent * loss_weights[1]
        return loss

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
                                lr=self.learning_rate)

    def _common_step(self, batch, batch_idx, category="train"):
        x, recon_true, y = batch
        out = self.model(x)
        # Compute weight for each class
        weight = torch.FloatTensor(
            [len(y) * 1.0 / (y == 0).sum(), len(y) * 1.0 / (y == 1).sum(),
             len(y) * 1.0 / (y == 2).sum()]
        ).to(y.device)
        losses = self.compute_loss(y, out, recon_true, weight, self.loss_w)
        if category == "train":
            self.train_accuracy(torch.argmax(out["y_pred"], dim=-1), y)
            self.train_f1(torch.softmax(out["y_pred"], dim=-1), y)
        elif category == "val":
            self.val_accuracy(torch.argmax(out["y_pred"], dim=-1), y)
            self.val_f1(torch.softmax(out["y_pred"], dim=-1), y)
        self.add_log(losses, category)
        return losses["total_loss"]


def create_decoder_layers(enc_layers, target_template_list):
    return [enc_layers[::-1] + [get_lengths_of_template(target)] for target in target_template_list]


def cli_main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--enc_layers",
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
    parser.add_argument("--load_from_ckpt", type=int, default=0,
                        help="Start from a checkpoint for training")
    parser.add_argument("--write_dir", type=str, default="lightning_logs")
    parser.add_argument("--split", type=float, default=0.8,
                        help="Fraction of data for training and rest for validation")
    parser.add_argument("--input_template", type=str, choices=list(template_to_idx_mapping.keys()),
                        nargs="+", default="Input template list (usually a higher resolution template)")
    parser.add_argument("--cls_layers", type=int, nargs="+",
                        help="Classification layer dimensions (without the initial input layers)")
    parser.add_argument("--target_template_list", type=str, nargs="+",
                        choices=list(template_to_idx_mapping.keys()),
                        default=list(template_to_idx_mapping.keys()),
                        help="Choose a list of templates to train on")
    parser.add_argument("--loss_weights", type=float, nargs="+", help="Weights for kld and recon losses")
    args = parser.parse_args()
    print("Hyper-parameters:")
    device = "cuda:0" if args.gpus == 1 else "cpu"
    for k, v in vars(args).items():
        print("{} -> {}".format(k, v))

    if args.mode == "train":
        data_loader = schaeferMultiScaleLoader(
            batch_size=args.batch_size,
            input_template=args.input_template,
            target_template_list=args.target_template_list
        )
        # Create write dir
        write_dir = create_version_dir(
            os.path.join(args.write_dir, args.exp_name),
            prefix="run"
        )
        args.write_dir = write_dir
        ckpt = ModelCheckpoint(
            dirpath=os.path.join(write_dir, "checkpoints"),
            monitor="val/cls_loss",
            every_n_epochs=20,
            save_top_k=10,
            auto_insert_metric_name=False,
            filename='epoch-{epoch:02d}-loss-{val/cls_loss:.3f}'
        )
        tb_logger = pl_loggers.TensorBoardLogger(write_dir, name="tb_logs")
        trainer = pl.Trainer(
            gpus=args.gpus,
            max_epochs=args.max_epochs,
            logger=tb_logger,
            log_every_n_steps=10,
            callbacks=[ckpt]
        )
        input_dim = get_length_of_cuts(args.input_template)
        if args.load_from_ckpt == 1:
            model = multiScaleVAETrainer.load_from_checkpoint(checkpoint_path=args.ckpt)
        else:
            args.decoder_layer_list = create_decoder_layers(args.enc_layers, args.target_template_list)
            model = multiScaleVAETrainer(
                enc_layers=[input_dim] + args.enc_layers,
                cls_layers=args.cls_layers,
                decoder_layer_list=args.decoder_layer_list,
                lr=args.lr,
                batch_norm=True if args.batch_norm == 0 else False,
                dropout=args.dropout,
                loss_w=torch.FloatTensor(args.loss_weights).to(device)
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
