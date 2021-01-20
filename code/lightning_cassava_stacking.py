import os
import sys
import yaml
import numpy as np
import pandas as pd

import pickle
from sklearn.metrics import accuracy_score, log_loss
from colorama import Fore

y_ = Fore.YELLOW

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.cuda import amp

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


if os.getcwd() in ["/kaggle/working", "/content"]:
    sys.path.append("./kaggle_Cassava/code")
else:
    sys.path.append(r"C:\Users\81908\MyGitHub\kaggle_Cassava\code")

from mix_aug_table import cutmix_for_tabular
from smooth_ce_loss import SmoothCrossEntropyLoss
from bi_tempered_loss import BiTemperedLoss
from pytorch_stacking import (
    MLP,
    CNNStacking1d,
    CNNStacking2d,
    GCNStacking,
    set_random_seed,
)


class StackingDatasetMLP(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        ## 過学習防ぐためにガウシアンノイズ加算
        ## https://www.kaggle.com/c/stanford-covid-vaccine/discussion/189709
        # if noise_scale > 0.0:
        #    x = x + np.random.normal(0.0, scale=noise_scale)

        return {"x": x, "y": y}

    def __len__(self):
        return len(self.y)


class StackingDatasetCNN(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.reset_model_order()

    def reset_model_order(self):
        self.model_order = np.arange(self.x.shape[-1])

    def shuffle_model_order(self, seed):
        np.random.seed(seed)
        self.model_order = np.random.permutation(self.model_order)

    def __getitem__(self, index):
        x = self.x[index][..., self.model_order]
        y = self.y[index]

        ## 過学習防ぐためにガウシアンノイズ加算
        ## https://www.kaggle.com/c/stanford-covid-vaccine/discussion/189709
        # if noise_scale > 0.0:
        #    x = x + np.random.normal(0.0, scale=noise_scale)

        return {"x": x, "y": y}

    def __len__(self):
        return len(self.y)


class StackingDataModule(pl.LightningDataModule):
    def __init__(
        self, x_train, x_valid, y_train, y_valid, CFG, stacking_type="mlp", tmp_seed=0
    ):
        super().__init__()
        self.x_train = x_train
        self.x_valid = x_valid
        self.y_train = y_train
        self.y_valid = y_valid
        self.CFG = CFG
        self.stacking_type = stacking_type
        self.tmp_seed = tmp_seed

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if self.stacking_type == "mlp":
            self.train_dataset = StackingDatasetMLP(self.x_train, self.y_train)
            self.val_dataset = StackingDatasetMLP(self.x_valid, self.y_valid)
        else:
            self.train_dataset = StackingDatasetCNN(self.x_train, self.y_train)
            self.val_dataset = StackingDatasetCNN(self.x_valid, self.y_valid)

            # shffule order モデルの並び順が性能に影響するので順番入れ替え
            self.train_dataset.reset_model_order()
            self.train_dataset.shuffle_model_order(self.tmp_seed)
            self.val_dataset.reset_model_order()
            self.val_dataset.shuffle_model_order(self.tmp_seed)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.CFG.batch_size,
            shuffle=self.CFG.shuffle,
            num_workers=self.CFG.num_workers,
        )
        print(f"train_loader: {len(train_loader)}")
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.CFG.batch_size,
            shuffle=False,
            num_workers=self.CFG.num_workers,
        )
        print(f"val_loader: {len(val_loader)}")
        return val_loader


class LitStackingModel(pl.LightningModule):
    def __init__(self, CFG):
        super().__init__()
        self.CFG = CFG

        self.use_amp = True  # apex amp を有効にする(16-bit mixed precision) https://pytorch-lightning.readthedocs.io/en/stable/lightning_module.html

        if "mlp" in CFG.arch:
            self.net = MLP(**CFG.mlp_params)
        elif "cnmn1d" in CFG.arch:
            self.net = CNNStacking1d(**CFG.cnn1d_params)
        elif "cnmn2d" in CFG.arch:
            self.net = CNNStacking2d(**CFG.cnn2d_params)

    def forward(self, x):
        out = self.net(x)
        return out

    def configure_optimizers(self):
        if self.CFG.opt == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.CFG.lr,
                weight_decay=self.CFG.weight_decay,
                amsgrad=False,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.CFG.lr, weight_decay=self.CFG.weight_decay
            )

        if self.CFG.lr_scheduler == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.CFG.T_max, eta_min=self.CFG.min_lr
            )
        elif self.CFG.lr_scheduler == "CosineAnnealingWarmRestarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=self.CFG.T_max, T_mult=1, eta_min=self.CFG.min_lr
            )
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=2, gamma=0.1
            )

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch["x"].float(), batch["y"]

        if CFG.gauss_scale > 0.0:
            # 過学習防ぐためにガウシアンノイズ加算
            # https://www.kaggle.com/c/stanford-covid-vaccine/discussion/189709
            x = x + np.random.normal(0.0, scale=CFG.gauss_scale)  # 平均=0, 標準偏差はパラメータで変更

        if CFG.cutmix_p > 0.0:
            # cutmix for table
            x, y = cutmix_for_tabular(
                x, y, alpha=CFG.alpha, p=CFG.cutmix_p, random_state=None
            )

        if self.CFG.train_loss_name == "SmoothCrossEntropyLoss":
            loss_fn = SmoothCrossEntropyLoss(smoothing=self.CFG.smoothing).to(
                self.CFG.device
            )
        elif self.CFG.train_loss_name == "BiTemperedLoss":
            loss_fn = BiTemperedLoss(
                t1=self.CFG.t1, t2=self.CFG.t2, smoothing=self.CFG.smoothing
            ).to(self.CFG.device)
        # else:
        #    loss_fn = nn.CrossEntropyLoss().to(self.CFG.device)

        y_hat = self(x.float())
        loss = loss_fn(y_hat, y)

        acc = accuracy(y_hat, y)
        self.log(
            "train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["x"].float(), batch["y"]
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        acc = accuracy(y_hat, y)
        self.log("val_acc", acc, prog_bar=True, logger=True),
        self.log("val_loss", loss, prog_bar=True, logger=True)
