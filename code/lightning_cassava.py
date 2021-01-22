import os
import cv2
import glob
import random
import shutil
import pickle
import json
import yaml
import pathlib
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.cuda import amp

import torchvision
from torchvision import models

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

from tqdm.notebook import tqdm
from collections import defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm

import wandb

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.loggers.wandb import WandbLogger

import warnings

warnings.filterwarnings("ignore")

import os, sys

if os.getcwd() in ["/kaggle/working", "/content"]:
    sys.path.append("./kaggle_Cassava/code")
else:
    sys.path.append(r"C:\Users\81908\MyGitHub\kaggle_Cassava\code")
    # sys.path.append(r"C:\Users\shingo\jupyter_notebook\pytorch_lightning_work\kaggle_Cassava\kaggle_Cassava\code")

from gem_pooling import GeM, GeMNet
from smooth_ce_loss import SmoothCrossEntropyLoss
from bi_tempered_loss import BiTemperedLoss
from symmetric_ce_loss import SymmetricCrossEntropyLoss
from mix_aug import cutmix, fmix, snapmix, SnapMixLoss
from sampling import (
    minor_class_over_sample,
    imblearn_over_sampling,
    imblearn_under_sampling,
)
from my_lr_scheduler import _CosineAnnealingWarmupRestarts
from balanced_batch_sampler import BalancedBatchSampler

# from gradcam_util import GradcamUtil
from util_torch import freeze_until, freeze_bn
from visualize import visualize_transpose, show_images, show_pred_diff_images
from sharpen import sharpen
from dataset_normalize_param import dataset_normalize_param


class CassavaDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        train: bool = True,
        transforms=None,
        name_mapping={
            0: "Cassava Bacterial Blight (CBB)",
            1: "Cassava Brown Streak Disease (CBSD)",
            2: "Cassava Green Mottle (CGM)",
            3: "Cassava Mosaic Disease (CMD)",
            4: "Healthy",
        },
    ):
        self.df = df
        self.train = train
        self.transforms = transforms
        self.name_mapping = name_mapping

    def __getitem__(self, index):
        im_path = self.df.iloc[index]["file_path"]
        x = cv2.imread(im_path, cv2.IMREAD_COLOR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

        if self.transforms:
            x = self.transforms(image=x)["image"]

        if self.train:
            y = self.df.iloc[index]["label"]
            y_soft = self.df.iloc[index][
                list(self.name_mapping.values())
            ].values.astype(np.float32)
            return {"x": x, "y": y, "y_soft": y_soft}
        else:
            return {"x": x}

    def __len__(self):
        return len(self.df)


class CassavaDataModule(pl.LightningDataModule):
    def __init__(
        self, train_df, valid_df, data_transforms, CFG,
    ):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.data_transforms = data_transforms
        self.CFG = CFG

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_dataset = CassavaDataset(
            self.train_df, train=True, transforms=self.data_transforms["train"],
        )

        self.valid_dataset = CassavaDataset(
            self.valid_df, train=True, transforms=self.data_transforms["valid"],
        )

        self.valid_tta_dataset = CassavaDataset(
            self.valid_df, train=True, transforms=self.data_transforms["test"],
        )  # test setのTTA条件

    def train_dataloader(self):
        if self.CFG.is_balanced_batch:
            assert (
                self.CFG.batch_size % self.CFG.n_classes == 0
            ), "batch_sizeはクラス数の整数倍にしないとダメ！！！"

            balanced_batch_sampler = BalancedBatchSampler(
                self.train_dataset, self.train_df["label"].values
            )

            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.CFG.batch_size,
                num_workers=self.CFG.num_workers,
                drop_last=True,
                sampler=balanced_batch_sampler,
            )
        else:
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.CFG.batch_size,
                num_workers=self.CFG.num_workers,
                drop_last=True,
                shuffle=True,
            )

        return train_loader

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.CFG.batch_size,
            num_workers=self.CFG.num_workers,
            shuffle=False,
        )

    def val_tta_dataloader(self):
        return DataLoader(
            self.valid_tta_dataset,
            batch_size=self.CFG.batch_size,
            num_workers=self.CFG.num_workers,
            shuffle=False,
        )


class CassavaLite(pl.LightningModule):
    def __init__(self, CFG):
        super().__init__()

        self.CFG = CFG

        self.use_amp = True  # apex amp を有効にする(16-bit mixed precision) https://pytorch-lightning.readthedocs.io/en/stable/lightning_module.html

        self.net = timm.create_model(CFG.arch, pretrained=True)

        if CFG.model_path is not None:
            # 学習済みモデルファイル指定ある場合ロード
            self.net.load_state_dict(torch.load(CFG.model_path))
            print("load pth:", CFG.model_path)

        if "eff" in CFG.arch:
            self.net.classifier = nn.Linear(
                self.net.classifier.in_features, CFG.n_classes
            )
            if CFG.gem_p > 0.0:
                self.net = GeMNet(
                    list(self.net.children())[:-2],
                    GeM(p=CFG.gem_p),
                    self.net.classifier.in_features,
                    CFG.n_classes,
                )
            self.feat_net = nn.Sequential(*list(self.net.children())[:-2])

        elif "rexnet" in CFG.arch:
            self.net.head.fc = nn.Linear(self.net.head.fc.in_features, CFG.n_classes)
            if CFG.gem_p > 0.0:
                self.net = GeMNet(
                    list(self.net.children())[:-1],
                    GeM(p=CFG.gem_p),
                    self.net.head.fc.in_features,
                    CFG.n_classes,
                )
            self.feat_net = nn.Sequential(*list(self.net.children())[:-1])

        elif "vit" in CFG.arch:
            self.net.head = nn.Linear(self.net.head.in_features, CFG.n_classes)
            self.feat_net = nn.Sequential(*list(self.net.children())[:-1])

        else:
            self.net.fc = nn.Linear(self.net.fc.in_features, CFG.n_classes)
            if CFG.gem_p > 0.0:
                self.net = GeMNet(
                    list(self.net.children())[:-2],
                    GeM(p=CFG.gem_p),
                    self.net.fc.in_features,
                    CFG.n_classes,
                )
            self.feat_net = nn.Sequential(
                *list(self.net.children())[:-2]
            )  # global_poolとfc層除く

    def forward(self, x):
        out = self.net(x)
        return out

    def feat_forward(self, x):
        """全結合層の直前の予測値取得用"""
        feat = self.feat_net(x)
        return feat.detach()  # 「detach()」はTensor型から勾配情報を抜いたものを取得

    def configure_optimizers(self):
        param_groups = self.parameters()

        if self.CFG.opt == "adam":
            optimizer = torch.optim.Adam(
                param_groups,
                lr=self.CFG.lr,
                weight_decay=self.CFG.weight_decay,
                amsgrad=False,
            )
        elif self.CFG.opt == "adabelief":
            optimizer = AdaBelief(
                param_groups, lr=CFG.lr, weight_decay=self.CFG.weight_decay
            )
        elif self.CFG.opt == "radam":
            optimizer = torch_optimizer.RAdam(
                param_groups, lr=self.CFG.lr, weight_decay=self.CFG.weight_decay
            )
        else:
            optimizer = torch.optim.AdamW(
                param_groups, lr=self.CFG.lr, weight_decay=self.CFG.weight_decay
            )

        if self.CFG.lr_scheduler == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.CFG.T_max, eta_min=self.CFG.min_lr
            )
        elif self.CFG.lr_scheduler == "CosineAnnealingWarmRestarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=self.CFG.T_max, T_mult=1, eta_min=self.CFG.min_lr
            )
        elif self.CFG.lr_scheduler == "CosineAnnealingWarmUpRestarts":
            scheduler = _CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=self.CFG.max_epochs,
                max_lr=self.CFG.lr,
                min_lr=self.CFG.min_lr,
                warmup_steps=1,
            )
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=2, gamma=0.1
            )

        return [optimizer], [scheduler]

    def loss(self, y_hat, y, mix_decision):
        if self.CFG.train_loss_name == "SmoothCrossEntropyLoss":
            loss_fn = SmoothCrossEntropyLoss(smoothing=self.CFG.smoothing).to(
                self.CFG.device
            )
        elif self.CFG.train_loss_name == "FocalCosineLoss":
            loss_fn = FocalCosineLoss(smoothing=self.CFG.smoothing).to(self.CFG.device)
        elif self.CFG.train_loss_name == "BiTemperedLoss":
            # labelsmoothing と mix_aug を半々にする
            # https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/209065
            if mix_decision < self.CFG.mix_decision_th:
                loss_fn = BiTemperedLoss(
                    t1=self.CFG.t1, t2=self.CFG.t2, smoothing=0.0
                ).to(self.CFG.device)
            else:
                loss_fn = BiTemperedLoss(
                    t1=self.CFG.t1, t2=self.CFG.t2, smoothing=self.CFG.smoothing
                ).to(self.CFG.device)
        elif self.CFG.train_loss_name == "SymmetricCrossEntropyLoss":
            if mix_decision < self.CFG.mix_decision_th:
                loss_fn = SymmetricCrossEntropyLoss(
                    num_classes=self.CFG.n_classes,
                    alpha=self.CFG.symmetric_alpha,
                    beta=self.CFG.symmetric_beta,
                    smoothing=0.0,
                ).to(self.CFG.device)
            else:
                loss_fn = SymmetricCrossEntropyLoss(
                    num_classes=self.CFG.n_classes,
                    alpha=self.CFG.symmetric_alpha,
                    beta=self.CFG.symmetric_beta,
                    smoothing=self.CFG.smoothing,
                ).to(self.CFG.device)
        else:
            loss_fn = nn.CrossEntropyLoss().to(self.CFG.device)

        # BiTemperedLoss, SymmetricCrossEntropyLoss はソフトラベル可能
        return loss_fn(y_hat, y)

    def training_step(self, batch, batch_idx):
        x, y, y_soft = batch["x"], batch["y"], batch["y_soft"]

        # cutmix/fmix/snapmix
        mix_decision = np.random.rand()
        if mix_decision < self.CFG.mix_decision_th:
            if self.CFG.mixmethod == "cutmix":
                x, y_mixs = cutmix(x, y.long(), self.CFG.mix_alpha)
                y_hat = self(x.float())
                loss = self.loss(y_hat, y_mixs[0], mix_decision) * y_mixs[
                    2
                ] + self.loss(y_hat, y_mixs[1], mix_decision) * (1.0 - y_mixs[2])

            elif self.CFG.mixmethod == "fmix":
                x, y_mixs = fmix(
                    x,
                    y.long(),
                    alpha=self.CFG.mix_alpha,
                    decay_power=5.0,
                    shape=(self.CFG.height, self.CFG.width),
                )
                y_hat = self(x.float())
                loss = self.loss(y_hat, y_mixs[0], mix_decision) * y_mixs[
                    2
                ] + self.loss(y_hat, y_mixs[1], mix_decision) * (1.0 - y_mixs[2])

            else:
                y_hat = self(x.float())
                if CFG.is_onehot_label:
                    loss = self.loss(y_hat, y_soft, mix_decision)
                else:
                    loss = self.loss(y_hat, y, mix_decision)

        else:
            y_hat = self(x.float())
            if self.CFG.is_onehot_label:
                loss = self.loss(y_hat, y_soft, mix_decision)
            else:
                loss = self.loss(y_hat, y, mix_decision)

        acc = accuracy(y_hat, y)

        # on_epoch=Tureでepoch単位の平均値を記録する
        # logger=Trueでtensorboardやwandbに記録する
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_idx):
        x, y, y_soft = batch["x"], batch["y"], batch["y_soft"]
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y).to(self.CFG.device)
        acc = accuracy(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss": loss, "val_acc": acc}
