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
from sklearn.metrics import accuracy_score, log_loss
from colorama import Fore

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


name_mapping = {
    0: "Cassava Bacterial Blight (CBB)",
    1: "Cassava Brown Streak Disease (CBSD)",
    2: "Cassava Green Mottle (CGM)",
    3: "Cassava Mosaic Disease (CMD)",
    4: "Healthy",
}


class Config:
    def __init__(self):
        max_epochs = 11
        self.seeds = [42]
        self.n_classes = 5
        self.max_epochs = max_epochs
        self.patience = max_epochs - 1
        self.n_splits = 5
        self.shuffle = True
        self.batch_size = 16
        self.accumulate_grad_batches = (
            2  # https://www.kaggle.com/yingpengchen/pytorch-cldc-train-with-vit
        )
        self.height = 384
        self.width = 384
        self.arch = "vit_base_patch32_384"  # "vit_large_patch16_384"
        self.opt = "adam"
        self.lr_scheduler = "CosineAnnealingWarmRestarts"
        self.T_max = max_epochs
        self.T_0 = max_epochs
        self.lr = 1e-4
        self.min_lr = 1e-6
        self.weight_decay = 1e-6
        self.smoothing = (
            0.3  # https://www.kaggle.com/yingpengchen/pytorch-cldc-train-with-vit
        )
        self.train_loss_name = "BiTemperedLoss"
        self.t1 = 0.8
        self.t2 = 1.4
        self.n_tta = 5
        self.gem_p = 0  # GemPooling
        # self.mix_decision_th = 0.5  # cutmixなどの発生確率
        self.mix_decision_th = 0.0
        # self.mixmethod = "cutmix"
        self.mixmethod = ""
        self.mix_alpha = 1.0
        self.is_over_sample = False
        self.is_under_sample = False
        self.n_over = 0  # train set を倍々するか
        self.is_only_first_fold = False  # 1foldだけ学習するか
        self.is_onehot_label = False  # onehotの形式でラベル渡すか
        self.is_balanced_batch = (
            False  # balanced_batch 1epoch当たりのstep数は2倍以上になるので学習時間も増える
        )
        # self.wandb_project = "kaggle_cassava"  # wandb
        self.wandb_project = None
        self.is_lr_find = False  # 学習率探索
        self.is_old_compe_train = False  # 過去コンペのデータ追加
        self.monitor = "val_acc"  # ModelCheckpoint
        self.pretrained = True
        self.freeze_bn_epochs = 5  # freeze bn weights before epochs  https://www.kaggle.com/yingpengchen/pytorch-cldc-train-with-vit
        self.device = "cuda"
        self.num_workers = 0


def get_transforms(CFG):
    data_transforms = {
        "train": A.Compose(
            [
                # A.Resize(CFG.height, CFG.width),
                A.RandomResizedCrop(CFG.height, CFG.width),
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                A.HueSaturationValue(
                    hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                A.CoarseDropout(p=0.5),
                A.Cutout(p=0.5),
                ToTensorV2(),
            ],
            p=1.0,
        ),
        "valid": A.Compose(
            [
                # A.Resize(CFG.height, CFG.width),
                A.CenterCrop(CFG.height, CFG.width, p=1.0),
                A.Resize(CFG.height, CFG.width),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ],
            p=1.0,
        ),
        "test": A.Compose(
            [
                # A.Resize(CFG.height, CFG.width),
                A.RandomResizedCrop(CFG.height, CFG.width),
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                A.HueSaturationValue(
                    hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ],
            p=1.0,
        ),
    }
    return data_transforms


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def inference_one_epoch(model, data_loader, device):
    model.eval()
    image_preds_all = []
    with torch.no_grad():  # 勾配計算を無効にしてメモリ効率化
        for step, batch in enumerate(data_loader):
            imgs = batch["x"].to(device).float()
            model = model.to(device)
            image_preds = model(imgs)
            image_preds_all += [torch.softmax(image_preds, 1).detach().cpu().numpy()]
    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all


def check_oof(y):
    y_ = Fore.YELLOW
    Y_oof = pickle.load(open(f"Y_pred.pkl", "rb"))
    Y_tta_oof = pickle.load(open(f"Y_pred_tta.pkl", "rb"))
    oof = accuracy_score(y, Y_oof.values.argmax(1))
    oof_tta = accuracy_score(y, Y_tta_oof.values.argmax(1))
    oof_loss = log_loss(y, Y_oof.values)
    oof_loss_tta = log_loss(y, Y_tta_oof.values)
    print(y_, f"oof:", round(oof, 4))
    print(y_, f"oof_tta:", round(oof_tta, 4))
    print(y_, f"oof_loss:", round(oof_loss, 4))
    print(y_, f"oof_loss_tta:", round(oof_loss_tta, 4))
    return oof, oof_tta, oof_loss, oof_loss_tta


class CassavaDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, train: bool = True, transforms=None,
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
            y_soft = self.df.iloc[index][list(name_mapping.values())].values.astype(
                np.float32
            )
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

        self.net = timm.create_model(CFG.arch, pretrained=CFG.pretrained)

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


def run_train(
    df, data_transforms, CFG, wandb_logger=None, old_compe_df=None,
):
    print(f"df.shape:", df.shape)
    print("wandb_logger:", wandb_logger)
    print(f"CFG: {CFG.__dict__}")

    Y_pred = pd.DataFrame(
        np.zeros((df.shape[0], CFG.n_classes)),
        columns=name_mapping.values(),
        index=df.index,
    )
    Y_pred_tta = Y_pred.copy()

    for i in CFG.seeds:
        set_seed(seed=i)
        pl.seed_everything(i)

        cv = StratifiedKFold(n_splits=CFG.n_splits, shuffle=CFG.shuffle, random_state=i)

        for j, (train_idx, valid_idx) in enumerate(cv.split(df, df["label"])):

            if CFG.is_only_first_fold:
                if j > 0:
                    break

            train_df, valid_df = df.iloc[train_idx], df.iloc[valid_idx]

            if CFG.is_over_sample:
                train_df = imblearn_over_sampling(
                    train_df, train_df["label"], random_state=i, is_plot=False
                )
            elif CFG.is_under_sample:
                train_df = imblearn_under_sampling(
                    train_df, train_df["label"], random_state=i, is_plot=False
                )
            elif CFG.n_over > 0:
                # マイナークラスのみover sampling
                train_df = minor_class_over_sample(
                    train_df, n_over=CFG.n_over, is_plot=False
                )

            if CFG.is_old_compe_train:
                # 過去コンペのデータすべてtrainに入れる
                train_df = train_df.append(old_compe_df, ignore_index=True)

            dm = CassavaDataModule(train_df, valid_df, data_transforms, CFG)

            trainer_params = {
                "max_epochs": CFG.max_epochs,
                "deterministic": True,  # cudaの乱数固定
            }
            trainer_params[
                "accumulate_grad_batches"
            ] = CFG.accumulate_grad_batches  # 勾配をnバッチ分溜めてから誤差逆伝播
            early_stopping = EarlyStopping("val_loss", patience=CFG.patience)
            if CFG.monitor == "val_loss":
                model_checkpoint = ModelCheckpoint(
                    monitor="val_loss", save_top_k=1, mode="min"
                )
            else:
                model_checkpoint = ModelCheckpoint(
                    monitor="val_acc", save_top_k=1, mode="max"
                )
            trainer_params["callbacks"] = [model_checkpoint, early_stopping]

            if CFG.device == "cuda":
                trainer_params["gpus"] = 1
            if type(CFG.device) != str:
                trainer_params["tpu_cores"] = 1  # xm.xrt_world_size()
                trainer_params["precision"] = 16

            if CFG.wandb_project is not None:
                trainer = pl.Trainer(logger=wandb_logger, **trainer_params)
            else:
                trainer = pl.Trainer(**trainer_params)

            if CFG.is_lr_find:
                # 学習率探索
                lr_finder = trainer.tuner.lr_find(CassavaLite(CFG), dm)
                suggested_lr = lr_finder.suggestion()
                fig = lr_finder.plot()
                plt.title(f"suggested_lr: {suggested_lr}")
                fig.show()
                fig.savefig("lr_finder.png")
                break
            else:
                # 学習実行
                trainer.fit(CassavaLite(CFG), dm)

            shutil.copy(
                trainer.checkpoint_callback.best_model_path,
                f"model_seed_{i}_fold_{j}.ckpt",
            )

            # torch.save(model, PATH)

            # ---------- val predict ---------
            pretrained_model = CassavaLite(CFG).load_from_checkpoint(
                trainer.checkpoint_callback.best_model_path
            )
            with torch.no_grad():  # 勾配計算を無効してメモリ効率化
                for _ in range(CFG.n_tta):
                    Y_pred.iloc[valid_idx] += (
                        inference_one_epoch(
                            pretrained_model, dm.val_dataloader(), CFG.device
                        )
                        / CFG.n_tta
                    )
            val_loss = metrics.log_loss(valid_df.label.values, Y_pred.iloc[valid_idx])
            val_acc = (
                valid_df.label.values
                == np.argmax(Y_pred.iloc[valid_idx].values, axis=1)
            ).mean()
            print(f"fold {j} validation loss = {val_loss}")
            print(f"fold {j} validation accuracy = {val_acc}\n")

            with torch.no_grad():
                for _ in range(CFG.n_tta):
                    Y_pred_tta.iloc[valid_idx] += (
                        inference_one_epoch(
                            pretrained_model, dm.val_tta_dataloader(), device
                        )
                        / CFG.n_tta
                    )
            val_loss_tta = metrics.log_loss(
                valid_df.label.values, Y_pred_tta.iloc[valid_idx]
            )
            val_acc_tta = (
                valid_df.label.values
                == np.argmax(Y_pred_tta.iloc[valid_idx].values, axis=1)
            ).mean()
            print(f"fold {j} validation tta loss = {val_loss_tta}")
            print(f"fold {j} validation tta accuracy = {val_acc_tta}\n")

            print("-" * 100)

            del pretrained_model
            torch.cuda.empty_cache()  # 空いているキャッシュメモリを解放してGPUメモリの断片化を減らす

    pickle.dump(Y_pred, open("Y_pred.pkl", "wb"))
    pickle.dump(Y_pred_tta, open("Y_pred_tta.pkl", "wb"))

    if CFG.is_only_first_fold:
        oof, oof_tta = val_acc, val_acc_tta
        oof_loss, oof_loss_tta = val_loss, val_loss_tta
    else:
        oof, oof_tta, oof_loss, oof_loss_tta = check_oof(df["label"].values)

    if wandb_logger is not None:
        wandb_logger.log_metrics(
            {
                "oof": oof,
                "oof_tta": oof_tta,
                "oof_loss": oof_loss,
                "oof_loss_tta": oof_loss_tta,
            }
        )

    return oof, oof_tta, oof_loss, oof_loss_tta


def main(df):
    CFG = Config()
    with open("cfg.yaml", "w") as wf:
        yaml.dump(CFG.__dict__, wf)

    if CFG.wandb_project is not None:
        # 環境変数 WANDB_API_KEY に API キーをセット  https://github.com/MLHPC/wandb_tutorial
        os.environ["WANDB_API_KEY"] = "ace10b29622f5bd54e16d665a4b7c485e2094353"
        wandb_logger = WandbLogger(
            name=f"tf_efficientnet_b4_ns_fold10_{str(datetime.now().strftime('%Y/%m/%d_%H:%M'))}",
            project=CFG.wandb_project,
        )
        wandb_logger.log_hyperparams(params=CFG.__dict__)
    else:
        wandb_logger = None

    data_transforms = get_transforms(CFG)

    oof, oof_tta, oof_loss, oof_loss_tta = run_train(
        df, data_transforms, CFG, wandb_logger=wandb_logger
    )

    Y_pred = pickle.load(open("Y_pred.pkl", "rb"))
    oof_preds_df = pd.DataFrame(
        {
            "target": df["label"],
            "prediction": Y_pred.values.argmax(1),
            "logit": Y_pred.values.max(1),
            "file_path": df["file_path"],
        }
    )
    oof_preds_df.to_csv("oof_preds_df.csv", index=False)
    # display(oof_preds_df)

    Y_pred_tta = pickle.load(open("Y_pred_tta.pkl", "rb"))
    oof_preds_df = pd.DataFrame(
        {
            "target": df["label"],
            "prediction": Y_pred_tta.values.argmax(1),
            "logit": Y_pred_tta.values.max(1),
            "file_path": df["file_path"],
        }
    )
    oof_preds_df.to_csv("oof_preds_df_tta.csv", index=False)
    # display(oof_preds_df)

    sns.countplot(y=sorted(oof_preds_df["prediction"].map(name_mapping)), orient="v")
    plt.title("Prediction distribution")
    plt.show()

    print(
        metrics.classification_report(
            oof_preds_df["target"], oof_preds_df["prediction"]
        )
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        metrics.confusion_matrix(oof_preds_df["target"], oof_preds_df["prediction"]),
        annot=True,
        cmap="Blues",
    )
    plt.title("Oof confusion_matrix")
    plt.show()


if __name__ == "__main__":
    pass
