import os
import glob
import json
import pickle
import yaml
import numpy as np
import pandas as pd

from tqdm.notebook import tqdm

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import cv2
import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn

import timm
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet

from deit_models import deit_base_patch16_224

import warnings

warnings.filterwarnings("ignore")


class CassavaLite(pl.LightningModule):
    def __init__(self, CFG, source="timm"):
        super().__init__()
        self.CFG = CFG

        if source == "efficientnet-pytorch":
            self.net = EfficientNet.from_name(self.CFG.arch)
            self.net._fc = nn.Linear(self.net._fc.in_features, self.CFG.n_classes)

        elif source == "timm":
            self.net = timm.create_model(self.CFG.arch, pretrained=None)
            if "eff" in CFG.arch:
                self.net.classifier = nn.Linear(
                    self.net.classifier.in_features, self.CFG.n_classes
                )
            elif "rexnet" in self.CFG.arch:
                self.net.head.fc = nn.Linear(
                    self.net.head.fc.in_features, self.CFG.n_classes
                )
            elif "vit" in self.CFG.arch:
                self.net.head = nn.Linear(self.net.head.in_features, self.CFG.n_classes)
            else:
                self.net.fc = nn.Linear(self.net.fc.in_features, self.CFG.n_classes)

    def forward(self, x):
        out = self.net(x)
        return out


class CassavaLite4(pl.LightningModule):
    def __init__(self, CFG):
        super().__init__()

        self.net = timm.create_model(CFG.arch, pretrained=None)

        if "eff" in CFG.arch:
            self.net.classifier = nn.Linear(
                self.net.classifier.in_features, CFG.n_classes
            )
            self.feat_net = nn.Sequential(*list(self.net.children())[:-2])

        elif "rexnet" in CFG.arch:
            self.net.head.fc = nn.Linear(self.net.head.fc.in_features, CFG.n_classes)
            self.feat_net = nn.Sequential(*list(self.net.children())[:-1])

        elif "vit" in CFG.arch:
            self.net.head = nn.Linear(self.net.head.in_features, CFG.n_classes)
            self.feat_net = nn.Sequential(*list(self.net.children())[:-1])

        elif "deit" in CFG.arch:
            self.net = deit_base_patch16_224(
                pretrained=False
            )  # pretrained=True にすると失敗する。。。
            self.net.head = nn.Linear(self.net.head.in_features, CFG.n_classes)
            self.feat_net = nn.Sequential(*list(self.net.children())[:-1])

        else:
            self.net.fc = nn.Linear(self.net.fc.in_features, CFG.n_classes)
            self.feat_net = nn.Sequential(
                *list(self.net.children())[:-2]
            )  # global_poolとfc層除く

    def forward(self, x):
        out = self.net(x)
        return out


def get_models(model_paths, CFG=None):
    models = []
    for model_path in model_paths:
        if CFG.lite_type == "CassavaLite":
            model = CassavaLite().load_from_checkpoint(model_path).to(device)
        elif CFG.lite_type == "CassavaLite3":
            model = (
                CassavaLite3(CFG).load_from_checkpoint(model_path, CFG=CFG).to(device)
            )
        elif CFG.lite_type == "CassavaLite4":
            model = CassavaLite4().load_from_checkpoint(model_path).to(device)
        model.eval()
        models.append(model)
        print(f"{model_path} is loaded")
    return models


class CassavaDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, imfolder: str, train: bool = True, transforms=None
    ):
        self.df = df
        self.imfolder = imfolder
        self.train = train
        self.transforms = transforms

    def __getitem__(self, index):
        im_path = os.path.join(self.imfolder, self.df.iloc[index]["image_id"])
        x = cv2.imread(im_path, cv2.IMREAD_COLOR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

        if self.transforms:
            x = self.transforms(image=x)["image"]

        if self.train:
            y = self.df.iloc[index]["label"]
            return {"x": x, "y": y}
        else:
            return {"x": x}

    def __len__(self):
        return len(self.df)


def get_dataloader(test_df, img_dir, img_size, CFG):
    test_transform = A.Compose(
        [
            A.RandomResizedCrop(img_size, img_size),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # A.HueSaturationValue(
            #    hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
            # ),
            # A.RandomBrightnessContrast(
            #    brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
            # ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )
    test_dataset = CassavaDataset(
        test_df, img_dir, train=False, transforms=test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=False,
    )
    return test_loader


def inference_one_epoch_models(models, data_loader, device, CFG):
    preds_all = []
    with torch.no_grad():  # 勾配計算を無効にしてメモリ効率化
        for step, batch in enumerate(data_loader):
            imgs = batch["x"].to(device).float()

            # batchごとに複数のモデルの予測を平均してデータローダのループ回数を減らす
            p_mean = np.zeros((imgs.shape[0], CFG.n_classes))
            for model in models:
                p_batch = model(imgs)
                p_mean += torch.softmax(p_batch, 1).detach().cpu().numpy() / len(models)
            preds_all += [p_mean]

    preds_all = np.concatenate(preds_all, axis=0)
    return preds_all
