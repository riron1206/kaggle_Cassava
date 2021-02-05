import os
import glob
import json
import pickle
import yaml
import random
import cv2
import numpy as np
import pandas as pd
from colorama import Fore
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

import timm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.metrics.functional import accuracy

from gem_pooling import GeM, GeMNet
from deit_models import deit_base_patch16_224

import warnings

warnings.filterwarnings("ignore")

name_mapping = {
    0: "Cassava Bacterial Blight (CBB)",
    1: "Cassava Brown Streak Disease (CBSD)",
    2: "Cassava Green Mottle (CGM)",
    3: "Cassava Mosaic Disease (CMD)",
    4: "Healthy",
}


class Config:
    def __init__(self):
        self.gem_p = 0
        self.n_classes = 5
        self.height = 512
        self.width = 512
        self.arch = "resnest101e"
        self.n_tta = 5
        self.n_splits = 5
        self.seeds = [0]
        self.num_workers = 0
        self.device = "cuda"
        self.model_paths = []


def set_seed(seed=42):
    """Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def check_oof(y, Y_oof):
    y_ = Fore.YELLOW
    oof = accuracy_score(y, Y_oof.values.argmax(1))
    oof_loss = log_loss(y, Y_oof.values)
    print(y_, f"oof:", oof)
    print(y_, f"oof_loss:", oof_loss)
    return oof, oof_loss


######################## Dataset, Lightning Data Module ########################
class CassavaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool = True, transforms=None):
        self.df = df
        self.train = train
        self.transforms = transforms

    def __getitem__(self, index):
        im_path = self.df.iloc[index]["file_path"]
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


class CassavaDataModule(pl.LightningDataModule):
    def __init__(
        self, train_df, valid_df, data_transforms, CFG,
    ):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.data_transforms = data_transforms

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_dataset = CassavaDataset(
            self.train_df, train=True, transforms=self.data_transforms["train"]
        )

        self.valid_dataset = CassavaDataset(
            self.valid_df, train=True, transforms=self.data_transforms["valid"]
        )

        self.valid_tta_dataset = CassavaDataset(
            self.valid_df, train=True, transforms=self.data_transforms["test"]
        )  # test setのTTA条件

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=CFG.batch_size,
            num_workers=CFG.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=CFG.batch_size,
            num_workers=CFG.num_workers,
            shuffle=False,
        )

    def val_tta_dataloader(self):
        return DataLoader(
            self.valid_tta_dataset,
            batch_size=CFG.batch_size,
            num_workers=CFG.num_workers,
            shuffle=False,
        )


######################## LightningModule ########################
class CassavaLite(pl.LightningModule):
    def __init__(self, CFG):
        super().__init__()
        self.CFG = CFG
        self.use_amp = True

        from efficientnet_pytorch import EfficientNet

        if self.CFG.source == "efficientnet-pytorch":
            self.net = EfficientNet.from_name(self.CFG.arch)
            self.net._fc = nn.Linear(self.net._fc.in_features, self.CFG.n_classes)

        elif self.CFG.source == "timm":
            self.net = timm.create_model(self.CFG.arch, pretrained=None)
            if "eff" in self.CFG.arch:
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


class CassavaLite2(pl.LightningModule):
    def __init__(self, CFG):
        super().__init__()
        self.use_amp = True

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
        else:
            self.net.fc = nn.Linear(self.net.fc.in_features, CFG.n_classes)
            self.feat_net = nn.Sequential(
                *list(self.net.children())[:-2]
            )  # global_poolとfc層除く

    def forward(self, x):
        out = self.net(x)
        return out


class CassavaLite3(pl.LightningModule):
    def __init__(self, CFG):
        super().__init__()
        self.use_amp = True

        self.net = timm.create_model(CFG.arch, pretrained=None)

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


class CassavaLite4(pl.LightningModule):
    def __init__(self, CFG):
        super().__init__()
        self.use_amp = True

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


######################## for tta method ########################
def run_tta(
    df,
    img_size,
    batch_size,
    lite_type,
    n_tta,
    CFG,
    model_name,
    tta_type="tta_inference_one_epoch",
):
    set_seed(CFG.seeds[0])
    Y_pred = pd.DataFrame(
        np.zeros((df.shape[0], CFG.n_classes)),
        columns=name_mapping.values(),
        index=df.index,
    )
    cv = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seeds[0])
    for j, (train_idx, valid_idx) in enumerate(cv.split(df, df["label"])):

        if len(CFG.model_paths) <= j:
            break

        valid_df = df.iloc[valid_idx]
        valid_loader = get_dataloader(img_size, valid_df, batch_size, CFG=CFG)

        if df.shape != (5656, 3):
            # 現コンペデータの場合は1モデルだけで計算
            valid_model_paths = [CFG.model_paths[j]]
        else:
            print("--- old_comp_train ---")
            valid_model_paths = CFG.model_paths

        if tta_type == "tta_inference_one_epoch":
            print("tta_type:", "tta_inference_one_epoch")
            Y_pred.iloc[valid_idx] = tta_inference_one_epoch(
                valid_model_paths, lite_type, valid_loader, n_tta, CFG=CFG
            )
        else:
            print("tta_type:", tta_type)
            Y_pred.iloc[valid_idx] = tta_inference_one_epoch_models(
                valid_model_paths, lite_type, valid_loader, n_tta, CFG=CFG
            )

        val_acc = accuracy_score(
            valid_df["label"].values, Y_pred.iloc[valid_idx].values.argmax(1)
        )
        print(f"fold {j} acc:", val_acc)
        print("-" * 100)

    pickle.dump(Y_pred, open(f"{model_name}_Y_pred.pkl", "wb"))
    oof, oof_loss = check_oof(df["label"], Y_pred)


def tta_inference_one_epoch(model_paths, lite_type, test_loader, n_tta, CFG=None):
    test_preds = []
    models = get_models(model_paths, lite_type, CFG=CFG)
    for i, model in enumerate(models):
        for _ in range(n_tta):
            test_preds += [
                inference_one_epoch(model, test_loader, CFG.device)
                / (len(model_paths) * n_tta)
            ]
    for model in models:
        del model
        torch.cuda.empty_cache()
    test_preds = np.sum(test_preds, axis=0)
    return test_preds


def tta_inference_one_epoch_models(
    model_paths, lite_type, test_loader, n_tta, CFG=None
):
    test_preds = []
    models = get_models(model_paths, lite_type, CFG=CFG)
    for _ in range(n_tta):
        test_preds += [
            inference_one_epoch_models(models, test_loader, CFG.device, CFG.n_classes)
            / n_tta
        ]
    for model in models:
        del model
        torch.cuda.empty_cache()
    test_preds = np.sum(test_preds, axis=0)
    return test_preds


def inference_one_epoch(model, data_loader, device):
    model.eval()
    image_preds_all = []
    with torch.no_grad():
        for step, batch in tqdm(enumerate(data_loader)):
            imgs = batch["x"].to(device).float()
            model = model.to(device)
            image_preds = model(imgs)
            image_preds_all += [torch.softmax(image_preds, 1).detach().cpu().numpy()]
    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all


def inference_one_epoch_models(models, data_loader, device, n_classes):
    preds_all = []
    with torch.no_grad():  # 勾配計算を無効にしてメモリ効率化
        for step, batch in tqdm(enumerate(data_loader)):
            imgs = batch["x"].to(device).float()
            # batchごとに複数のモデルの予測を平均してデータローダのループ回数を減らす
            p_mean = np.zeros((imgs.shape[0], n_classes))
            for model in models:
                model = model.to(device)
                model.eval()
                p_batch = model(imgs)
                p_mean += torch.softmax(p_batch, 1).detach().cpu().numpy() / len(models)
            preds_all += [p_mean]
    preds_all = np.concatenate(preds_all, axis=0)
    return preds_all


def get_models(model_paths, lite_type, CFG=None):
    models = []
    for model_path in model_paths:
        if lite_type == "CassavaLite":
            model = (
                CassavaLite(CFG)
                .load_from_checkpoint(model_path, CFG=CFG)
                .to(CFG.device)
            )
        elif lite_type == "CassavaLite2":
            model = (
                CassavaLite2(CFG)
                .load_from_checkpoint(model_path, CFG=CFG)
                .to(CFG.device)
            )
        elif lite_type == "CassavaLite3":
            model = (
                CassavaLite3(CFG)
                .load_from_checkpoint(model_path, CFG=CFG)
                .to(CFG.device)
            )
        elif lite_type == "CassavaLite4":
            model = (
                CassavaLite4(CFG)
                .load_from_checkpoint(model_path, CFG=CFG)
                .to(CFG.device)
            )
        model.eval()
        models.append(model)
        print(f"{model_path} is loaded")
    return models


def get_dataloader(img_size, valid_df, batch_size, CFG=None):
    if img_size < 500:
        test_transform = A.Compose(
            [
                # A.CenterCrop(img_size, img_size,
                #             #p=0.5
                #            ),
                A.RandomResizedCrop(
                    img_size,
                    img_size,
                    #                    p=0.5
                ),
                # A.Resize(img_size, img_size),
                # A.Transpose(p=0.5),
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
    else:
        test_transform = A.Compose(
            [
                A.CenterCrop(
                    img_size,
                    img_size,
                    # p=0.5
                ),
                # A.RandomResizedCrop(img_size, img_size,
                #                    p=0.5
                #                   ),
                # A.Resize(img_size, img_size),
                # A.Transpose(p=0.5),
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
    dataset = CassavaDataset(valid_df, train=False, transforms=test_transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=CFG.num_workers, shuffle=False
    )
    return loader
