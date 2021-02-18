import os
import sys
import yaml
import shutil
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    classification_report,
    confusion_matrix,
    f1_score,
)
from colorama import Fore

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

# from mix_aug import cutmix
from mix_aug_table import cutmix_for_tabular, cutmix, mixup
from smooth_ce_loss import SmoothCrossEntropyLoss
from bi_tempered_loss import BiTemperedLoss
from pytorch_stacking import (
    MLP,
    CNNStacking1d,
    CNNStacking2d,
    GCNStacking,
    set_random_seed,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


class StackingDatasetMLP(Dataset):
    def __init__(self, x, y=None):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        if self.y is not None:
            return {"x": self.x[index], "y": self.y[index]}
        else:
            return {"x": self.x[index]}

    def __len__(self):
        return len(self.x)


class StackingDatasetCNN(Dataset):
    def __init__(self, x, y=None):
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
        if self.y is not None:
            return {"x": x, "y": self.y[index]}
        else:
            return {"x": x}

    def __len__(self):
        return len(self.x)


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
        # print([_y for _y in y_train])

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        if self.CFG.cls3_undersample_rate > 0.0:
            # cls3をundersampleing
            _y = pd.DataFrame(self.y_train)
            del_idx = _y[_y[0] == 3].sample(frac=self.CFG.cls3_undersample_rate).index
            _y = _y.drop(index=del_idx)
            idx = _y.index
            self.y_train = _y[0].values
            self.x_train = self.x_train[idx]
            print(
                f"cls3 undersampleing x, y shape: {self.x_train.shape}, {self.y_train.shape}"
            )

        # バイナリ用
        if self.CFG.train_n_classes == 2 and self.CFG.n_over > 0:
            _y = pd.DataFrame(self.y_train)
            _y[_y[0] == 1].values
            idx = _y[_y[0] == 1].index
            for ii in range(self.CFG.n_over):
                self.x_train = np.vstack([self.x_train, self.x_train[idx]])
                self.y_train = np.concatenate([self.y_train, self.y_train[idx]])
            print(
                f"n_over={self.CFG.n_over} x, y.shape:",
                self.x_train.shape,
                self.y_train.shape,
            )

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
        # print(f"train_loader: {len(train_loader)}")
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.CFG.batch_size,
            shuffle=False,
            num_workers=self.CFG.num_workers,
        )
        # print(f"val_loader: {len(val_loader)}")
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
        x, y = batch["x"].float(), batch["y"].long()

        if self.CFG.gauss_scale > 0.0:
            # 過学習防ぐためにガウシアンノイズ加算
            # https://www.kaggle.com/c/stanford-covid-vaccine/discussion/189709
            # 平均=0, 標準偏差はパラメータで変更
            # g_noise = np.random.normal(0.0, scale=self.CFG.gauss_scale, size=x.shape)  # gpuだとエラー
            g_noise = torch.normal(mean=0.0, std=self.CFG.gauss_scale, size=x.shape)
            x = x + g_noise.to(self.CFG.device)
            # x = torch.softmax(x, -1)  # 確率値に戻す 0.2-1.9 にしかならなくなるからやめる

        # print("\ny before", y)
        if self.CFG.cutmix_p > 0.0:
            # onehot
            y_onehot = torch.eye(self.CFG.train_n_classes)[y]
            if x.ndim == 2:
                # cutmix for table
                x, y_onehot = cutmix_for_tabular(
                    x,
                    y_onehot,
                    alpha=self.CFG.alpha,
                    p=self.CFG.cutmix_p,
                    random_state=None,
                )
                # x, y_onehot = mixup(x, y_onehot, alpha=self.CFG.alpha, p=self.CFG.cutmix_p, random_state=None,)
            else:
                try:
                    x, y_onehot = cutmix(
                        x,
                        y_onehot,
                        alpha=self.CFG.alpha,
                        p=self.CFG.cutmix_p,
                        random_state=None,
                    )  # mix_aug_table.py はなぜか学習途中でエラーになることがある np.random.randint(h - r_h) が原因ぽいが...
                except:
                    print("Error cutmix....")
                # print("x cutmix", x)
                # print("y cutmix", y_onehot)
            y_onehot = y_onehot.to(self.CFG.device)

        if self.CFG.train_loss_name == "SmoothCrossEntropyLoss":
            loss_fn = SmoothCrossEntropyLoss(smoothing=self.CFG.smoothing).to(
                self.CFG.device
            )
        elif self.CFG.train_loss_name == "BiTemperedLoss":
            loss_fn = BiTemperedLoss(
                t1=self.CFG.t1, t2=self.CFG.t2, smoothing=self.CFG.smoothing
            ).to(self.CFG.device)
        else:
            loss_fn = nn.CrossEntropyLoss().to(self.CFG.device)

        y_hat = self(x.float())
        # print(y_hat)

        acc = accuracy(y_hat, y)
        # print(acc, y_hat.shape, y.shape)

        if self.CFG.cutmix_p > 0.0:
            loss = loss_fn(y_hat, y_onehot)
        else:
            loss = loss_fn(y_hat, y.long())

        self.log(
            "train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["x"].float(), batch["y"].long()
        # print(x, y)
        y_hat = self(x)
        # print(y_hat)
        # print(y_hat.shape)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        acc = accuracy(y_hat, y)
        self.log("val_acc", acc, prog_bar=True, logger=True),
        self.log("val_loss", loss, prog_bar=True, logger=True)


def inference_one_epoch(model, data_loader, device):
    model.eval()
    preds_all = []
    with torch.no_grad():  # 勾配計算を無効にしてメモリ効率化
        for step, batch in enumerate(data_loader):
            x = batch["x"].to(device).float()
            model = model.to(device)
            preds = model(x)
            preds_all += [torch.softmax(preds, 1).detach().cpu().numpy()]
    preds_all = np.concatenate(preds_all, axis=0)
    return preds_all


def check_oof(y):
    y_ = Fore.YELLOW
    Y_oof = pickle.load(open(f"Y_pred.pkl", "rb"))
    oof = accuracy_score(y, Y_oof.values.argmax(1))
    oof_loss = log_loss(y, Y_oof.values)
    print(y_, f"oof:", round(oof, 4))
    print(y_, f"oof_loss:", round(oof_loss, 4))
    print(y_, classification_report(y, Y_oof.values.argmax(1)))
    print(y_, "cfm:\n", confusion_matrix(y, Y_oof.values.argmax(1)))
    return oof, oof_loss


def train_stacking(
    x,
    y,
    StackingCFG,
    is_check_model=False,
    add_train_x=None,
    add_train_y=None,
    noise_idx=None,
):
    print(f"x.shape: {x.shape}")
    print(f"y.shape: {y.shape}")

    Y_pred = pd.DataFrame(
        np.zeros((len(y), StackingCFG.train_n_classes)),
        # columns=name_mapping.values(),
        # index=df.index,
    )
    for i in StackingCFG.seeds:
        print(f"---------------------------- seed: {i} ----------------------------")
        set_random_seed(seed=i)
        pl.seed_everything(i)

        cv = StratifiedKFold(
            n_splits=StackingCFG.n_splits, shuffle=StackingCFG.shuffle, random_state=i
        )

        for j, (train_idx, valid_idx) in enumerate(cv.split(y, y)):
            print(f"train_idx, valid_idx: {len(train_idx)}, {len(valid_idx)}")

            if noise_idx is not None:
                # ノイズとおぼしきサンプルtrainから除く
                train_idx = list(set(train_idx) - set(noise_idx))
                print("denoise train_idx", len(train_idx))

            x_train, x_valid = (
                x[train_idx],
                x[valid_idx],
            )
            y_train, y_valid = y[train_idx], y[valid_idx]

            if add_train_x is not None:
                # trainに追加。pseudo label用
                x_train = np.vstack((x_train, add_train_x))
                y_train = np.concatenate([y_train, add_train_y])

            dm = StackingDataModule(
                x_train, x_valid, y_train, y_valid, StackingCFG, tmp_seed=i
            )

            trainer_params = {
                "max_epochs": StackingCFG.max_epochs,
                "deterministic": True,  # cudaの乱数固定
            }
            trainer_params[
                "accumulate_grad_batches"
            ] = StackingCFG.accumulate_grad_batches  # 勾配をnバッチ分溜めてから誤差逆伝播

            if StackingCFG.monitor == "val_loss":
                model_checkpoint = ModelCheckpoint(
                    monitor="val_loss", save_top_k=1, mode="min"
                )
                early_stopping = EarlyStopping(
                    monitor="val_loss", patience=StackingCFG.patience, mode="min"
                )
            else:
                model_checkpoint = ModelCheckpoint(
                    monitor="val_acc", save_top_k=1, mode="max"
                )
                early_stopping = EarlyStopping(
                    monitor="val_acc", patience=StackingCFG.patience, mode="max"
                )
            trainer_params["callbacks"] = [model_checkpoint, early_stopping]
            if StackingCFG.device == "cuda":
                trainer_params["gpus"] = 1

            trainer = pl.Trainer(**trainer_params)
            pl_model = LitStackingModel(StackingCFG)
            if is_check_model:
                print(pl_model)
            trainer.fit(pl_model, dm)

            shutil.copy(
                trainer.checkpoint_callback.best_model_path,
                f"{StackingCFG.out_dir}/model_seed_{i}_fold_{j}.ckpt",
            )

            # ---------- val predict ---------
            pl_model = pl_model.load_from_checkpoint(
                trainer.checkpoint_callback.best_model_path, CFG=StackingCFG
            )
            Y_pred.iloc[valid_idx] += inference_one_epoch(
                pl_model, dm.val_dataloader(), device
            ) / len(StackingCFG.seeds)
            val_loss = log_loss(y_valid, Y_pred.iloc[valid_idx])
            val_acc = (
                y_valid == np.argmax(Y_pred.iloc[valid_idx].values, axis=1)
            ).mean()
            print(f"fold {j} validation loss = {val_loss}")
            print(f"fold {j} validation accuracy = {val_acc}\n")

            del pl_model
            torch.cuda.empty_cache()  # 空いているキャッシュメモリを解放してGPUメモリの断片化を減らす

        # _, _ = check_oof(y)
        print(f"fold {i} mean acc:", (y == np.argmax(Y_pred.values, axis=1)).mean())

    pickle.dump(Y_pred, open("Y_pred.pkl", "wb"))
    oof, oof_loss = check_oof(y)

    return oof, oof_loss


def pred_stacking(x, StackingCFG, y=None):
    """stackingモデルで予測"""
    if StackingCFG.arch == "mlp":
        test_dataset = StackingDatasetMLP(x, y)
    else:
        test_dataset = StackingDatasetCNN(x, y)

    test_preds = np.zeros((len(x), StackingCFG.train_n_classes))

    for i in StackingCFG.seeds:
        print(f"seed: {i}")
        set_random_seed(seed=i)
        pl.seed_everything(i)

        if StackingCFG.arch != "mlp":
            test_dataset.reset_model_order()
            test_dataset.shuffle_model_order(i)

        for j in range(StackingCFG.n_splits):
            test_loader = DataLoader(
                test_dataset,
                batch_size=StackingCFG.batch_size,
                shuffle=False,
                num_workers=StackingCFG.num_workers,
            )

            ckpt = f"{StackingCFG.out_dir}/model_seed_{i}_fold_{j}.ckpt"
            pl_model = LitStackingModel(StackingCFG).load_from_checkpoint(
                ckpt, CFG=StackingCFG
            )
            print(f"load ckpt: {ckpt}")

            test_preds += inference_one_epoch(
                pl_model, test_loader, StackingCFG.device
            ) / (len(StackingCFG.seeds) * StackingCFG.n_splits)

    print(test_preds.argmax(axis=1))
    if y is not None:
        print("oof mean:", accuracy_score(y, test_preds.argmax(1)))

    return test_preds


class StackingConfig:
    def __init__(self):
        self.seeds = [0]
        self.n_classes = 5  # 特徴量の次元。要は特徴量データフレームの列数にしないとだめ。変数名ややこしいがpytorch_stacking.py のstackingモデルの関係からこうなってる
        self.train_n_classes = 5  # 予測するクラス数
        self.max_epochs = 200
        self.patience = 50
        self.n_splits = 5
        self.shuffle = True
        self.batch_size = 256
        self.accumulate_grad_batches = 1
        self.opt = "adam"
        self.lr_scheduler = "CosineAnnealingWarmRestarts"
        self.T_max = self.max_epochs
        self.T_0 = self.max_epochs
        self.lr = 0.1
        self.min_lr = 1e-3
        self.weight_decay = 1e-5
        self.smoothing = 0.2
        self.train_loss_name = "BiTemperedLoss"
        self.t1 = 1.0
        self.t2 = 1.0
        self.monitor = "val_acc"
        self.out_dir = "."
        self.arch = ""
        self.mlp_params = None
        self.cnn1d_params = None
        self.cnn2d_params = None
        self.gauss_scale = 0.1
        self.cutmix_p = 0.0
        self.alpha = 1.0
        self.cls3_undersample_rate = 0.0
        self.n_over = 0  # マイナークラスn倍
        self.device = device
        self.num_workers = 0
