"""stacking_cnmn2d のパラメータチューニング"""
# Usage:
#  conda activate lightning
#  python C:\Users\81908\MyGitHub\kaggle_Cassava\code\objective_stacking_cnmn2d_cassava.py

import os
import argparse
import pickle
import json
import yaml
import shutil
import warnings
import optuna
import torch
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
from colorama import Fore
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_stacking import set_random_seed
from lightning_cassava import inference_one_epoch
from lightning_cassava_stacking import (
    StackingDataModule,
    LitStackingModel,
)

import params.objective_stacking_cnmn2d_cassava_params as params_py

warnings.filterwarnings("ignore")

max_epochs = 10
n_trials = 100
# n_trials = 2
n_classes = 5
num_workers = 0
device = "cuda" if torch.cuda.is_available() else "cpu"
arch = "cnmn2d"
out_dir = arch
os.makedirs(out_dir, exist_ok=True)

df = params_py.df
name_mapping = params_py.name_mapping
preds = params_py.preds
n_models = len(preds)
cnn_pred = np.stack(preds).transpose(1, 2, 0)
cnn_pred = cnn_pred.reshape(len(cnn_pred), 1, n_classes, n_models)


class Config:
    def __init__(self):
        self.seeds = [0]
        self.n_classes = n_classes
        self.max_epochs = max_epochs
        self.patience = max_epochs - 1
        self.n_splits = 5
        self.shuffle = True
        self.batch_size = 256
        self.accumulate_grad_batches = 1
        self.opt = "adam"
        self.lr_scheduler = "CosineAnnealingWarmRestarts"
        self.T_max = max_epochs
        self.T_0 = max_epochs
        self.lr = 1e-3
        self.min_lr = 1e-5
        self.weight_decay = 1e-5
        self.smoothing = 0.2
        self.train_loss_name = "BiTemperedLoss"
        self.t1 = 0.8
        self.t2 = 1.4
        self.monitor = "val_acc"
        self.out_dir = "."
        self.arch = ""
        self.mlp_params = None
        self.cnn1d_params = None
        self.cnn2d_params = None
        self.gauss_scale = 0.0
        self.cutmix_p = 0.0
        self.alpha = 1.0
        self.device = device
        self.num_workers = num_workers


def check_oof(y):
    y_ = Fore.YELLOW
    Y_oof = pickle.load(open(f"Y_pred.pkl", "rb"))
    oof = accuracy_score(y, Y_oof.values.argmax(1))
    oof_loss = log_loss(y, Y_oof.values)
    print(y_, f"oof:", round(oof, 4))
    print(y_, f"oof_loss:", round(oof_loss, 4))
    return oof, oof_loss


def train_stacking(x, y, StackingCFG, is_check_model=False):
    print(f"x.shape: {x.shape}")
    print(f"y.shape: {y.shape}")

    Y_pred = pd.DataFrame(
        np.zeros((df.shape[0], StackingCFG.n_classes)),
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

            x_train, x_valid = (
                x[train_idx],
                x[valid_idx],
            )
            y_train, y_valid = y[train_idx], y[valid_idx]
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
            early_stopping = EarlyStopping("val_loss", patience=StackingCFG.patience)
            if StackingCFG.monitor == "val_loss":
                model_checkpoint = ModelCheckpoint(
                    monitor="val_loss", save_top_k=1, mode="min"
                )
            else:
                model_checkpoint = ModelCheckpoint(
                    monitor="val_acc", save_top_k=1, mode="max"
                )
            trainer_params["callbacks"] = [model_checkpoint, early_stopping]

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
            ) / (len(StackingCFG.seeds) * StackingCFG.n_splits)
            val_loss = log_loss(y_valid, Y_pred.iloc[valid_idx])
            val_acc = (
                y_valid == np.argmax(Y_pred.iloc[valid_idx].values, axis=1)
            ).mean()
            print(f"fold {j} validation loss = {val_loss}")
            print(f"fold {j} validation accuracy = {val_acc}\n")

            del pl_model
            torch.cuda.empty_cache()  # 空いているキャッシュメモリを解放してGPUメモリの断片化を減らす

    pickle.dump(Y_pred, open("Y_pred.pkl", "wb"))
    oof, oof_loss = check_oof(df["label"].values)

    return oof, oof_loss


def objective(trial):
    CFG = Config()
    CFG.arch = arch

    kwargs_head = dict(
        n_features_list=[-1, n_classes],
        use_tail_as_out=True,
        drop_rate=trial.suggest_discrete_uniform("drop_rate", 0.1, 0.9, 0.2),
        use_bn=trial.suggest_categorical("use_bn", [True, False]),
        use_wn=trial.suggest_categorical("use_wn", [True, False]),
    )

    n_channel = trial.suggest_discrete_uniform("n_channel", 4, 32, 4)
    cnn2d_params = dict(
        n_models=n_models,
        n_classes=n_classes,
        n_channels_list=[1, int(n_channel)],
        use_bias=trial.suggest_categorical("use_bias", [True, False]),
        kwargs_head=kwargs_head,
    )
    CFG.cnn2d_params = cnn2d_params

    CFG.smoothing = trial.suggest_discrete_uniform("smoothing", 0.0, 0.3, 0.1)
    CFG.t1 = trial.suggest_discrete_uniform("t1", 0.7, 1.0, 0.1)
    CFG.t2 = trial.suggest_discrete_uniform("t2", 1.0, 1.3, 0.1)
    CFG.gauss_scale = trial.suggest_discrete_uniform("gauss_scale", 0.05, 0.25, 0.01)
    # CFG.cutmix_p = trial.suggest_categorical("cutmix_p", [1.0, 0.5, 0.2])
    # CFG.alpha = trial.suggest_categorical("alpha", [1.0, 0.5, 0.2, 5.0])

    print("-" * 100)
    print(f"CFG: {CFG.__dict__}")

    oof, oof_loss = train_stacking(cnn_pred, df["label"].values, CFG)

    return oof
    # return oof_loss


if __name__ == "__main__":
    study = optuna.create_study(
        study_name=f"study_{arch}",
        # storage=f"sqlite:///study.db",
        load_if_exists=True,
        direction="maximize",  # "minimize",
        sampler=optuna.samplers.TPESampler(seed=1),
    )
    study.optimize(objective, n_trials=n_trials)
    study.trials_dataframe().to_csv(f"{out_dir}/objective_history.csv", index=False)
    with open(f"{out_dir}/objective_best_params.txt", mode="w") as f:
        f.write(str(study.best_params))
    print(f"\nstudy.best_params:\n{study.best_params}")
    print(f"\nstudy.best_value:\n{study.best_value}\n")

    # ログ大量に出るから消す
    if os.path.exists("lightning_logs/"):
        shutil.rmtree("lightning_logs/")
