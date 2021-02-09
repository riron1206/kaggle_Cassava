"""
stacking_cnmn2d のパラメータチューニング
notebookだとログ大量に出るのでpyから実行
"""
# Usage:
#  conda activate lightning
#  python C:\Users\81908\MyGitHub\kaggle_Cassava\code\objective_stacking_cnmn2d_cassava.py
#  python C:\Users\81908\MyGitHub\kaggle_Cassava\code\objective_stacking_cnmn2d_cassava.py -p objective_stacking_cnmn2d_cassava_params2

import os
import argparse
import shutil
import warnings
import optuna
import torch
import pytorch_lightning as pl
import numpy as np
import pandas as pd

from lightning_cassava_stacking import (
    train_stacking,
    pred_stacking,
    StackingConfig,
)

warnings.filterwarnings("ignore")

# 引数でパラメータ(pkl)の条件変える
parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--params_py", type=str, default="objective_stacking_cnmn2d_cassava_params"
)
args = parser.parse_args()

if args.params_py == "objective_stacking_cnmn2d_cassava_params":
    import params.objective_stacking_cnmn2d_cassava_params as params_py

    print(f"import objective_stacking_cnmn2d_cassava_params.py")
elif args.params_py == "objective_stacking_cnmn2d_cassava_params2":
    import params.objective_stacking_cnmn2d_cassava_params2 as params_py

    print(f"import objective_stacking_cnmn2d_cassava_params2.py")


max_epochs = 35
n_trials = 300
# n_trials = 2  # DEBUG
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


def objective(trial):
    CFG = StackingConfig()
    CFG.max_epochs = max_epochs
    CFG.patience = CFG.max_epochs // 3
    CFG.T_max = max_epochs
    CFG.T_0 = max_epochs
    CFG.n_classes = n_classes
    CFG.device = device
    CFG.num_workers = num_workers
    CFG.arch = arch

    kwargs_head = dict(
        n_features_list=[-1, n_classes],
        use_tail_as_out=True,
        drop_rate=trial.suggest_discrete_uniform("drop_rate", 0.1, 0.9, 0.1),
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

    CFG.weight_decay = trial.suggest_categorical("weight_decay", [1e-5, 1e-3, 1e-1])
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
