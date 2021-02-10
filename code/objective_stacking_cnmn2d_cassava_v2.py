"""
stacking_cnmn2d のパラメータチューニング
notebookだとログ大量に出るのでpyから実行
"""
# Usage:
#  conda activate lightning
#  python C:\Users\81908\MyGitHub\kaggle_Cassava\code\objective_stacking_cnmn2d_cassava_v2.py
#  python C:\Users\81908\MyGitHub\kaggle_Cassava\code\objective_stacking_cnmn2d_cassava_v2.py -p objective_stacking_cnmn2d_cassava_params2

import os
import argparse
import shutil
import warnings
import optuna
import torch
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from lightning_cassava_stacking import (
    train_stacking,
    pred_stacking,
    StackingConfig,
)

warnings.filterwarnings("ignore")

# 引数でパラメータ(pkl)の条件変える
parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--params_py", type=str, default="objective_stacking_cnmn2d_cassava_params3"
)
args = parser.parse_args()

old_preds = None
old_df = None
is_objective_2019 = False
is_pseudo_2019 = False
if args.params_py == "objective_stacking_cnmn2d_cassava_params":
    import params.objective_stacking_cnmn2d_cassava_params as params_py

    print(f"------- import objective_stacking_cnmn2d_cassava_params.py -------")

elif args.params_py == "objective_stacking_cnmn2d_cassava_params2":
    import params.objective_stacking_cnmn2d_cassava_params2 as params_py

    print(f"------- import objective_stacking_cnmn2d_cassava_params2.py -------")

elif args.params_py == "objective_stacking_cnmn2d_cassava_params3":
    import params.objective_stacking_cnmn2d_cassava_params3 as params_py

    print(f"------- import objective_stacking_cnmn2d_cassava_params3.py -------")
    old_preds = params_py.old_preds
    old_df = params_py.old_df
    is_objective_2019 = params_py.is_objective_2019
    is_pseudo_2019 = params_py.is_pseudo_2019

elif args.params_py == "objective_stacking_cnmn2d_cassava_params4":
    import params.objective_stacking_cnmn2d_cassava_params4 as params_py

    print(f"------- import objective_stacking_cnmn2d_cassava_params4.py -------")
    old_preds = params_py.old_preds
    old_df = params_py.old_df
    is_objective_2019 = params_py.is_objective_2019
    is_pseudo_2019 = params_py.is_pseudo_2019

elif args.params_py == "objective_stacking_cnmn2d_cassava_params5":
    import params.objective_stacking_cnmn2d_cassava_params5 as params_py

    print(f"------- import objective_stacking_cnmn2d_cassava_params5.py -------")


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

if is_pseudo_2019:
    # 2019年のpklで pseudo label
    preds = [np.vstack((preds[ii], old_preds[ii])) for ii in range(len(preds))]
    cnn_pred = np.stack(preds).transpose(1, 2, 0)
    cnn_pred = cnn_pred.reshape(len(cnn_pred), 1, n_classes, n_models)
    y = np.concatenate([df["label"].values, old_df["label"].values])
else:
    cnn_pred = np.stack(preds).transpose(1, 2, 0)
    cnn_pred = cnn_pred.reshape(len(cnn_pred), 1, n_classes, n_models)
    y = df["label"].values


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
    CFG.cutmix_p = trial.suggest_categorical("cutmix_p", [0.0, 1.0, 0.5, 0.2])
    CFG.alpha = trial.suggest_categorical("alpha", [1.0, 0.5, 0.2, 5.0])

    print("-" * 100)
    print(f"CFG: {CFG.__dict__}")

    oof, oof_loss = train_stacking(cnn_pred, y, CFG)

    # 2019年のpklも最適化のデータに使う場合
    if is_objective_2019:
        old_x = np.array(old_preds).transpose(1, 2, 0)
        old_x = old_x.reshape(len(old_x), 1, CFG.n_classes, len(old_preds))
        old_pred_cnn2d = pred_stacking(old_x, CFG)
        old_acc = accuracy_score(old_df["label"].values, old_pred_cnn2d.argmax(1))
        oof = (oof + old_acc) / 2.0
        trial.set_user_attr("2019_acc:", old_acc)
        trial.set_user_attr("(2020+2019_acc) / 2:", oof)
        print(f"old_acc:", old_acc)
        print(f"(oof + old_acc) / 2.0:", oof)

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
