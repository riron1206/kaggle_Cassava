import os
import glob
import pickle
import json
import numpy as np
import pandas as pd

if os.getcwd() in "/kaggle/working":
    ROOT_DIR = "../input/cassava-leaf-disease-classification"
    TRAIN_DIR = "../input/cassava-leaf-disease-classification/train_images"
    TEST_DIR = "../input/cassava-leaf-disease-classification/test_images"
    MERGED_DIR = "../input/cassava-leaf-disease-merged"

elif os.getcwd() in "/content":
    ROOT_DIR = "/content/drive/MyDrive/competitions/cassava"
    TRAIN_DIR = "/content/drive/MyDrive/competitions/cassava/train_images"
    TEST_DIR = "/content/drive/MyDrive/competitions/cassava/test_images"
    MERGED_DIR = "../input/cassava-leaf-disease-merged"

else:
    ROOT_DIR = r"C:\Users\81908\jupyter_notebook\pytorch_lightning_work\kaggle_Cassava\input\cassava-leaf-disease-classification"
    TRAIN_DIR = r"C:\Users\81908\jupyter_notebook\pytorch_lightning_work\kaggle_Cassava\input\cassava-leaf-disease-classification\train_images"
    TEST_DIR = r"C:\Users\81908\jupyter_notebook\pytorch_lightning_work\kaggle_Cassava\input\cassava-leaf-disease-classification\test_images"
    MERGED_DIR = r"C:\Users\81908\jupyter_notebook\pytorch_lightning_work\kaggle_Cassava\input\Cassava_Leaf_Disease_Merged"

with open(f"{ROOT_DIR}/label_num_to_disease_map.json", "r") as f:
    name_mapping = json.load(f)
name_mapping = {int(k): v for k, v in name_mapping.items()}

df = pd.read_csv(f"{ROOT_DIR}/train.csv")
df["file_path"] = f"{TRAIN_DIR}/" + df["image_id"]
onehot_label = np.identity(len(name_mapping))[df["label"].values]
onehot_label = pd.DataFrame(onehot_label, columns=name_mapping.values())
df = df.join(onehot_label)
df["logit"] = 1.0
df["target"] = df["label"]

old_df = pd.read_csv(f"{MERGED_DIR}/merged.csv")
old_df = old_df[old_df["source"] == 2019].reset_index(drop=True)
old_df["file_path"] = f"{MERGED_DIR}/train/" + old_df["image_id"]

# 6model optuna best oof
# [
#    f"{ano_m_dir}/tf_efficientnet_b4_ns_tta.pkl",
#    f"{ano_m_dir}/resnest101e_cleanlab_noise_cutmix_tta.pkl",
#    f"{ano_m_dir}/vit_base_patch32_384.fit_tta.pkl",
#    f"{sin_m_dir}/22019632_timm-resnest101e.pkl",
#    f"{sin_m_dir}/ex07C_efficientnet-b4.pkl",
#    f"{sin_m_dir}/ex16C_timm-resnest101e.pkl",
# ]

preds = []
# ---- anonamename ----
"""
ttaの結果のpkl
"""
m_dir = r"C:\Users\81908\jupyter_notebook\pytorch_lightning_work\kaggle_Cassava\notebook\check_oof\cassava-emsemble-v2_tta_oof\kaggle_upload_oof_tta"

pkl = f"{m_dir}/tf_efficientnet_b4_ns_tta.pkl"
pred = pickle.load(open(pkl, "rb"))
preds.append(pred.values)

pkl = f"{m_dir}/resnest101e_cleanlab_noise_cutmix_tta.pkl"
pred = pickle.load(open(pkl, "rb"))
preds.append(pred.values)

pkl = f"{m_dir}/vit_base_patch32_384.fit_tta.pkl"
pred = pickle.load(open(pkl, "rb"))
preds.append(pred.values)

# # ---- SiNpcw ----
"""
ttaの結果のpkl
"""
m_dir = r"C:\Users\81908\jupyter_notebook\pytorch_lightning_work\kaggle_Cassava\kaggle_datasets_dl\cassavapkl"

pkl = f"{m_dir}/22019632_timm-resnest101e.pkl"
pred = pickle.load(open(pkl, "rb"))
preds.append(pred.values)

pkl = f"{m_dir}/ex07C_efficientnet-b4.pkl"
pred = pickle.load(open(pkl, "rb"))
preds.append(pred.values)

pkl = f"{m_dir}/ex16C_timm-resnest101e.pkl"
pred = pickle.load(open(pkl, "rb"))
preds.append(pred.values)
