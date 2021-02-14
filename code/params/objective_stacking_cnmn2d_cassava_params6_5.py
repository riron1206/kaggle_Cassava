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

# ['sub_tf_efficientnet_b4_ns', 'sub_resnest101e_cleanlab_noise_cutmix', '22019632_timm-resnest101e.pkl', 'ex02C_efficientnet-b4.pkl', 'vit_base_patch16_384_noTTA.pkl']
# [0.8904051969902322, 0.8943309809786418, 0.897275318969949, 0.8997990372482124, 0.8869935037622096]
# best acc: 0.9057811842781698
# best_ens_weights:
# [[0.21317501, 0.19585202, 0.20264682, 0.19972413, 0.1907759]
# , [0.19790783, 0.19932735, 0.20212986, 0.19942631, 0.2042922]
# , [0.1931015, 0.20426009, 0.20119934, 0.20044516, 0.20222864]
# , [0.20949958, 0.19977578, 0.20337055, 0.20046083, 0.20367313]
# , [0.18631609, 0.20078475, 0.19065343, 0.19994357, 0.19903013]]

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

# # ---- SiNpcw ----
"""
ttaの結果のpkl
"""
m_dir = r"C:\Users\81908\jupyter_notebook\pytorch_lightning_work\kaggle_Cassava\kaggle_datasets_dl\cassavapkl"

pkl = f"{m_dir}/22019632_tta3_timm-resnest101e.pkl"
pred = pickle.load(open(pkl, "rb"))
preds.append(pred.values)

pkl = f"{m_dir}/ex02C_tta5_efficientnet-b4.pkl"
pred = pickle.load(open(pkl, "rb"))
preds.append(pred.values)

# # ---- Fukuda ----
m_dir = r"C:\Users\81908\jupyter_notebook\pytorch_lightning_work\kaggle_Cassava\kaggle_datasets_dl\confidence20210207"

pkl = f"{m_dir}/vit_base_patch16_384_TTA.pkl"
pred = pickle.load(open(pkl, "rb"))
preds.append(pred.values)
