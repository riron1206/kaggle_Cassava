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

# ['resnest101e_tta.pkl', 'tf_efficientnet_b4_ns_fold3_tta.pkl', 'vit_b16_224_fold10_tta.pkl', '22019720_tta3_efficientnet-b4.pkl', 'ex02C_efficientnet-b4.pkl']
# avg oof tta=0.90606 [0.8941907744076273, 0.8924615600317801, 0.8860587932887788, 0.8969949058279199, 0.8997990372482124]
# best acc: 0.9057811842781698
# best_ens_weights:
# , [[0.20922478, 0.19727891, 0.19879455, 0.19959025, 0.20360584]
# , [0.20283412, 0.20147392, 0.20721189, 0.19943385, 0.19096467]
# , [0.18588497, 0.19580499, 0.18819495, 0.20062245, 0.19935758]
# , [0.1961656, 0.20340136, 0.2013925, 0.20034094, 0.20153352]
# , [0.20589053, 0.20204082, 0.20440611, 0.20001251, 0.20453839]]

preds = []
# ---- anonamename ----
"""
ttaの結果のpkl
"""
m_dir = r"C:\Users\81908\jupyter_notebook\pytorch_lightning_work\kaggle_Cassava\notebook\check_oof\cassava-emsemble-v2_tta_oof\kaggle_upload_oof_tta"

pkl = f"{m_dir}/resnest101e_tta.pkl"
pred = pickle.load(open(pkl, "rb"))
preds.append(pred.values)

pkl = f"{m_dir}/tf_efficientnet_b4_ns_fold3_tta.pkl"
pred = pickle.load(open(pkl, "rb"))
preds.append(pred.values)

pkl = f"{m_dir}/vit_b16_224_fold10_tta.pkl"
pred = pickle.load(open(pkl, "rb"))
preds.append(pred.values)

# # ---- SiNpcw ----
"""
ttaの結果のpkl
"""
m_dir = r"C:\Users\81908\jupyter_notebook\pytorch_lightning_work\kaggle_Cassava\kaggle_datasets_dl\cassavapkl"

pkl = f"{m_dir}/22019720_tta3_efficientnet-b4.pkl"
pred = pickle.load(open(pkl, "rb"))
preds.append(pred.values)

pkl = f"{m_dir}/ex02C_efficientnet-b4.pkl"
pred = pickle.load(open(pkl, "rb"))
preds.append(pred.values)
