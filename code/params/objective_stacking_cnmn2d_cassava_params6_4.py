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

# ['sub_tf_efficientnet_b4_ns', 'sub_resnest101e', 'sub_vit-base-patch16-224', 'sub_byol_seresnext50_32x4d', 'sub_microsoftvisionresnet50', 'ex02C_tta5_efficientnet-b4.pkl']
# [0.8924148245081086, 0.8941907744076273, 0.8792354068327336, 0.8941907744076273, 0.8866196195728373, 0.9008272187689863]
# best acc: 0.9055007711361406
# cfm_oof_tta = 0.9056
# [[0.17496477, 0.16712826, 0.16989827, 0.16699778, 0.16001712]
# , [0.17684359, 0.16617324, 0.16633336, 0.16680173, 0.1682363]
# , [0.14419915, 0.16636424, 0.15303017, 0.16654032, 0.16738014]
# , [0.16885862, 0.16645975, 0.17328928, 0.1667233, 0.16455479]
# , [0.15782057, 0.16263967, 0.16528997, 0.16600444, 0.17071918]
# , [0.17731329, 0.17123484, 0.17215894, 0.16693243, 0.16909247]]

preds = []
# ---- anonamename ----
"""
ttaの結果のpkl
"""
m_dir = r"C:\Users\81908\jupyter_notebook\pytorch_lightning_work\kaggle_Cassava\notebook\check_oof\cassava-emsemble-v2_tta_oof\kaggle_upload_oof_tta"

pkl = f"{m_dir}/tf_efficientnet_b4_ns_tta.pkl"
pred = pickle.load(open(pkl, "rb"))
preds.append(pred.values)

pkl = f"{m_dir}/resnest101e_tta.pkl"
pred = pickle.load(open(pkl, "rb"))
preds.append(pred.values)

pkl = f"{m_dir}/vit_base_patch16_224.fit_tta.pkl"
pred = pickle.load(open(pkl, "rb"))
preds.append(pred.values)

pkl = f"{m_dir}/byol_seresnext50_32x4d_cutmix_labelsmooth_half_tta.pkl"
pred = pickle.load(open(pkl, "rb"))
preds.append(pred.values)

pkl = f"{m_dir}/microsoftvisionresnet50_tta.pkl"
pred = pickle.load(open(pkl, "rb"))
preds.append(pred.values)

# # ---- SiNpcw ----
"""
ttaの結果のpkl
"""
m_dir = r"C:\Users\81908\jupyter_notebook\pytorch_lightning_work\kaggle_Cassava\kaggle_datasets_dl\cassavapkl"

pkl = f"{m_dir}/ex02C_tta5_efficientnet-b4.pkl"
pred = pickle.load(open(pkl, "rb"))
preds.append(pred.values)
