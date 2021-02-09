import os
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

preds = []
# ---- anonamename ----
# m_dir = "../input/cassava-efficientnetwithpytorchlightning"

# pkl = f"{m_dir}/kaggle_upload_tf_efficientnet_b4_ns_BiTemperedLoss/Y_pred.pkl"
pkl = r"C:\Users\81908\jupyter_notebook\pytorch_lightning_work\kaggle_Cassava\notebook\cassava-bitempered-logistic-loss\20201227\kaggle_upload_tf_efficientnet_b4_ns_BiTemperedLoss\Y_pred.pkl"
pred = pickle.load(open(pkl, "rb"))
preds.append(pred.values)

# pkl = f"{m_dir}/kaggle_upload_byol_seresnext50_32x4d_cutmix_labelsmooth_half/Y_pred.pkl"
pkl = r"C:\Users\81908\jupyter_notebook\pytorch_lightning_work\kaggle_Cassava\run_old_myPC\notebook\kaggle_upload_byol_seresnext50_32x4d_cutmix_labelsmooth_half\Y_pred.pkl"
pred = pickle.load(open(pkl, "rb"))
preds.append(pred.values)

# pkl = f"{m_dir}/kaggle_upload_resnest101e_cleanlab_noise_cutmix/Y_pred.pkl"
# pred = pickle.load(open(pkl, "rb"))
# preds.append(pred.values)

# pkl = f"{m_dir}/kaggle_upload_tf_efficientnet_b4_ns_cleanlab_noise_cutmix_fmix_n_over/Y_pred.pkl"
# pred = pickle.load(open(pkl, "rb"))
# preds.append(pred.values)

# pkl = f"{m_dir}/kaggle_upload_vit_b16_224_fold10/kaggle_upload_vit_b16_224_fold10/Y_pred.pkl"
pkl = r"C:\Users\81908\jupyter_notebook\pytorch_lightning_work\kaggle_Cassava\run_old_myPC\notebook\cassava-bitempered-logistic-loss\20210121\kaggle_upload_vit_base_patch16_224_fold3\vit_b16_224_fold10\kaggle_upload_vit_b16_224_fold10\Y_pred.pkl"
pred = pickle.load(open(pkl, "rb"))
preds.append(pred.values)


# ---- SiNpcw ----
m_dir = r"C:\Users\81908\jupyter_notebook\pytorch_lightning_work\kaggle_Cassava\kaggle_datasets_dl\cassavapkl"

pkl = f"{m_dir}/22019613_efficientnet-b4.pkl"
pred = pickle.load(open(pkl, "rb"))
preds.append(pred.values)

pkl = f"{m_dir}/22019632_timm-resnest101e.pkl"
pred = pickle.load(open(pkl, "rb"))
preds.append(pred.values)
