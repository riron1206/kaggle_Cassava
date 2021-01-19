import os
import glob
import pickle
import pathlib
import imagehash

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import torch


def dup_img_hash(df, sim_th=0.9):
    """画像のハッシュ値求めて類似画像を検索
    https://www.kaggle.com/anonamename/duplicate-images-in-two-competitions

    Usage:
        duplicate_image_ids, old_compe_df_dup_del = dup_img_hash(old_compe_df)
        print(old_compe_df.shape, old_compe_df_dup_del.shape)

        # 重複画像のファイル名など保存
        pickle.dump(duplicate_image_ids, open("old_compe_train_test_pred_dup_del.pkl", 'wb'))
        old_compe_df_dup_del.to_csv("old_compe_train_test_pred_dup_del.csv", index=False)

        # 重複画像確認
        dup_show(duplicate_image_ids, show_count=5)
    """
    img_paths = df["file_path"]

    funcs = [
        imagehash.average_hash,
        imagehash.phash,
        imagehash.dhash,
        imagehash.whash,
    ]
    image_ids = []
    hashes = []

    # 画像のハッシュ計算
    for path in tqdm(img_paths, desc="calc img hash"):
        image = Image.open(path)
        image_id = path  # os.path.basename(path)
        image_ids.append(image_id)
        hashes.append(np.array([f(image).hash for f in funcs]).reshape(256))

    hashes_all = np.array(hashes)
    hashes_all = torch.Tensor(hashes_all.astype(int))

    # ハッシュから類似度計算
    sims = np.array(
        [
            (hashes_all[i] == hashes_all).sum(dim=1).numpy() / 256
            for i in tqdm(range(hashes_all.shape[0]), desc="calc similarity")
        ]
    )

    # 閾値以上の類似画像検索
    indices1 = np.where(sims > sim_th)
    indices2 = np.where(indices1[0] != indices1[1])
    image_ids1 = [image_ids[i] for i in indices1[0][indices2]]
    image_ids2 = [image_ids[i] for i in indices1[1][indices2]]
    dups = {
        tuple(sorted([image_id1, image_id2])): True
        for image_id1, image_id2 in zip(image_ids1, image_ids2)
    }
    print("found %d duplicates" % len(dups))

    duplicate_image_ids = sorted(list(dups))

    # 重複レコード削除（ファイル名同じものを持つ可能性あるのでlen(dup_paths)とdf_dup_del.shapeが一致しないケースあり）
    dup_paths = [p[1] for p in duplicate_image_ids]
    df_dup_del = df[~df["file_path"].isin(dup_paths)]
    print(len(dup_paths))

    return duplicate_image_ids, df_dup_del


def dup_show(duplicate_image_ids, show_count=10):
    """類似画像並べて表示"""
    fig, axs = plt.subplots(show_count, 2, figsize=(15, 6 * show_count))
    for row in range(show_count):
        for col in range(2):
            img_path = duplicate_image_ids[row][col]
            img = Image.open(img_path)
            axs[row, col].imshow(img)
            axs[row, col].set_title("image_id : " + pathlib.Path(img_path).name)
            axs[row, col].axis("off")
