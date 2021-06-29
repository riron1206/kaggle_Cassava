import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset

""" データセット """
def getTrainLoader(conf, train_df, batch, augment_op=None, teacher_op=None, num_workers=4, drop_last=True, undersample=None) -> torch.utils.data.DataLoader:
    """
    学習データローダーを取得するためのユーティリティ.
    """
    stdout = conf['stdout'] if 'stdout' in conf and conf['stdout'] is not None else print
    imgdir = conf['image_dir']
    loader = torch.utils.data.DataLoader(
        TrainDataset(stdout, imgdir, train_df, augment_op=augment_op, teacher_op=teacher_op, undersample=undersample),
        batch_size=batch,
        shuffle=True,
        drop_last=drop_last,
        pin_memory=True,
        num_workers=num_workers,
    )
    return loader

def getValidLoader(conf, valid_df, batch=1, augment_op=None, num_workers=4) -> torch.utils.data.DataLoader:
    """
    検証データローダーを取得するためのユーティリティ.
    """
    stdout = conf['stdout'] if 'stdout' in conf and conf['stdout'] is not None else print
    imgdir = conf['image_dir']
    loader = torch.utils.data.DataLoader(
        TrainDataset(stdout, imgdir, valid_df, augment_op=augment_op),
        batch_size=batch,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers
    )
    return loader

""" データセットクラス """
class TrainDataset(Dataset):
    def __init__(self, stdout, image_dir, dataframe, augment_op=None, teacher_op=None, undersample=None) -> Dataset:
        self.stdout = stdout
        self.image_dir = image_dir
        self.dataframe = dataframe
        self.sample_df = dataframe
        self.augment_op = augment_op
        self.teacher_op = teacher_op
        self.undersample = undersample
        self.epoch_count = 0
        self.getitem_fn = self.__getitem_basic__ if self.teacher_op is None else self.__getitem_teach__
        self.shuffle()

    def __len__(self):
        return len(self.sample_df)

    def __getitem__(self, idx):
        return self.getitem_fn(idx)

    def __getitem_basic__(self, idx):
        # columns:
        colimg = 0
        collbl = 1
        # process:
        imf = os.path.join(self.image_dir, self.sample_df.iat[idx, colimg])
        if not os.path.exists(imf):
            self.stdout(f'(FileNotFound) {imf}')
        img = cv2.imread(imf)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lbl = self.sample_df.iat[idx, collbl]
        ret = self._inner_augment_(self.augment_op, img.copy())
        img = ret['image'].astype(np.float32)
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img), lbl

    def __getitem_teach__(self, idx):
        # columns:
        colimg = 0
        collbl = 1
        # process:
        imf = os.path.join(self.image_dir, self.sample_df.iat[idx, colimg])
        if not os.path.exists(imf):
            self.stdout(f'(FileNotFound) {imf}')
        img = cv2.imread(imf)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lbl = self.sample_df.iat[idx, collbl]
        ret1 = self._inner_augment_(self.augment_op, img.copy())
        img1 = ret1['image'].astype(np.float32)
        img1 = img1.transpose(2, 0, 1)
        ret2 = self._inner_augment_(self.teacher_op, img.copy())
        img2 = ret2['image'].astype(np.float32)
        img2 = img2.transpose(2, 0, 1)
        return torch.from_numpy(img1), torch.from_numpy(img2), lbl

    def _inner_augment_(self, ops, img):
        if ops is not None:
            return ops(force_apply=False, image=img)
        else:
            return { 'image' : img }

    def shuffle(self):
        if self.undersample is None or self.undersample <= 0:
            self.sample_df = self.dataframe
        else:
            self.sample_df = pd.concat([
                self.dataframe[self.dataframe['label'] != 3],
                self.dataframe[self.dataframe['label'] == 3].sample(n=self.undersample, random_state=self.epoch_count)
            ])
        self.epoch_count += 1
