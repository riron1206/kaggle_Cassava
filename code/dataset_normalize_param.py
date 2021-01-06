import cv2
import pandas as pd

from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from albumentations.pytorch import ToTensorV2


def dataset_normalize_param(df, batch_size=256, num_workers=0):
    """
    dfは file_path 列が必要
    メモリ足りない場合は batch_size 下げること
    Usage:
        mean, std = dataset_normalize_param(df)
    """
    torch_dataset = TMPDataset(df)
    dataloader = torch.utils.data.DataLoader(
        torch_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    mean, std = get_mean_std(dataloader)
    print("mean, std:", mean, std)
    return mean, std


def get_mean_std(dataloader):
    """
    指定データセットについて画像チャネルの平均値と標準値を計算する

    データセット全体の標準偏差は、ミニバッチごとのstdを計算してから、ミニバッチのstdの平均として最終stdを計算する
    データセット全体に近づけるためバッチサイズをできるだけ大きくすること

    https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_std_mean.py
    """
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for batch in tqdm(dataloader):
        data = batch["x"]
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


# サンプルデータセット
class TMPDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, transforms=transforms.ToTensor(),
    ):
        self.df = df
        self.transforms = transforms

    def __getitem__(self, index):
        im_path = self.df.iloc[index]["file_path"]
        x = cv2.imread(im_path, cv2.IMREAD_COLOR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        # x = self.transforms(image=x)["image"] / 255.0  # albumentations の場合
        x = self.transforms(x)  # torchvosion の場合(albumentations の結果と同じになる)
        return {"x": x}

    def __len__(self):
        return len(self.df)
