import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch
import torch.nn as nn


def freeze_until(net, n_freeze_until=0):
    """n_freeze_until 層までの重み固定
    http://maruo51.com/2020/03/17/torchvision_models_tips/"""
    for i, param in enumerate(net.parameters()):

        if n_freeze_until == -1:
            # -1なら全て重み固定する
            param.requires_grad = False

        elif (n_freeze_until > 0) and (i <= n_freeze_until):
            param.requires_grad = False

    return net


def freeze_bn(net):
    """バッチノーマライゼーションの重み凍結
    https://discuss.pytorch.org/t/how-to-freeze-bn-layers-while-training-the-rest-of-network-mean-and-var-wont-freeze/89736
    https://www.kaggle.com/yingpengchen/pytorch-cldc-train-with-vit
    """
    try:
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                m.eval()
    except ValuError:
        print("error with batchnorm2d or layernorm")
        return


def torchvision_joint(images, nrow=2, padding=0, is_show=False):
    """画像をただ並べる
    https://blog.shikoan.com/torchvision-image-tile/
    Usage:
        # 2x2のカラー画像を作る
        def create_color_images(n=4):
            images = np.zeros((n, 64, 64, 3), np.uint8)  # NHWCのフォーマット
            # 0-2 = Red, Green, Blue
            images[0, :, :, 0] = 255
            images[1, :, :, 1] = 255
            images[2, :, :, 2] = 255
            # 3-5 = Yellow, Aqua, Fuchsia
            images[3, :, :, [0, 1]] = 255
            return images
        images = create_color_images(4)
        joined_images = torchvision_joint(images, nrow=2, padding=0, is_show=True)
    """
    images = np.transpose(images, [0, 3, 1, 2])  # PyTorch用に, NHWC -> NCHW に変換
    # PyTorchのテンソルにする（Numpy配列から作る場合はtorch.Tensorよりas_tensorのほうが良い）
    images_tensor = torch.as_tensor(images)
    # 一つの画像に統合
    # paddingの値→隙間, nrow=行あたりの画像数（端数は埋められる）
    joined_images_tensor = torchvision.utils.make_grid(
        images_tensor, nrow=nrow, padding=padding
    )
    joined_images = joined_images_tensor.numpy()  # PyTorchのテンソル→Numpy配列
    joined_images = np.transpose(joined_images, [1, 2, 0])  # NCHW -> NHWCに変換
    if is_show:
        plt.imshow(joined_images)
        plt.show()
    return joined_images
