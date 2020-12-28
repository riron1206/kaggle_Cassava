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
    https://discuss.pytorch.org/t/how-to-freeze-bn-layers-while-training-the-rest-of-network-mean-and-var-wont-freeze/89736"""
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    return net
