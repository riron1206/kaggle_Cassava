# https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/208239
import torch
import torch.nn as nn
import torch.nn.functional as F


def onehot(outputs, targets):
    """正解ラベルonehot化"""
    if len(targets.shape) < len(outputs.shape):  # not one-hot
        targets_onehot = torch.zeros_like(outputs)
        targets_onehot.scatter_(1, targets[..., None], 1)
    else:
        targets_onehot = targets
        targets = torch.max(targets_onehot, 1)[1]
    return targets, targets_onehot


def label_smoothing(smoothing, targets_onehot):
    """label_smoothing"""
    num_classes = targets_onehot.shape[-1]
    targets_onehot = (
        1 - smoothing * num_classes / (num_classes - 1)
    ) * targets_onehot + smoothing / (num_classes - 1)
    return targets_onehot


class SymmetricCrossEntropy(nn.Module):
    def __init__(self, num_classes=5, alpha=0.1, beta=1.0, smoothing=0.0):
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.smoothing = smoothing

    def forward(self, outputs, targets, reduction="mean"):
        targets, targets_onehot = onehot(outputs, targets)

        if self.label_smoothing > 0:
            targets_onehot = label_smoothing(self.label_smoothing, targets_onehot)

        ce_loss = F.cross_entropy(outputs, targets, reduction=reduction)
        rce_loss = (-targets_onehot * outputs.softmax(1).clamp(1e-7, 1.0).log()).sum(1)

        if reduction == "mean":
            rce_loss = rce_loss.mean()
        elif reduction == "sum":
            rce_loss = rce_loss.sum()

        return self.alpha * ce_loss + self.beta * rce_loss
