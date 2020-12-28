# https://www.kaggle.com/dunklerwald/pytorch-efficientnet-with-tta-training
# source: https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/173733
# https://www.kaggle.com/c/bengaliai-cv19/discussion/128115
import torch
import torch.nn as nn
import torch.nn.functional as F


def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
    assert 0 <= smoothing < 1
    with torch.no_grad():
        # label one-hot化してスムージング
        targets = (
            torch.empty(size=(targets.size(0), n_classes), device=targets.device)
            .fill_(smoothing / (n_classes - 1))
            .scatter_(1, targets.data.unsqueeze(1), 1.0 - smoothing)
        )
    return targets


class FocalCosineLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, xent=0.1, smoothing=0.0):
        super(FocalCosineLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.xent = xent
        self.y = torch.Tensor([1]).to(device)
        self.smoothing = smoothing

    def forward(self, inputs, targets, reduction="mean"):
        if self.smoothing > 0.0:
            # LabelSmoothing
            targets_onehot = _smooth_one_hot(
                targets, inputs.size(-1), smoothing=self.smoothing
            )
        else:
            # label one-hot化
            targets_onehot = F.one_hot(targets, num_classes=inputs.size(-1))

        cosine_loss = F.cosine_embedding_loss(
            inputs, targets_onehot, self.y, reduction=reduction,
        )

        cent_loss = F.cross_entropy(F.normalize(inputs), targets, reduce=False)
        pt = torch.exp(-cent_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * cent_loss

        if reduction == "mean":
            focal_loss = torch.mean(focal_loss)

        return cosine_loss + self.xent * focal_loss
