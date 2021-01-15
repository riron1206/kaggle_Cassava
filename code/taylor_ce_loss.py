# https://www.kaggle.com/yerramvarun/cassava-taylorce-loss-label-smoothing-combo?scriptVersionId=51916750
# implementations reference - https://github.com/CoinCheung/pytorch-loss/blob/master/pytorch_loss/taylor_softmax.py
# paper - https://www.ijcai.org/Proceedings/2020/0305.pdf
"""
ラベルノイズにロバストなTaylor Cross Entropy (TCE) loss

TCEはテイラー級数展開でCCEの効果を緩めたCategorical Cross Entropy (CCE)
MAE はCCEの1次テイラー級数近似（TCEのt=1）
MAE と平均2乗誤差(MSE)の平均がCCEの2次テイラー級数近似（TCEのt=2）
次数大きなTCEはCCEに近づく

バイテンパーロス（外れ値のロスを減らす+softmaxの裾の幅広げてラベルノイズの効果弱めるloss）
とは精度比較してないのは作為的なものを感じる
パラメータは次数tだけなのはいいけど
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TaylorSoftmax(nn.Module):
    def __init__(self, dim=1, n=2):
        super(TaylorSoftmax, self).__init__()
        assert n % 2 == 0
        self.dim = dim
        self.n = n

    def forward(self, x):

        fn = torch.ones_like(x)
        denor = 1.0
        for i in range(1, self.n + 1):
            denor *= i
            fn = fn + x.pow(i) / denor
        out = fn / fn.sum(dim=self.dim, keepdims=True)
        return out


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        """Taylor Softmax and log are already applied on the logits"""
        # pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class TaylorCrossEntropyLoss(nn.Module):
    def __init__(
        self, num_classes, n=2, ignore_index=-1, reduction="mean", smoothing=0.2
    ):
        """
        Usage:
            optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)
            criterion = TaylorCrossEntropyLoss(n=2, smoothing=0.2)  # nがテイラー級数展開の次数
        """
        super(TaylorCrossEntropyLoss, self).__init__()
        assert n % 2 == 0
        self.taylor_softmax = TaylorSoftmax(dim=1, n=n)
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.lab_smooth = LabelSmoothingLoss(num_classes, smoothing=smoothing)

    def forward(self, logits, labels):

        log_probs = self.taylor_softmax(logits).log()
        # loss = F.nll_loss(log_probs, labels, reduction=self.reduction,
        #        ignore_index=self.ignore_index)
        loss = self.lab_smooth(log_probs, labels)
        return loss
