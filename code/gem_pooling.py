import torch
import torch.nn as nn

# https://amaarora.github.io/2020/08/30/gempool.html
# https://github.com/amaarora/amaarora.github.io/blob/master/nbs/GeM%20Pooling.ipynb
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class GeMNet(nn.Module):
    def __init__(self, features, pool, in_features, n_classes):
        super(GeMNet, self).__init__()
        self.features = nn.Sequential(*features)
        self.fc = nn.Linear(in_features, n_classes)
        self.pool = pool

    def forward(self, x):
        o = self.features(x)
        o = self.pool(o).squeeze(-1).squeeze(-1)
        o = self.fc(o)
        return o
