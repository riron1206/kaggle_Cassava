# https://tawara.hatenablog.com/entry/2020/12/16/132415
# https://www.kaggle.com/c/lish-moa/discussion/204685
# https://www.kaggle.com/anonamename/stacking-test

import re
import typing as tp

import torch
from torch import nn


def get_activation(activ_name: str = "relu"):
    """"""
    act_dict = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "identity": nn.Identity(),
    }
    if activ_name in act_dict:
        return act_dict[activ_name]
    elif re.match(r"^htanh\_\d{4}$", activ_name):
        bound = int(activ_name[-4:]) / 1000
        return nn.Hardtanh(-bound, bound)
    else:
        raise NotImplementedError


class LBAD(nn.Module):
    """Linear (-> BN) -> Activation (-> Dropout)"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        drop_rate: float = 0.0,
        use_bn: bool = False,
        use_wn: bool = False,
        activ: str = "relu",
    ):
        """"""
        super(LBAD, self).__init__()
        layers = [nn.Linear(in_features, out_features)]
        if use_wn:
            layers[0] = nn.utils.weight_norm(layers[0])

        if use_bn:
            layers.append(nn.BatchNorm1d(out_features))

        layers.append(get_activation(activ))

        if drop_rate > 0:
            layers.append(nn.Dropout(drop_rate))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        return self.layers(x)


class MLP(nn.Module):
    """Stacked Dense layers
    Usage:
        mlp_params = dict(
            n_features_list=[n_classes*n_models, 8, n_classes],
            use_tail_as_out=True,
            drop_rate=0.2,
            use_bn=False,
            use_wn=True,
        )
        mlp = MLP(**mlp_params)
        print(mlp)

        # 予測
        x = mlp_pred.values[0]
        x = torch.from_numpy(x.astype(np.float32)).clone()
        mlp(x)
    """

    def __init__(
        self,
        n_features_list: tp.List[int],
        use_tail_as_out: bool = False,
        drop_rate: float = 0.0,
        use_bn: bool = False,
        use_wn: bool = False,
        activ: str = "relu",
    ):
        """"""
        super(MLP, self).__init__()
        n_layers = len(n_features_list) - 1
        layers = []
        for i in range(n_layers):
            in_feats, out_feats = n_features_list[i : i + 2]
            if i == n_layers - 1 and use_tail_as_out:
                layer = nn.Linear(in_feats, out_feats)
                if use_wn:
                    layer = nn.utils.weight_norm(layer)
            else:
                layer = LBAD(in_feats, out_feats, drop_rate, use_bn, use_wn, activ)
            layers.append(layer)

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        return self.layers(x)


class CNNStacking1d(nn.Module):
    """1D-CNN for Stacking.
    Usage:
        kwargs_head = dict(
            n_features_list=[64, n_classes],
            use_tail_as_out=True,
            drop_rate=0.8,
            use_bn=False,
            use_wn=True,
        )
        cnn1d_params = dict(
            n_models=n_models,
            n_channels_list=[n_classes, 64],
            use_bias=True,
            kwargs_head=kwargs_head,
        )
        cnmn1d = CNNStacking1d(**cnn1d_params)
        print(cnmn1d)

        # 予測
        cnn_pred = np.stack([pred1, pred2, pred3, pred4]).transpose(1,2,0)
        print(cnn_pred.shape)  # shape: (n_sample, n_classes, n_models)

        bs = 5
        x = cnn_pred[:bs] # shape: (bs, n_classes, n_models)
        print(x)
        x = torch.from_numpy(x.astype(np.float32)).clone()
        cnmn1d(x)
    """

    def __init__(
        self,
        n_models: int,
        n_channels_list: tp.List[int],
        use_bias: bool = False,
        kwargs_head: tp.Dict = {},
    ):
        """"""
        super(CNNStacking1d, self).__init__()
        self.n_conv_layers = len(n_channels_list) - 1
        for i in range(self.n_conv_layers):
            in_ch = n_channels_list[i]
            out_ch = n_channels_list[i + 1]
            layer = nn.Sequential(
                nn.Conv1d(
                    in_ch, out_ch, kernel_size=3, stride=1, padding=0, bias=use_bias
                ),
                # nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            )
            setattr(self, "conv{}".format(i + 1), layer)

        kwargs_head["n_features_list"][0] = (
            n_models - 2 * self.n_conv_layers
        ) * n_channels_list[-1]
        self.head = MLP(**kwargs_head)

    def forward(self, x: torch.FloatTensor) -> torch.Tensor:
        """"""
        bs = x.shape[0]
        h = x  # shape: (bs, n_classes, n_models)
        for i in range(self.n_conv_layers):
            h = getattr(self, "conv{}".format(i + 1))(h)

        h = torch.reshape(h, (bs, -1))
        h = self.head(h)
        return h


class CNNStacking2d(nn.Module):
    """2D-CNN for Stacking.
    Usage:
        kwargs_head = dict(
            n_features_list=[-1, n_classes],
            use_tail_as_out=True,
            drop_rate=0.8,
            use_bn=False,
            use_wn=True,
        )
        cnn2d_params = dict(
            n_models=n_models,
            n_classes=n_classes,
            n_channels_list=[1, 8],
            use_bias=True,
            kwargs_head=kwargs_head,
        )
        cnn2d = CNNStacking2d(**cnn2d_params)
        print(cnn2d)

        # 予測
        cnn_pred = np.stack([pred1, pred2, pred3, pred4]).transpose(1,2,0)
        print(cnn_pred.shape)  # shape: (n_sample, n_classes, n_models)

        bs = 5
        x = cnn_pred[:bs]
        x = x.reshape(bs,1,n_classes, n_models)# shape: (bs, 1, n_classes, n_models)
        print(x)
        x = torch.from_numpy(x.astype(np.float32)).clone()
        cnn2d(x)
    """

    def __init__(
        self,
        n_models: int,
        n_classes: int,
        n_channels_list: tp.List[int],
        use_bias: bool = False,
        kwargs_head: tp.Dict = {},
    ):
        """"""
        super(CNNStacking2d, self).__init__()
        self.n_conv_layers = len(n_channels_list) - 1
        for i in range(self.n_conv_layers):
            in_ch = n_channels_list[i]
            out_ch = n_channels_list[i + 1]
            layer = nn.Sequential(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=(1, 3),
                    stride=1,
                    padding=0,
                    bias=use_bias,
                ),
                # nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
            setattr(self, "conv{}".format(i + 1), layer)

        kwargs_head["n_features_list"][0] = (
            (n_models - 2 * self.n_conv_layers) * n_classes * n_channels_list[-1]
        )
        self.head = MLP(**kwargs_head)

    def forward(self, x: torch.FloatTensor) -> torch.Tensor:
        """"""
        bs = x.shape[0]
        h = x  # shape: (bs, 1, n_classes, n_models)
        for i in range(self.n_conv_layers):
            h = getattr(self, "conv{}".format(i + 1))(h)

        h = torch.reshape(h, (bs, -1))
        h = self.head(h)
        return h


class GCNStacking(nn.Module):
    """GCN for Stacking."""

    def __init__(
        self,
        n_classes: int,
        n_channels_list: tp.List[int],
        add_self_loop: bool = False,
        kwargs_head: tp.Dict = {},
    ):
        """"""
        super(GCNStacking, self).__init__()
        self.n_conv_layers = len(n_channels_list) - 1
        for i in range(self.n_conv_layers):
            in_ch = n_channels_list[i]
            out_ch = n_channels_list[i + 1]
            # layer = CustomGraphConv(in_ch, out_ch, n_classes)
            layer = GraphConv(
                in_ch,
                out_ch,
                n_classes,
                shrare_msg=False,
                share_model_self=False,
                share_bias=False,
            )
            setattr(self, "conv{}".format(i + 1), layer)

        self.relu = nn.ReLU(inplace=True)
        if add_self_loop:
            adj_mat = torch.ones(n_classes, n_classes) / n_classes
        else:
            adj_mat = (1 - torch.eye(n_classes, n_classes)) / (n_classes - 1)
        self.register_buffer("A", adj_mat.float())

        kwargs_head["n_features_list"][0] = n_classes * n_channels_list[-1]
        self.head = MLP(**kwargs_head)

    def forward(self, X: torch.FloatTensor) -> torch.Tensor:
        """"""
        bs, n_classes = X.shape[:2]
        H = X  # shape: (bs, n_classes, n_models)
        for i in range(self.n_conv_layers):
            H = getattr(self, "conv{}".format(i + 1))(H, self.A[None, ...])
            H = self.relu(H)

        h = torch.reshape(H, (bs, -1))
        h = self.head(h)
        return h
