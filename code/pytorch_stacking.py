# https://tawara.hatenablog.com/entry/2020/12/16/132415
# https://www.kaggle.com/c/lish-moa/discussion/204685
# https://www.kaggle.com/anonamename/stacking-test
import os
import gc
import sys
import random
import shutil
import warnings
import typing as tp
from pathlib import Path
from copy import deepcopy

import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

import torch
from torch import nn
from torch.utils import data


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
        import glob
        import pathlib

        n_classes = 3

        m_dir = r"C:/Users/81908/jupyter_notebook/pytorch_lightning_work/kaggle_Cassava/notebook/check_oof/cassava-emsemble-v2_tta_oof/kaggle_upload_oof_tta"
        preds = []
        for pkl_path in sorted(glob.glob(f"{m_dir}/*.pkl")):
            preds.append(pickle.load(open(f"{pkl_path}", "rb")))  # データフレームでないとだめ
        mlp_n_classes = preds[0].shape[1]
        n_models = len(preds)
        print("n_models:", n_models)  # n_models: 14

        mlp_params = dict(
            n_features_list=[mlp_n_classes * n_models, 8, n_classes],  # [mlp_n_classes * n_models, n_classes] だと出力層1層だけになり、drop_rate指定してもdropoutが使われない
            use_tail_as_out=True,
            drop_rate=0.2,
            use_bn=False,
            use_wn=True,
        )
        mlp = MLP(**mlp_params)
        print(mlp)

        # 予測
        mlp_pred = pd.concat(preds, axis=1)
        print("mlp_pred.shape:", mlp_pred.shape)  # mlp_pred.shape: (21397, 70)

        bs = 5
        x = mlp_pred.values[:bs]
        print("x.shape:", x.shape)  # x.shape: (5, 70)
        x = torch.from_numpy(x.astype(np.float32)).clone()
        print("x.shape:", x.shape)  # x.shape: torch.Size([5, 70])
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
        import glob
        import pathlib

        n_classes = 3

        m_dir = r"C:/Users/81908/jupyter_notebook/pytorch_lightning_work/kaggle_Cassava/notebook/check_oof/cassava-emsemble-v2_tta_oof/kaggle_upload_oof_tta"
        preds = []
        for pkl_path in sorted(glob.glob(f"{m_dir}/*.pkl")):
            preds.append(pickle.load(open(f"{pkl_path}", "rb")).values)
        preds = np.array(preds)
        cnn1d_n_classes = preds.shape[2]
        print("cnn1d_n_classes:", cnn1d_n_classes)  # cnn1d_n_classes: 5
        n_models = len(preds)
        print("preds.shape:", preds.shape)  # preds.shape: (14, 21397, 5)
        print("n_models:", n_models)  # n_models: 14

        kwargs_head = dict(
            n_features_list=[-1,　15,　n_classes,],  # [-1, n_classes] だと出力層1層だけになり、drop_rate指定してもdropoutが使われない
            use_tail_as_out=True,
            drop_rate=0.8,
            use_bn=False,
            use_wn=True,
        )
        cnn1d_params = dict(
            n_models=n_models,
            n_channels_list=[cnn1d_n_classes, 64],
            use_bias=True,
            kwargs_head=kwargs_head,
        )
        cnmn1d = CNNStacking1d(**cnn1d_params)
        print(cnmn1d)

        # 予測
        cnn_pred = np.stack(preds).transpose(1, 2, 0)
        print("cnn_pred.shape:", cnn_pred.shape)  # cnn_pred.shape: (21397, 5, 14)

        bs = 5
        x = cnn_pred[:bs]  # shape: (bs, n_classes, n_models)
        print("x.shape:", x.shape)  # x.shape: (5, 5, 14)
        # print(x)
        x = torch.from_numpy(x.astype(np.float32)).clone()
        print("x.shape:", x.shape)  # x.shape: torch.Size([5, 5, 14])
        cnmn1d(x)
        print()
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
        import glob
        import pathlib

        n_classes = 3

        m_dir = r"C:/Users/81908/jupyter_notebook/pytorch_lightning_work/kaggle_Cassava/notebook/check_oof/cassava-emsemble-v2_tta_oof/kaggle_upload_oof_tta"
        preds = []
        for pkl_path in sorted(glob.glob(f"{m_dir}/*.pkl")):
            preds.append(pickle.load(open(f"{pkl_path}", "rb")).values)
        preds = np.array(preds)
        cnn2d_n_classes = preds.shape[2]
        print("cnn2d_n_classes:", cnn2d_n_classes)  # cnn2d_n_classes: 5
        n_models = len(preds)
        print("preds.shape:", preds.shape)  # preds.shape: (14, 21397, 5)
        print("n_models:", n_models)  # n_models: 14

        kwargs_head = dict(
            n_features_list=[-1, 15, n_classes],  # [-1, n_classes] だと出力層1層だけになり、drop_rate指定してもdropoutが使われない
            use_tail_as_out=True,
            drop_rate=0.8,
            use_bn=False,
            use_wn=True,
        )
        cnn2d_params = dict(
            n_models=n_models,
            n_classes=cnn2d_n_classes,
            n_channels_list=[1, 8],
            use_bias=True,
            kwargs_head=kwargs_head,
        )
        cnn2d = CNNStacking2d(**cnn2d_params)
        print(cnn2d)

        # 予測
        cnn_pred = np.stack(preds).transpose(1, 2, 0)
        print("cnn_pred.shape:", cnn_pred.shape)  # shape: (n_sample, n_classes, n_models)  # cnn_pred.shape: (21397, 5, 14)

        bs = 15
        x = cnn_pred[:bs]
        print("x.shape:", x.shape)  # x.shape: (15, 5, 14)
        x = x.reshape(bs, 1, cnn_pred.shape[1], n_models)  # shape: (bs, 1, n_classes, n_models)
        print("x.shape:", x.shape)  # x.shape: (15, 1, 5, 14)
        # print(x)
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


################################# for GCNs #################################
def vector_wise_matmul(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """
    See input matrixes X as bags of vectors, and multiply corresponding weight matrices by vector.

    Args:
        X: Input Tensor, shape: (batch_size, **n_vectors**, in_features)
        W: Weight Tensor, shape: (**n_vectors**, out_features, in_features)
    """
    X = torch.transpose(X, 0, 1)  # shape: (n_vectors, batch_size, in_features)
    W = torch.transpose(W, 1, 2)  # shape: (n_vectors, in_features, out_features)
    H = torch.matmul(X, W)  # shape: (n_vectors, batch_size, out_features)
    H = torch.transpose(H, 0, 1)  # shape: (batch_size, n_vectors, out_features)

    return H


def vector_wise_shared_matmul(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """
    See input matrixes X as bags of vectors, and multiply **shared** weight matrices.

    Args:
        X: Input Tensor, shape: (batch_size, **n_vectors**, in_features)
        W: Weight Tensor, shape: (out_features, in_features)
    """
    # W = torch.transpose(W, 0, 1)  # shape: (in_features, out_features)
    # H = torch.matmul(X, W)        # shape: (batch_size, n_vectors, out_features)

    H = nn.functional.linear(X, W)  # shape: (batch_size, n_vectors, out_features)

    return H


def _calculate_fan_in_and_fan_out_for_vwl(tensor) -> tp.Tuple[int]:
    """
    Input tensor: (n_vectors, out_features, in_features) or (out_features, in_features)
    """
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    fan_in = tensor.size(-1)
    fan_out = tensor.size(-2)

    return fan_in, fan_out


def _calculate_correct_fan_for_vwl(tensor, mode) -> int:
    """"""
    mode = mode.lower()
    valid_modes = ["fan_in", "fan_out"]
    if mode not in valid_modes:
        raise ValueError(
            "Mode {} not supported, please use one of {}".format(mode, valid_modes)
        )

    fan_in, fan_out = _calculate_fan_in_and_fan_out_for_vwl(tensor)
    return fan_in if mode == "fan_in" else fan_out


def kaiming_uniform_for_vwl(tensor, a=0, mode="fan_in", nonlinearity="leaky_relu"):
    """"""
    fan = _calculate_correct_fan_for_vwl(tensor, mode)
    gain = nn.init.calculate_gain(nonlinearity, a)
    std = gain / np.sqrt(fan)
    bound = np.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


class VectorWiseLinear(nn.Module):
    """
    For mini batch which have several matrices,
    see as these matrixes as bags of vectors, and multiply weight matrices by vector.

    input    X: (batch_size, **n_vectors**, in_features)
    weight W: (**n_vector**, out_features, in_features)
    output  Y: (batch_size, **n_vectors**, out_features)

    **Note**: For simplicity, bias is not described.

    X and W are can be seen as below.
    X: [
            [vec_{ 1, 1}, vec_{ 1, 2}, ... vec_{ 1, n_vectors}],
            [vec_{ 2, 1}, vec_{ 2, 2}, ... vec_{ 2, n_vectors}],
                                            .
                                            .
            [vec_{bs, 1}, vec_{bs, 2}, ... vec_{bs, n_vectors}]
        ]
    W: [
            Mat_{1}, Mat_{2}, ... , Mat_{n_vectors}
        ]
    Then Y is calclauted as:
    Y: [
        [ Mat_{1} vec_{ 1, 1}, Mat_{2} vec_{ 1, 2}, ... Mat_{n_vectors} vec_{ 1, n_vectors}],
        [ Mat_{1} vec_{ 2, 1}, Mat_{2} vec_{ 2, 2}, ... Mat_{n_vectors} vec_{ 2, n_vectors}],
        .
        .
        [ Mat_{1} vec_{bs, 1}, Mat_{2} vec_{bs, 2}, ... Mat_{n_vectors} vec_{bs, n_vectors}],
    ]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_vectors: int,
        bias: bool = True,
        weight_shared: bool = True,
    ) -> None:
        """Initialize."""
        super(VectorWiseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_vectors = n_vectors
        self.weight_shared = weight_shared

        if self.weight_shared:
            self.weight = nn.Parameter(
                torch.Tensor(self.out_features, self.in_features)
            )
            self.matmul_func = vector_wise_shared_matmul
        else:
            self.weight = nn.Parameter(
                torch.Tensor(self.n_vectors, self.out_features, self.in_features)
            )
            self.matmul_func = vector_wise_matmul

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weight and bias."""
        kaiming_uniform_for_vwl(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out_for_vwl(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward."""
        H = self.matmul_func(X, self.weight)
        if self.bias is not None:
            H = H + self.bias

        return H


class GraphConv(nn.Module):
    """Basic Graph Convolution Layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_nodes: int,
        shrare_msg: bool = True,
        model_self: bool = True,
        share_model_self: bool = True,
        bias: bool = True,
        share_bias: bool = True,
    ) -> None:
        """Intialize."""
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_nodes = n_nodes
        self.model_self = model_self
        super(GraphConv, self).__init__()

        # # message
        self.msg = VectorWiseLinear(
            in_channels, out_channels, n_nodes, False, shrare_msg
        )

        # # self-modeling
        if model_self:
            self.model_self = VectorWiseLinear(
                in_channels, out_channels, n_nodes, False, share_model_self
            )

        # # bias
        if bias:
            if share_bias:
                self.bias = nn.Parameter(torch.Tensor(out_channels))
            else:
                self.bias = nn.Parameter(torch.Tensor(n_nodes, out_channels))
            bound = 1 / np.sqrt(out_channels)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    def forward(
        self, X: torch.Tensor, A: torch.Tensor, W: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward.

        Args:
            X: (batch_size, n_nodes, n_channels)
                Array which represents bags of vectors.
                X[:, i, :] are corresponded to feature vectors of node i.
            A: (batch_size, n_nodes, n_nodes)
                Array which represents adjacency matrices.
                A[:, i, j] are corresponded to weights (scalar) of edges from node j to node i.
            W: (batch_size, n_nodes, n_nodes)
                Array which represents weight matrices between nodes.
        """
        if W is not None:
            A = A * W  # shape: (batch_size, n_nodes, n_nodes)

        # # update message
        M = X  #  shape: (batch_size, n_nodes, in_channels)
        # # # send message
        M = self.msg(M)  # shape: (batch_size, n_nodes, out_channels)
        # # # aggregate
        M = torch.matmul(A, M)  # shape: (batch_size, n_nodes, out_channels)

        # # update node
        # # # self-modeling
        H = M
        if self.model_self:
            H = H + self.model_self(X)
        if self.bias is not None:
            H = H + self.bias

        return H


######################################################################################

################################# utils for training #################################
class EvalFuncManager(nn.Module):
    """Manager Class for evaluation at the end of epoch"""

    def __init__(
        self,
        iters_per_epoch: int,
        evalfunc_dict: tp.Dict[str, nn.Module],
        prefix: str = "val",
    ) -> None:
        """Initialize"""
        super(EvalFuncManager, self).__init__()
        self.tmp_iter = 0
        self.iters_per_epoch = iters_per_epoch
        self.prefix = prefix
        self.metric_names = []
        for k, v in evalfunc_dict.items():
            setattr(self, k, v)
            self.metric_names.append(k)
        self.reset()

    def reset(self) -> None:
        """Reset State."""
        self.tmp_iter = 0
        for name in self.metric_names:
            getattr(self, name).reset()

    def __call__(self, y: torch.Tensor, t: torch.Tensor) -> None:
        """Forward."""
        for name in self.metric_names:
            getattr(self, name).update(y, t)
        self.tmp_iter += 1

        if self.tmp_iter == self.iters_per_epoch:
            # ppe.reporting.report(
            #    {
            #        "{}/{}".format(self.prefix, name): getattr(self, name).compute()
            #        for name in self.metric_names
            #    }
            # )
            self.reset()


class MeanLoss(nn.Module):
    def __init__(self):
        super(MeanLoss, self).__init__()
        self.loss_sum = 0
        self.n_examples = 0

    def forward(self, y: torch.Tensor, t: torch.Tensor):
        """Compute metric at once"""
        return self.loss_func(y, t)

    def reset(self):
        """Reset state"""
        self.loss_sum = 0
        self.n_examples = 0

    def update(self, y: torch.Tensor, t: torch.Tensor):
        """Update metric by mini batch"""
        self.loss_sum += self(y, t).item() * y.shape[0]
        self.n_examples += y.shape[0]

    def compute(self):
        """Compute metric for dataset"""
        return self.loss_sum / self.n_examples


class MyLogLoss(MeanLoss):
    def __init__(self, **params):
        super(MyLogLoss, self).__init__()
        self.loss_func = nn.BCEWithLogitsLoss(**params)


class LSBCEWithLogitsLoss(nn.Module):
    """"""

    def __init__(self, k: int, alpha: float = 0.01):
        """"""
        super(LSBCEWithLogitsLoss, self).__init__()
        self.k = k
        self.alpha = alpha
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, y, t):
        """"""
        t_s = t * (1 - self.alpha) + self.alpha / self.k
        loss = self.loss_func(y, t_s)
        return loss


class MyLSLogLoss(MeanLoss):
    def __init__(self, **params):
        super(MyLSLogLoss, self).__init__()
        self.loss_func = LSBCEWithLogitsLoss(**params)


def run_train_loop(
    manager, args, model, device, train_loader, optimizer, scheduler, loss_func
):
    """Run minibatch training loop"""
    while not manager.stop_trigger:
        model.train()
        for batch in train_loader:
            x, t = batch
            with manager.run_iteration():
                optimizer.zero_grad()
                y = model(x.to(device))
                loss = loss_func(y, t.to(device))
                # ppe.reporting.report({"train/loss": loss.item()})
                loss.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()


def run_eval(args, model, device, batch, eval_manager):
    """Run evaliation for val or test. this function is applied to each batch."""
    model.eval()
    x, t = batch
    y = model(x.to(device))
    eval_manager(y, t.to(device))


def get_optimizer(settings, model):
    optimizer = getattr(torch.optim, settings["optimizer"]["name"])(
        model.parameters(), **settings["optimizer"]["params"]
    )
    return optimizer


def get_scheduler(settings, train_loader, optimizer):
    if settings["scheduler"]["name"] is None:
        scheduler = None
    else:
        if settings["scheduler"]["name"] == "OneCycleLR":
            settings["scheduler"]["params"]["epochs"] = settings["globals"]["max_epoch"]
            settings["scheduler"]["params"]["steps_per_epoch"] = len(train_loader)

        scheduler = getattr(torch.optim.lr_scheduler, settings["scheduler"]["name"])(
            optimizer, **settings["scheduler"]["params"]
        )
    return scheduler


def get_loss_function(settings):
    if hasattr(nn, settings["loss"]["name"]):
        loss_func = getattr(nn, settings["loss"]["name"])(**settings["loss"]["params"])
    else:
        loss_func = eval(settings["loss"]["name"])(**settings["loss"]["params"])
    return loss_func


# def get_manager(
#    settings,
#    model,
#    device,
#    train_loader,
#    val_loader,
#    optimizer,
#    eval_manager,
#    output_path,
# ):
#    trigger = ppe.training.triggers.EarlyStoppingTrigger(
#        check_trigger=(1, "epoch"),
#        monitor="val/metric",
#        mode="min",
#        patience=settings["globals"]["patience"],
#        verbose=True,
#        max_trigger=(settings["globals"]["max_epoch"], "epoch"),
#    )
#
#    manager = ppe.training.ExtensionsManager(
#        model,
#        optimizer,
#        settings["globals"]["max_epoch"],
#        iters_per_epoch=len(train_loader),
#        stop_trigger=trigger,
#        out_dir=output_path,
#    )
#
#    log_extentions = [
#        ppe_extensions.observe_lr(optimizer=optimizer),
#        ppe_extensions.LogReport(),
#        ppe_extensions.PlotReport(
#            ["train/loss", "val/loss"], "epoch", filename="loss.png"
#        ),
#        ppe_extensions.PlotReport(["lr"], "epoch", filename="lr.png"),
#        ppe_extensions.PrintReport(
#            [
#                "epoch",
#                "iteration",
#                "lr",
#                "train/loss",
#                "val/loss",
#                "val/metric",
#                "elapsed_time",
#            ]
#        ),
#    ]
#    for ext in log_extentions:
#        manager.extend(ext)
#
#    manager.extend(  # evaluation
#        ppe_extensions.Evaluator(
#            val_loader,
#            model,
#            eval_func=lambda *batch: run_eval(
#                settings, model, device, batch, eval_manager
#            ),
#        ),
#        trigger=(1, "epoch"),
#    )
#
#    manager.extend(  # model snapshot
#        ppe_extensions.snapshot(target=model, filename="snapshot_epoch_{.epoch}.pth"),
#        trigger=ppe.training.triggers.MinValueTrigger(
#            key="val/metric", trigger=(1, "epoch")
#        ),
#    )
#
#    return manager


def set_random_seed(seed: int = 42, deterministic: bool = False):
    """Set seeds"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    if deterministic:
        torch.backends.cudnn.deterministic = True  # type: ignore


def run_train_one_fold(
    settings, model, train_all_dataset, train_val_index, device, output_path
):
    """Run training for one fold"""
    train_dataset = data.Subset(train_all_dataset, train_val_index[0])
    val_dataset = data.Subset(train_all_dataset, train_val_index[1])
    train_loader = data.DataLoader(train_dataset, **settings["loader"]["train"])
    val_loader = data.DataLoader(val_dataset, **settings["loader"]["val"])
    print("train: {}, val: {}".format(len(train_dataset), len(val_dataset)))

    model.to(device)
    optimizer = get_optimizer(settings, model)
    scheduler = get_scheduler(settings, train_loader, optimizer)
    loss_func = get_loss_function(settings)
    loss_func.to(device)

    eval_mgr = EvalFuncManager(
        len(val_loader), {"loss": loss_func, "metric": MyLogLoss(),}
    )

    manager = get_manager(
        settings,
        model,
        device,
        train_loader,
        val_loader,
        optimizer,
        eval_mgr,
        output_path,
    )

    run_train_loop(
        manager, settings, model, device, train_loader, optimizer, scheduler, loss_func
    )


######################################################################################
