from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from efficientnet_pytorch import EfficientNet

""" モジュール """
def swish(x, inplace: bool = False):
    """
    Swish - https://arxiv.org/abs/1710.05941
    """
    return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())

class Swish(nn.Module):
    """
    Swish - https://arxiv.org/abs/1710.05941
    """
    def __init__(self, inplace: bool = False):
        super(Swish, self).__init__()

    def forward(self, x):
        return swish(x, self.inplace)

def mish(x, inplace: bool = False):
    """
    Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    """
    return x.mul(F.softplus(x).tanh())

class Mish(nn.Module):
    """
    Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    """
    def __init__(self, inplace: bool = False):
        super(Mish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return mish(x)

def sigmoid(x, inplace: bool = False):
    """
    sigmoid
    """
    return x.sigmoid_() if inplace else x.sigmoid()

class Sigmoid(nn.Module):
    """
    sigmoid
    """
    def __init__(self, inplace: bool = False):
        super(Sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.sigmoid_() if self.inplace else x.sigmoid()

def tanh(x, inplace: bool = False):
    """
    tanh
    """
    return x.tanh_() if inplace else x.tanh()

class Tanh(nn.Module):
    """
    tanh
    """
    def __init__(self, inplace: bool = False):
        super(Tanh, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.tanh_() if self.inplace else x.tanh()

class Identity(nn.Module):
    """
    Identity
    """
    def __init__(self, inplace: bool = False):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

""" モデル取得 """
def getModel(conf: Dict, num_classes: int = 2, in_channels: int = 3, pretrained: bool = True) -> nn.Module:
    """
    モデルを取得するためのユーティリティ.
    """
    # parameter:
    mname = conf['model'] if 'model' in conf and conf['model'] is not None else None
    mckpt = conf['model_uri'] if 'model_uri' in conf and conf['model_uri'] is not None else None
    return getModelByName(conf, mname, num_classes=num_classes, in_channels=in_channels, pretrained=pretrained, pth=mckpt)

def getModelByName(conf: Dict, name: str, num_classes: int = 2, in_channels: int = 3, pretrained: bool = True, pth: str = None) -> nn.Module:
    """
    モデルを取得するためのユーティリティ.
    """
    # parameter:
    stdout = conf['stdout'] if 'stdout' in conf and conf['stdout'] is not None else print
    # process:
    if name is None:
        raise NameError('モデル定義がされていません.')
    param = name.split(':')
    model = None
    if param[0] == 'timm':
        if param[1].startswith('vit_'):
            subparam = param[1].split('_')
            psize = int(subparam[2][len('patch'):])
            isize = int(subparam[3])
            model = timm.create_model(model_name=param[1], num_classes=num_classes, in_chans=in_channels, pretrained=pretrained)
            if isize != 224 and isize != 384:
                w = model.patch_embed.proj.weight.data
                b = model.patch_embed.proj.bias.data
                model.patch_embed = timm.models.vision_transformer.PatchEmbed(img_size=isize, patch_size=psize, in_chans=in_channels, embed_dim=768)
                model.patch_embed.proj.load_state_dict( { 'weight' : w, 'bias' : b })
                model.pos_embed = nn.Parameter(torch.zeros(1, model.patch_embed.num_patches + 1, 768))
        elif param[1].startswith('vit_deit_'):
            subparam = param[1].split('_')
            psize = int(subparam[3][len('patch'):])
            isize = int(subparam[4])
            model = timm.create_model(model_name=param[1], num_classes=num_classes, in_chans=in_channels, pretrained=pretrained)
            if isize != 224 and isize != 384:
                w = model.patch_embed.proj.weight.data
                b = model.patch_embed.proj.bias.data
                model.patch_embed = timm.models.vision_transformer.PatchEmbed(img_size=isize, patch_size=psize, in_chans=in_channels, embed_dim=768)
                model.patch_embed.proj.load_state_dict( { 'weight' : w, 'bias' : b })
                model.pos_embed = nn.Parameter(torch.zeros(1, model.patch_embed.num_patches + 1, 768))
        else:
            model = timm.create_model(model_name=param[1], num_classes=num_classes, in_chans=in_channels, pretrained=pretrained)
    elif param[0].startswith('efficientnet-b'):
        model = EfficientNet.from_pretrained(param[0]) if pretrained else EfficientNet.from_name(param[0])
        model._fc = nn.Linear(model._fc.in_features, num_classes)
    else:
        pass
    if model is None:
        raise NameError('指定されたモデルは定義されていません. (model={})'.format(name))
    # パラメータを読込
    if pth is not None:
        state_dict = torch.load(pth, map_location='cpu')
        model.load_state_dict(state_dict, strict=conf['strict'])
        stdout('パラメータ読込: {}'.format(pth))
    return model
