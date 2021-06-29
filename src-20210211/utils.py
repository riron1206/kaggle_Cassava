import os
import numpy as np
import shutil
import random
import multiprocessing as mp
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast
from typing import Dict, Tuple

def getParamDict(args) -> Dict:
    """
    abslの引数を辞書型に変換する.
    """
    return { f : args[f].value for f in args }

def setSeed(conf: Dict) -> Dict:
    """
    各種の乱数シードを設定する.
    """
    # parameter:
    seed = conf['seed'] if 'seed' in conf and conf['seed'] is not None else 123456
    # process:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return conf

def setDevice(conf: Dict) -> Dict:
    """
    CPU/GPU数からデバイス設定を行う.
    """
    # parameter:
    outfn = conf['stdout'] if 'stdout' in conf and conf['stdout'] is not None else print
    # num_workersをCPU数から取得する.
    # 0または正の数が指定されている場合は指定された値を使用する.
    num_cpu = conf['num_workers'] if 'num_workers' in conf and conf['num_workers'] is not None and conf['num_workers'] >= 0 else mp.cpu_count()
    conf['num_workers'] = num_cpu
    # GPU数から自動決定する.
    # ただし,DistributedDataParallel(DDP)を利用する場合は専用の対応が必要のためオプションの確認を行う.
    ddp = conf['ddp'] if 'ddp' in conf and conf['ddp'] is not None else False
    idx = conf['local_rank'] if 'local_rank' in conf and conf['local_rank'] is not None else -1
    if ddp and idx >= 0:
        conf['device'] = f'cuda:{idx}' if torch.cuda.is_available() else 'cpu'
    else:
        dev = conf['device'] if 'device' in conf and conf['device'] is not None else 'auto'
        if torch.cuda.device_count() > 1:
            devstr = '0'
            for i in range(1, torch.cuda.device_count()):
                devstr += ',{}'.format(i)
            os.environ["CUDA_VISIBLE_DEVICES"]= devstr
        if dev == 'auto':
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif dev.startswith('cuda') and not torch.cuda.is_available():
            outfn('[警告] CUDAが使用できません. デバイス=CPUで実行します.')
            dev = 'cpu'
        conf['device'] = dev
    return conf

def setOutputDir(conf: Dict) -> Dict:
    """
    出力ディレクトリ設定する.
    """
    # parameter:
    outfn = conf['stdout'] if 'stdout' in conf and conf['stdout'] is not None else print
    # process:
    if 'output_dir' in conf and conf['output_dir'] is not None:
        os.makedirs(conf['output_dir'], exist_ok=True)
    else:
        outfn('[警告] 出力ディレクトリが指定されていません.')
    return conf

def setOutputSubDir(conf: Dict, subdir: str) -> Dict:
    """
    出力サブディレクトリ設定する.
    """
    # parameter:
    outfn = conf['stdout'] if 'stdout' in conf and conf['stdout'] is not None else print
    # process:
    if 'output_dir' in conf and conf['output_dir'] is not None:
        os.makedirs(os.path.join(conf['output_dir'], subdir), exist_ok=True)
    else:
        outfn('[警告] 出力ディレクトリが指定されていません.')
    return conf

def savePth(conf: Dict, model: nn.Module, optim, epoch=None) -> Dict:
    """
    PyTorchのモデルパラメータを保存するユーティリティ.
    """
    # parameter:
    save_optim = conf['save_optim'] if 'save_optim' in conf and conf['save_optim'] is not None else False
    # process:
    if epoch is None or (type(epoch) != int and type(epoch) != str):
        mpath = os.path.join(conf['output_dir'], 'model_last.pth')
        opath = os.path.join(conf['output_dir'], 'optim_last.pth')
        mdict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(mdict, mpath)
        if save_optim:
            torch.save(optim.state_dict(), opath)
    else:
        mpath = os.path.join(conf['output_dir'], 'model_ep{}.pth'.format(epoch))
        opath = os.path.join(conf['output_dir'], 'optim_ep{}.pth'.format(epoch))
        mdict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(mdict, mpath)
        if save_optim:
            torch.save(optim.state_dict(), opath)
    return conf

def saveBestPth(conf: Dict, epoch: int) -> Dict:
    """
    PyTorch パラメータの最良を取得して保存するためのユーティリティ.
    """
    # parameter:
    outfn = conf['stdout'] if 'stdout' in conf and conf['stdout'] is not None else print
    # process:
    mpath = os.path.join(conf['output_dir'], 'model_ep{}.pth'.format(epoch))
    if not os.path.exists(mpath):
        logstr = '[警告] 指定したパラメータファイルは存在しません. ({})'.format(mpath)
        outfn(logstr)
    else:
        shutil.copy(mpath, os.path.join(conf['output_dir'], 'model_best.pth'))
    return conf

def getCurrentLR(optim) -> float:
    """
    最適化エンジンの現在の学習率を取得するためのユーティリティ.
    """
    return optim.param_groups[0]['lr']

def setAutoMixedPrecision(conf: Dict, model: nn.Module) -> nn.Module:
    """
    Auto Mixed Precision (AMP)を設定する.
    """
    # parameter:
    ddp = conf['ddp'] if 'ddp' in conf and conf['ddp'] is not None else False
    # process:
    if not ddp:
        model = nn.DataParallel(model)
    if conf['device'] != 'cpu':
        model.forward = autocast()(model.forward)
    return model

def getQuitCommand(epoch=None) -> bool:
    """
    終了ファイルがあるか確認する.
    もしファイルが存在する場合は学習を停止する.
    """
    if os.path.exists('quit') or os.path.exists(f'quit.asap'):
        return True
    if epoch is not None and (type(epoch) == int or type(epoch) == str) and os.path.exists(f'quit.{epoch}'):
        return True
    return False

def enterDDP(conf: Dict, addr: str = 'localhost', port: str = '65501') -> None:
    """
    Distributed Data Parallelのグループを作成する.
    """
    ddp_rank = conf['local_rank']
    if ddp_rank is not None:
        ddp_size = torch.cuda.device_count()
        os.environ['MASTER_ADDR'] = addr
        os.environ['MASTER_PORT'] = port
        dist.init_process_group('nccl', rank=ddp_rank, world_size=ddp_size)
        torch.cuda.set_device(ddp_rank)

def leaveDDP(conf: Dict) -> None:
    """
    Distributed Data Parallelのグループを解放する.
    """
    ddp_rank = conf['local_rank']
    if ddp_rank is not None:
        dist.destroy_process_group()
        conf['local_rank'] = None

def syncronize(conf: Dict) -> None:
    """
    Distributed Data Parallelが利用できるときのみ、各GPUでの処理を同期する.
    """
    ddp_rank = conf['local_rank']
    if ddp_rank is not None:
        dist.barrier()

def setSyncronizedBatchNorm(conf: Dict, model: nn.Module) -> nn.Module:
    """
    Distributed Data Parallelが利用できるときのみ、モデルのBatchNormをSyncronizedBatchNormに変換する.
    """
    ddp_rank = conf['local_rank']
    if ddp_rank is not None:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model

def mixup(x: torch.Tensor, t: torch.Tensor, alpha: float, ratio: float) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    mixup - https://arxiv.org/abs/1710.09412
    arXiv上の論文中疑似コードではDataLoaderを複数用意して流している. 本処理ではミニバッチ内で混合してmixupを実施する.
    """
    if np.random.rand() < ratio:
        return x, t, 0
    j = torch.randperm(x.size(0))
    y = x[j]
    u = t[j]
    a = np.random.beta(alpha, alpha)
    w = a * x + (1.0 - a) * y
    return w, u, a
