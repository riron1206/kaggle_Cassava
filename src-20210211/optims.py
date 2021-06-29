import torch
import torch.optim
import torch.optim.lr_scheduler
from typing import Dict, Tuple

def getOptim(conf: Dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    最適化エンジンを取得するためのユーティリティ.
    """
    # parameter:
    oname = conf['optim'] if 'optim' in conf and conf['optim'] is not None else 'adam'
    # process:
    def getKey(defval, keyword):
        return conf[keyword] if keyword in conf and conf[keyword] is not None else defval
    if oname == 'sgd':
        mt = getKey(0.9, 'momentum')
        wd = getKey(0.0, 'weight_decay')
        optim = torch.optim.SGD(model.parameters(), lr=conf['lr'], momentum=mt, weight_decay=wd)
    elif oname == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=conf['lr'], betas=(0.5, 0.999))
    else:
        raise NameError('指定された最適化エンジンは定義されていません. (scheduler={})'.format(oname))
    # パラメータを読込
    if 'optim_uri' in conf and conf['optim_uri'] is not None:
        state_dict = torch.load(conf['optim_uri'], map_location=conf['device'])
        model.load_state_dict(state_dict, strict=True)
    return optim

def getScheduler(conf: Dict, optim: torch.optim.Optimizer) -> Tuple[torch.optim.lr_scheduler._LRScheduler, bool]:
    """
    スケジューラーを取得するためのユーティリティ.
    """
    # parameter:
    sname = conf['scheduler'] if 'scheduler' in conf and conf['scheduler'] is not None else 'default'
    # process:
    if sname == 'default':
        scheduler = None
        everystep = False
    elif sname == 'sc0':
        min_lr = conf['lr'] * 1e-3
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=conf['epoch'], eta_min=min_lr)
        everystep = False
    elif sname.startswith('sc0:'):
        param = sname[len('sc1:'):].split(':')
        min_lr = float(param[0])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=conf['epoch'], eta_min=min_lr)
        everystep = False
    elif sname == 'sc1':
        max_lr = conf['lr']
        min_lr = conf['lr'] * 1e-3
        nstep = 1000
        scheduler = torch.optim.lr_scheduler.CyclicLR(optim, min_lr, max_lr, step_size_up=nstep, step_size_down=None, mode='exp_range', gamma=0.9995, scale_fn=None, scale_mode='cycle', cycle_momentum=False)
        everystep = True
    elif sname.startswith('sc1:'):
        max_lr = conf['lr']
        min_lr = conf['lr'] * 1e-3
        param = sname[len('sc1:'):].split(':')
        nstep = int(param[0]) if len(param) > 0 and param[0] != '' else 1000
        gamma = float(param[1]) if len(param) > 1 and param[1] != '' else 0.9995
        scheduler = torch.optim.lr_scheduler.CyclicLR(optim, min_lr, max_lr, step_size_up=nstep, step_size_down=None, mode='exp_range', gamma=gamma, scale_fn=None, scale_mode='cycle', cycle_momentum=False)
        everystep = True
    elif sname == 'sc2':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min')
        everystep = False
    elif sname.startswith('sc2:'):
        param = sname[len('sc2:'):]
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, param)
        everystep = False
    else:
        raise NameError('指定されたスケジューラーは定義されていません. (scheduler={})'.format(sname))
    return scheduler, everystep

def stepScheduler(conf, scheduler, everystep, valid_loss) -> None:
    """
    スケジューラーの更新を行う.
    """
    if scheduler is not None and not everystep:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(valid_loss)
        else:
            scheduler.step()

""" カスタムオプティマイザ """


""" カスタムスケジューラー """
