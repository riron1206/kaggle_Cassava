#!/usr/bin/python3
# -*- coding:utf-8 -*-
import os
import numpy as np
import pandas as pd
import utils
import models
import optims
import commons
import augments
import datasets
from absl import app, flags
import torch
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score

""" 引数 """
FLAGS = flags.FLAGS
# Environment:
flags.DEFINE_enum('run', 'local', [ 'local', 'server' ], '実行マシンを指定します.')
# Random:
flags.DEFINE_integer('seed', 123456, '乱数シードを指定します.')
# Model:
flags.DEFINE_string('model', 'efficientnet-b4', 'モデル名を指定します.')
flags.DEFINE_string('model_uri', None, 'モデルパラメータファイルパスを指定します.')
# Optim:
flags.DEFINE_string('optim', 'adam', '最適化手法を指定します.')
flags.DEFINE_string('optim_uri', None, '最適化手法パラメータファイルパスを指定します.')
# Scheduler:
flags.DEFINE_string('scheduler', 'sc1', '学習率スケジュールを指定します.')
# Training Parameter:
flags.DEFINE_integer('epoch', 20, 'エポック数を指定します.')
flags.DEFINE_integer('batch', 24, 'バッチサイズを指定します.')
flags.DEFINE_float('lr', 1e-3, '学習率を指定します')
# Augment:
flags.DEFINE_string('train_augs', None, '学習オーグメンテーション指定します.')
flags.DEFINE_string('valid_augs', None, '検証オーグメンテーション指定します.')
# Data:
flags.DEFINE_integer('image_size', 512, '画像サイズを指定します.')
# Data I/O:
flags.DEFINE_string('train_csv', None, '学習用入力データ(train.csv)を指定します.')
flags.DEFINE_string('valid_csv', None, '検証用入力データ(valid.csv)を指定します.')
flags.DEFINE_string('image_dir', './data/images', '学習用入力画像ディレクトリを指定します.')
flags.DEFINE_string('output_dir', './outputs', '出力先ディレクトリを指定します.')
# DataLoader:
flags.DEFINE_integer('num_workers', 4, 'ワーカー数を指定します.')
# Other:
flags.DEFINE_bool('save_optim', False, '最適化手法のパラメータファイルを保存要否を指定します.')
# Extra Option:
flags.DEFINE_integer('undersample', None, 'Class3のアンダーサンプルを実施します. Noneを指定した場合は実行されません.')

def main(argv):
    """
    主処理部分
    """
    conf = utils.getParamDict(FLAGS)
    conf = utils.setOutputDir(conf)
    conf = utils.setDevice(conf)
    conf = utils.setSeed(conf)
    train_runner(conf)

def train_setup(conf):
    """
    入力データ設定
    """
    # クラス数:
    num_classes = 5
    # train:
    train_df = pd.read_csv(conf['train_csv'])
    # valid:
    valid_df = pd.read_csv(conf['valid_csv'])
    return train_df, valid_df, num_classes

def train_runner(conf):
    """
    学習メイン処理
    """
    train_df, valid_df, num_classes = train_setup(conf)
    train_aug = augments.getTrainAugs(size=conf['image_size'], name=conf['train_augs'])
    valid_aug = augments.getValidAugs(size=conf['image_size'], name=conf['valid_augs'])
    train_loader = datasets.getTrainLoader(conf, train_df, batch=conf['batch'], augment_op=train_aug, undersample=conf['undersample'])
    valid_loader = datasets.getValidLoader(conf, valid_df, batch=conf['batch'], augment_op=valid_aug)
    model = models.getModel(conf, num_classes=num_classes).to(conf['device'])
    optim = optims.getOptim(conf, model)
    scheduler, everystep = optims.getScheduler(conf, optim)
    train_loss_fn = torch.nn.CrossEntropyLoss()
    valid_loss_fn = torch.nn.CrossEntropyLoss()
    # 学習準備:
    conf['stdout'] = commons.WrapStdout(os.path.join(conf['output_dir'], 'stdout.txt'))
    conf['tboard'] = commons.WrapTensorboard(log_dir=conf['output_dir'])
    stdout = conf['stdout']
    tboard = conf['tboard']
    epoch = conf['epoch']
    score = {
        'epoch' : 0,
        'value' : 0,
        'metric_name' : 'accuracy'
    }
    scaler = GradScaler()
    stdout('学習開始:')
    for e in range(epoch):
        # 学習/検証:
        train_loss = train_step(conf, model, train_loader, train_loss_fn, optim, scaler, scheduler if everystep else None)
        valid_loss, valid_score = valid_step(conf, model, valid_loader, valid_loss_fn)
        optims.stepScheduler(conf, scheduler, everystep, valid_loss)
        # データローダー更新:
        if hasattr(train_loader.dataset, 'shuffle'):
            train_loader.dataset.shuffle()
        # ログ表示:
        updated = (score['value'] < valid_score)
        logstr = 'Epoch: {:2}, lr: {:.4e}, t-loss: {:.4e}, v-loss: {:.4e}, {}: {:.4f}'.format(e + 1, utils.getCurrentLR(optim), train_loss, valid_loss, score['metric_name'], valid_score)
        logstr = logstr + (' (*)' if updated else '')
        stdout(logstr)
        if updated:
            score['value'] = valid_score
            score['epoch'] = e + 1
            utils.savePth(conf, model, optim, epoch=e + 1)
        # 現在パラメータを保存:
        utils.savePth(conf, model, optim)
        # Tensorboard:
        tboard.writeScalar(score['metric_name'], e + 1, { 'best' : score['value'], 'current' : valid_score })
        tboard.writeScalar('loss', e + 1, { 'train' : train_loss, 'valid' : valid_loss })
        tboard.writeScalar('lr', e + 1, utils.getCurrentLR(optim))
        # 終了チェック:
        if utils.getQuitCommand(e + 1):
            stdout('--- 学習を停止します ---')
            break
    utils.saveBestPth(conf, score['epoch'])
    tboard.close()
    stdout('学習完了')
    stdout.close()

def train_step(conf, model, loader, loss_fn, optim, scaler, scheduler):
    """
    1epochごとの学習処理
    """
    # backward function:
    def gpu_backward(x, t):
        with autocast():
            y = model(x)
            loss = loss_fn(y, t)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        return loss
    def cpu_backward(x, t):
        y = model(x)
        loss = loss_fn(y, t)
        loss.backward()
        optim.step()
        return loss
    backward_fn = gpu_backward if conf['device'] != 'cpu' else cpu_backward
    # process:
    value = 0
    count = 0
    model.train()
    wtqdm = commons.WrapTqdm(total=len(loader), run=conf['run'])
    for _, itr in enumerate(loader):
        optim.zero_grad()
        x, t = itr
        bs = x.shape[0]
        x = x.to(conf['device'])
        t = t.to(conf['device'])
        loss = backward_fn(x, t)
        if scheduler is not None:
            scheduler.step()
        value += bs * loss.item()
        count += bs
        wtqdm.set_description('  train loss = {:e}'.format(value / count))
        wtqdm.update()
    wtqdm.close()
    value = value / count if count > 0 else 0
    return value

def valid_step(conf, model, loader, loss_fn):
    """
    1epochごとの検証処理
    """
    y_true = [ ]
    y_pred = [ ]
    value = 0
    count = 0
    model.eval()
    with torch.no_grad():
        actfn = torch.nn.Softmax(dim=1)
        wtqdm = commons.WrapTqdm(total=len(loader), run=conf['run'])
        for _, itr in enumerate(loader):
            x, t = itr
            bs = x.shape[0]
            x = x.to(conf['device'])
            t = t.to(conf['device'])
            y = model(x)
            loss = loss_fn(y, t)
            value += bs * loss.item()
            count += bs
            wtqdm.set_description('  valid loss = {:e}'.format(value / count))
            wtqdm.update()
            y = actfn(y)
            y = y.detach().cpu().numpy()
            y = np.argmax(y, axis=1)
            t = t.detach().cpu().numpy()
            y_pred.extend(y)
            y_true.extend(t)
        wtqdm.close()
    value = value / count if count > 0 else 0
    score = accuracy_score(y_true, y_pred)
    return value, score

""" エントリポイント """
if __name__ == "__main__":
    app.run(main)
