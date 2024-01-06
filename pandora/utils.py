# -*-coding:utf-8 -*-
"""
:创建时间: 2023/9/11 0:55
:作者: 苍之幻灵
:我的主页: https://cpcgskill.com
:Github: https://github.com/cpcgskill
:QQ: 2921251087
:aboutcg: https://www.aboutcg.org/teacher/54335
:bilibili: https://space.bilibili.com/351598127
:爱发电: https://afdian.net/@Phantom_of_the_Cang

"""
from __future__ import unicode_literals, print_function, division

import json
import os

if False:
    from typing import *

import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
import numpy as np
import matplotlib.pyplot as plt


def add_random_value_by_weights(module, std=0.1):
    """
    为模型的权重添加随机值

    :type module: nn.Module
    :type std: float
    """
    with torch.no_grad():
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                m.weight += torch.normal(mean=0, std=std, size=m.weight.size()).to(m.weight.device)
                if m.bias is not None:
                    m.bias.data += torch.normal(mean=0, std=std, size=m.bias.size()).to(m.weight.device)
            if isinstance(m, nn.Linear):
                m.weight += torch.normal(mean=0, std=std, size=m.weight.size()).to(m.weight.device)
                if m.bias is not None:
                    m.bias.data += torch.normal(mean=0, std=std, size=m.bias.size()).to(m.weight.device)
            if isinstance(m, nn.BatchNorm2d):
                m.weight += torch.normal(mean=0, std=std, size=m.weight.size()).to(m.weight.device)
                if m.bias is not None:
                    m.bias.data += torch.normal(mean=0, std=std, size=m.bias.size()).to(m.weight.device)


class WarmupScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_epochs, init_lr, max_lr,
                 gamma=0.9,
                 last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.gamma = gamma
        self.now_lr = init_lr
        self.lr_list = []
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            self.now_lr = self.init_lr + (self.max_lr - self.init_lr) * (self.last_epoch / self.warmup_epochs)
        elif self.last_epoch == self.warmup_epochs:
            self.now_lr = self.max_lr
        else:
            # lr = (self.max_lr - self.base_lr) * (1 / (self.last_epoch + 1)) + self.base_lr
            self.now_lr = self.now_lr * self.gamma
        self.lr_list.append(self.now_lr)
        return [self.now_lr for _ in self.base_lrs]


def print_loss_list_graph(loss_list):
    x = range(len(loss_list))
    y = loss_list

    plt.plot(x, y)

    # 设置标题和轴标签
    plt.title("Loss curve")
    plt.xlabel("step")
    plt.ylabel("loss")

    # 设置平均分布的刻度
    plt.grid(True)


    plt.show()


def save_loss_list_graph(loss_list, path):
    x = range(len(loss_list))
    y = loss_list

    plt.plot(x, y)

    # 设置标题和轴标签
    plt.title("Loss curve")
    plt.xlabel("step")
    plt.ylabel("loss")

    # 设置平均分布的参考线
    plt.grid(True)

    plt.savefig(path)


class TrainCtx:
    def __init__(self):
        self.step = 0
        self.loss_list = []

    def state_dict(self):
        return {
            'global_step': self.step,
            'loss_list': self.loss_list,
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict['global_step']
        self.loss_list = state_dict['loss_list']

    def save_loss_list_graph(self, path):
        save_loss_list_graph(self.loss_list, path)

    def export_loss_data(self, export_path):
        self.save_loss_list_graph(os.path.join(export_path, 'loss.png'))
        with open(os.path.join(export_path, 'loss.json'), 'wb') as f:
            f.write(json.dumps(self.loss_list).encode('utf-8'))
