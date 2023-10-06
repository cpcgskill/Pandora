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

if False:
    from typing import *

import torch
from torch import nn


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
