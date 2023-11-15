# -*-coding:utf-8 -*-
"""
:创建时间: 2023/11/15 1:57
:作者: 苍之幻灵
:我的主页: https://cpcgskill.com
:Github: https://github.com/cpcgskill
:QQ: 2921251087
:aboutcg: https://www.aboutcg.org/teacher/54335
:bilibili: https://space.bilibili.com/351598127
:爱发电: https://afdian.net/@Phantom_of_the_Cang

"""
from __future__ import unicode_literals, print_function, division

tokenizer_path = 'tokenizer2.json'
module = {
    "embed_size": 768,
    "num_layers": 64,
    "heads": 12,
    "dropout": 0.1,
    "res_net_block_num": 64,
    "res_net_block_layer_num": 6
}

train = {
    "gradient_accumulation_steps": 32,

    "random_effect": 0.001,
    "random_epoch": 1200,
    "warmup_epochs": 10_000,
    "init_lr": 0.0000005,
    "max_lr": 0.000025,
    "gamma": 0.97,
}
