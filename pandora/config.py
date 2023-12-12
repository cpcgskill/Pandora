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

tokenizer_path = './tokenizer.json'

module = {
    "embed_size": 768,
    "feedforward_size": 768*8,
    "num_layers": 8,
    "heads": 12,
    "dropout": 0.1,
}

train = {
    "gradient_accumulation_steps": 12,

    "warmup_epochs": 10_000,
    "init_lr": 0.00000025,
    "max_lr": 0.0000125,
    "gamma": 0.97,

    "prediction_length": 3,
    "main_prediction_length": 1,
}
