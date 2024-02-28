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

import dataclasses


@dataclasses.dataclass
class Config:
    tokenizer_path = './tokenizer.json'
    # model config
    embed_size: int = 768
    feedforward_dim: int = embed_size * 8
    num_layers: int = 16
    num_heads: int = 12
    dropout: float = 0.1

    # train config
    gradient_accumulation_steps: int = 12

    warmup_epochs: int = 30_000
    init_lr: float = 6.521404165561944e-08
    max_lr: float = 0.00016352170540541642
    gamma: float = 0.9835730074875503

    prediction_length: int = 1
    main_prediction_length: int = 1

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
