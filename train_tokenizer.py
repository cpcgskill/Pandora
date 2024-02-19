# -*-coding:utf-8 -*-
"""
:创建时间: 2023/9/24 23:10
:作者: 苍之幻灵
:我的主页: https://cpcgskill.com
:Github: https://github.com/cpcgskill
:QQ: 2921251087
:aboutcg: https://www.aboutcg.org/teacher/54335
:bilibili: https://space.bilibili.com/351598127
:爱发电: https://afdian.net/@Phantom_of_the_Cang

"""
from __future__ import unicode_literals, print_function, division

from dataproc.compile_dataset import get_merge_custom_answer_dataset
from pandora.config import Config

config = Config()
source_dataset = get_merge_custom_answer_dataset()

from pandora.tokenizer_ import train_tokenizer

train_tokenizer(
    config,
    source_dataset
)
