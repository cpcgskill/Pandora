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


def start():
    from pandora.SkipGram import train_skip_gram, build_embedding_from_skip_gram

    # train_skip_gram(
    #     config,
    #     source_dataset.shuffle(keep_in_memory=True),
    #     './model/skip_gram7',
    # )

    # build_embedding_from_skip_gram(
    #     config,
    #     './model/skip_gram7',
    # )


import accelerate

num_processes = 1
if num_processes > 1:
    accelerate.notebook_launcher(start, (), num_processes=num_processes, mixed_precision='fp16')
else:
    start()
