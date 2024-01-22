# -*-coding:utf-8 -*-
"""
:创建时间: 2023/12/27 3:11
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

import json
import random

import datasets

from dataproc.compile_dataset import get_custom_answer_dataset
from dataproc.post_processe import *
from pandora.tokenizer_ import get_tokenizer
from pandora.config import Config
from dataproc.utils import dataset_cache

config = Config()


@dataset_cache
def generate_train_and_test_dataset(dataset, keep_in_memory=False, chunk_size=512):
    dataset = pretokenize_dataset(get_tokenizer(config), dataset, keep_in_memory=keep_in_memory)
    dataset = segmentation_dataset(dataset, chunk_size=chunk_size)
    train_dataset, test_dataset = split_dataset(dataset, keep_in_memory=keep_in_memory)
    return train_dataset, test_dataset


train_dataset, test_dataset = generate_train_and_test_dataset(get_custom_answer_dataset(), chunk_size=1024)


def start():
    from pandora.kernel import train_transformer
    configs = [
        Config(
            init_lr=Config.init_lr * 0.5,
            max_lr=Config.max_lr * 2,
        ),
        Config(
            init_lr=Config.init_lr * 0.5,
        ),
    ]
    new_train_dataset = datasets.concatenate_datasets([
        train_dataset.shuffle(keep_in_memory=True)
        for _ in range(20)
    ])
    for i in range(20):
        new_config = Config(
            init_lr=Config.init_lr + random.uniform(-Config.init_lr, Config.init_lr) + 1e-12,
            max_lr=Config.max_lr + random.uniform(-Config.max_lr, Config.max_lr) + 1e-12,
            gamma=Config.gamma + random.uniform(-(1 - Config.gamma), (1 - Config.gamma)) - 1e-12,
        )
        configs.append(new_config)
    for idx, i in enumerate(configs):
        try:
            train_transformer(
                i,
                new_train_dataset,
                './test/transformer{}'.format(idx),
            )
            with open('./test/{}.json'.format(idx), 'w', encoding='utf-8') as f:
                json.dump(i.to_dict(), f, ensure_ascii=False, indent=4)
        except:
            import traceback
            traceback.print_exc()

import accelerate

num_processes = 1
if num_processes > 1:
    accelerate.notebook_launcher(start, (), num_processes=num_processes, mixed_precision='fp16')
else:
    start()
