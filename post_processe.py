# -*-coding:utf-8 -*-
"""
:创建时间: 2023/11/12 1:50
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

import os
from datasets import load_dataset, concatenate_datasets, load_from_disk
import SkipGram

cache_dir = './hugging_hub_cache'
cpu_count = os.cpu_count()
tokenizer = SkipGram.get_tokenizer()
chunk_size = 698


def pretokenize_function(data):
    return {'ids': [i.ids for i in tokenizer.encode_batch(data['text'])]}


def pretokenize_dataset(keep_in_memory=False):
    dataset = load_from_disk('./dataset/base')
    dataset = dataset.map(
        pretokenize_function,
        batched=True,
        num_proc=cpu_count,
        keep_in_memory=keep_in_memory,
        remove_columns=dataset.column_names,
    )
    dataset.save_to_disk('./dataset/pretokenize')
    return dataset


def _segmentation_function(data):
    output = []
    for i in data['ids']:
        output.extend(i[j:j + chunk_size] for j in range(0, len(i), chunk_size))
    return {'ids': output}


def segmentation_dataset(dataset, keep_in_memory=False):
    # segmentation
    dataset = dataset.map(
        _segmentation_function,
        batched=True,
        num_proc=cpu_count,
        keep_in_memory=keep_in_memory,
        remove_columns=dataset.column_names,
    )
    return dataset


def split_dataset(dataset, keep_in_memory=False):
    # split dataset for train and test
    dataset = dataset.train_test_split(test_size=0.1, keep_in_memory=keep_in_memory)
    return dataset['train'], dataset['test']


def process_dataset(keep_in_memory=False):
    # load from file
    dataset = load_from_disk('./dataset/pretokenize')

    dataset = segmentation_dataset(dataset)
    train_dataset, test_dataset = split_dataset(dataset)

    # write to file
    train_dataset.save_to_disk('./dataset/train')
    test_dataset.save_to_disk('./dataset/test')

    return train_dataset, test_dataset


def get_train_dataset(keep_in_memory=False):
    # load from file
    dataset = load_from_disk('./dataset/train', keep_in_memory=keep_in_memory)
    return dataset


def get_test_dataset(keep_in_memory=False):
    # load from file
    dataset = load_from_disk('./dataset/test', keep_in_memory=keep_in_memory)
    return dataset


if __name__ == '__main__':
    train_dataset, test_dataset = process_dataset()
    print('train_dataset[0]', train_dataset[0])
    print('test_dataset[0]', test_dataset[0])
