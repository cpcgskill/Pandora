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
    pass

import os
import functools

from datasets import load_from_disk, concatenate_datasets
import pandora.tokenizer_ as tokenizer_

cache_dir = './hugging_hub_cache'
cpu_count = os.cpu_count()


def pretokenize_function(tokenizer, data):
    return {'ids': [i.ids for i in tokenizer.encode_batch(data['text'])]}


def pretokenize_dataset(tokenizer, dataset, keep_in_memory=False):
    dataset = dataset.map(
        functools.partial(pretokenize_function, tokenizer),
        batched=True,
        num_proc=cpu_count,
        keep_in_memory=keep_in_memory,
        remove_columns=dataset.column_names,
    )
    return dataset


def _segmentation_function(data, chunk_size=512, seam=5):
    output = []
    for i in data['ids']:
        if len(i) < chunk_size:
            output.append(i)
        else:
            start_ids = list(range(0, len(i), chunk_size - seam))
            output.extend(i[j:j + chunk_size] for j in start_ids[:-1])
            last_start_id = len(i) - chunk_size
            output.append(i[last_start_id:])
    return {'ids': output}


def segmentation_dataset(dataset, chunk_size=512, seam=5, keep_in_memory=False):
    # segmentation
    dataset = dataset.map(
        functools.partial(_segmentation_function, chunk_size=chunk_size, seam=seam),
        batched=True,
        num_proc=cpu_count,
        keep_in_memory=keep_in_memory,
        remove_columns=dataset.column_names,
    )
    return dataset


def extract_tail_function(data, chunk_size=512):
    return {'ids': [i[-chunk_size:] for i in data['ids']]}


def extract_tail_dataset(dataset, chunk_size=512, keep_in_memory=False):
    # segmentation
    dataset = dataset.map(
        functools.partial(extract_tail_function, chunk_size=chunk_size),
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


def generate_pretokenize_dataset(tokenizer, keep_in_memory=False):
    dataset = load_from_disk('./dataset/main')
    dataset = pretokenize_dataset(tokenizer, dataset)
    dataset.save_to_disk('./dataset/main_pretokenize')


def generate_train_and_test_dataset(keep_in_memory=False, chunk_size=512):
    dataset = load_from_disk('./dataset/main_pretokenize')
    dataset = segmentation_dataset(dataset, chunk_size=chunk_size)
    train_dataset, test_dataset = split_dataset(dataset, keep_in_memory=keep_in_memory)
    # write to file
    train_dataset.save_to_disk('./dataset/train')
    test_dataset.save_to_disk('./dataset/test')
    print('train_dataset[0]', train_dataset[0])
    print('test_dataset[0]', test_dataset[0])


def get_main_dataset(keep_in_memory=False):
    # load from file
    dataset = load_from_disk('./dataset/main', keep_in_memory=keep_in_memory)
    return dataset


def get_pretokenize_dataset(keep_in_memory=False):
    # load from file
    dataset = load_from_disk('./dataset/main_pretokenize', keep_in_memory=keep_in_memory)
    return dataset


def get_train_dataset(keep_in_memory=False):
    # load from file
    dataset = load_from_disk('./dataset/train', keep_in_memory=keep_in_memory)
    return dataset


def get_test_dataset(keep_in_memory=False):
    # load from file
    dataset = load_from_disk('./dataset/test', keep_in_memory=keep_in_memory)
    return dataset


def process_embedding_dataset(data):
    return {'pair': list(zip(data['ids'][:-1], data['ids'][1:]))}


def generate_embedding_dataset(tokenizer, keep_in_memory=False):
    dataset_list = []

    dataset = load_from_disk('./dataset/base')
    dataset = pretokenize_dataset(tokenizer, dataset)
    dataset = dataset.map(
        process_embedding_dataset,
        batched=True,
        num_proc=cpu_count,
        keep_in_memory=keep_in_memory,
        remove_columns=dataset.column_names,
    )
    dataset_list.append(dataset)

    dataset = load_from_disk('./dataset/textbook_base')
    dataset = pretokenize_dataset(tokenizer, dataset)
    dataset = dataset.map(
        process_embedding_dataset,
        batched=True,
        num_proc=cpu_count,
        keep_in_memory=keep_in_memory,
        remove_columns=dataset.column_names,
    )
    dataset_list.append(dataset)
    # concatenate
    dataset = concatenate_datasets(dataset_list)
    # write to file
    dataset.save_to_disk('./dataset/embedding')
    print('dataset[0]', dataset[0])


if __name__ == '__main__':
    generate_embedding_dataset()
