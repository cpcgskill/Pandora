# -*-coding:utf-8 -*-
"""
:创建时间: 2023/9/1 3:07
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


def preprocess_function(data):
    keys = data.formatter.features.keys()
    data = [
        ''.join(f"<{key}>{data[key][i]}</{key}>" for key in keys)
        for i in range(len(list(data.values())[0]))
    ]
    return {'text': data}


def compile_dataset(keep_in_memory=False):
    cache_dir = './hugging_hub_cache'
    cpu_count = os.cpu_count()
    load_dataset_config = {
        'cache_dir': cache_dir,
        'keep_in_memory': keep_in_memory,
        'num_proc': cpu_count,
    }

    # load dataset
    wikipedia_cn = load_dataset(
        'pleisto/wikipedia-cn-20230720-filtered',
        split="train",
        **load_dataset_config,
    )
    wikipedia_cn = wikipedia_cn.map(
        batched=True,
        num_proc=cpu_count,
        keep_in_memory=keep_in_memory,
        remove_columns=['source'],
    )
    wikipedia_simple = load_dataset(
        'wikipedia', '20220301.simple',
        beam_runner='DirectRunner',
        split='train',
        **load_dataset_config,
    )
    wikipedia_simple = wikipedia_simple.map(
        batched=True,
        num_proc=cpu_count,
        keep_in_memory=keep_in_memory,
        remove_columns=['id', 'url'],
    )

    wikipedia_en = load_dataset(
        'wikipedia', '20220301.en',
        beam_runner='DirectRunner',
        split='train',
        **load_dataset_config,
    )
    wikipedia_en = wikipedia_en.map(
        batched=True,
        num_proc=cpu_count,
        keep_in_memory=keep_in_memory,
        remove_columns=['id', 'url'],
    )
    tmp_hc3_cn = load_dataset(
        'Hello-SimpleAI/HC3-Chinese', 'all',
        split="train",
        **load_dataset_config,
    )
    hc3_cn_gpt = tmp_hc3_cn.map(
        batched=True,
        num_proc=cpu_count,
        keep_in_memory=keep_in_memory,
        remove_columns=['id', 'human_answers'],
    )
    hc3_cn_human = tmp_hc3_cn.map(
        batched=True,
        num_proc=cpu_count,
        keep_in_memory=keep_in_memory,
        remove_columns=['id', 'chatgpt_answers'],
    )

    dataset_config = [
        wikipedia_cn,
        load_dataset(
            'fka/awesome-chatgpt-prompts',
            split="train",
            **load_dataset_config,
        ),
        load_dataset(
            'sahil2801/CodeAlpaca-20k',
            split="train",
            **load_dataset_config,
        ),
        wikipedia_simple,
        wikipedia_en,
        hc3_cn_gpt,
        hc3_cn_human,
    ]

    dataset_config = [
        i.map(
            preprocess_function,
            batched=True,
            num_proc=cpu_count,
            keep_in_memory=keep_in_memory,
            remove_columns=i.column_names,
        )
        for i in dataset_config
    ]
    dataset = concatenate_datasets(dataset_config)
    print('dataset[0]', dataset[0])
    # shuffle dataset
    dataset = dataset.shuffle(1024 * 1024 * 16)
    print('dataset[0]', dataset[0])
    # write to file
    dataset.save_to_disk('./dataset')


def get_dataset(keep_in_memory=False):
    # load from file
    dataset = load_from_disk('./dataset', keep_in_memory=keep_in_memory)
    return dataset


if __name__ == '__main__':
    dataset = get_dataset()
    print('dataset[0]', dataset[0])
    # compile_dataset()
