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

import datasets

if False:
    from typing import *

import os
from datasets import load_dataset, concatenate_datasets, load_from_disk

cache_dir = './hugging_hub_cache'
cpu_count = os.cpu_count()


def preprocess_function(data):
    keys = data.formatter.features.keys()
    data = [
        ''.join(f"{key}\n{data[key][i]}\nend\n" for key in keys)
        for i in range(len(list(data.values())[0]))
    ]
    return {'text': data}


def download_base_dataset(keep_in_memory=False):
    load_dataset_config = {
        'cache_dir': cache_dir,
        'keep_in_memory': keep_in_memory,
        'num_proc': cpu_count,
    }
    # load dataset
    # red_1t = load_dataset(
    #     "togethercomputer/RedPajama-Data-1T",
    #     "default",
    #     **load_dataset_config,
    # )
    red_sample = load_dataset(
        'togethercomputer/RedPajama-Data-1T-Sample',
        split="train",
        **load_dataset_config,
    )

    wikipedia_cn = load_dataset(
        'pleisto/wikipedia-cn-20230720-filtered',
        split="train",
        **load_dataset_config,
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

    dataset_config = [
        # red_1t,
        red_sample,
        wikipedia_simple,
        wikipedia_en,
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
    return concatenate_datasets(dataset_config)


def download_cn_dataset(keep_in_memory=False):
    load_dataset_config = {
        'cache_dir': cache_dir,
        'keep_in_memory': keep_in_memory,
        'num_proc': cpu_count,
    }

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
    return concatenate_datasets(dataset_config)


def preprocess_function_emrgnt_cmplxty_sciphi_textbooks_are_all_you_need(data):
    return {
        'text': [
            '\n'.join([question, answer])
            for question, answer in zip(data['formatted_prompt'], data['completion'])
        ]
    }


def download_textbook_dataset(keep_in_memory=False):
    load_dataset_config = {
        'cache_dir': cache_dir,
        'keep_in_memory': keep_in_memory,
        'num_proc': cpu_count,
    }
    dataset_list = []
    # load dataset

    ##
    dataset = load_dataset(
        'emrgnt-cmplxty/sciphi-textbooks-are-all-you-need',
        split="train",
        **load_dataset_config,
    )
    dataset = dataset.map(
        preprocess_function_emrgnt_cmplxty_sciphi_textbooks_are_all_you_need,
        batched=True,
        num_proc=cpu_count,
        keep_in_memory=keep_in_memory,
        remove_columns=dataset.column_names,
    )
    dataset_list.append(dataset)
    ##

    # save to file
    return concatenate_datasets(dataset_list)




def generate_main_dataset(keep_in_memory=False):
    en_dataset = download_base_dataset(keep_in_memory=keep_in_memory)
    cn_dataset = download_cn_dataset(keep_in_memory=keep_in_memory)
    textbook_dataset = download_textbook_dataset(keep_in_memory=keep_in_memory)
    dataset = concatenate_datasets([en_dataset, cn_dataset, cn_dataset, textbook_dataset]).shuffle()
    print('dataset[0]', dataset[0])
    # shuffle dataset
    dataset = dataset.shuffle(1024 * 1024 * 16)
    print('dataset[0]', dataset[0])

    # write to file
    dataset.save_to_disk('./dataset/main')


def generate_textbook_dataset(keep_in_memory=False):
    # save to file
    dataset = download_textbook_dataset(keep_in_memory=keep_in_memory)
    dataset.save_to_disk('./dataset/textbook')


def get_main_dataset(keep_in_memory=False):
    return load_from_disk('./dataset/main', keep_in_memory=keep_in_memory)


def get_textbook_dataset(keep_in_memory=False):
    return load_from_disk('./dataset/textbook', keep_in_memory=keep_in_memory)

def get_custom_answer_dataset():
    # load dataset
    with open('./custom_data/answer.jsonl', 'r', encoding='utf-8') as f:
        data_list = [{'text': i} for i in f.read().splitlines() if i.strip() != '']
    dataset = datasets.Dataset.from_list(data_list)
    return dataset

if __name__ == '__main__':
    generate_full_dataset()
