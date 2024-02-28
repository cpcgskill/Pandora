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

# from pandora.data.compile_dataset import get_main_dataset
# from pandora.tokenizer_ import train_tokenizer
# train_tokenizer(get_main_dataset(keep_in_memory=True))

# from pandora.data.post_processe import generate_train_and_test_dataset, generate_pretokenize_dataset
# generate_pretokenize_dataset()
# generate_train_and_test_dataset(chunk_size=512)

from dataproc.compile_dataset import get_custom_answer_dataset, get_main_dataset, get_custom_new_answer_dataset, \
    get_merge_custom_answer_dataset
from dataproc.post_processe import *
from pandora.tokenizer_ import get_tokenizer
from pandora.config import Config
from dataproc.utils import dataset_cache

config = Config()


@dataset_cache
def generate_pretokenize_dataset(dataset, keep_in_memory=False, chunk_size=512, seam=5):
    dataset = pretokenize_dataset(get_tokenizer(config), dataset, keep_in_memory=keep_in_memory)
    return dataset


@dataset_cache
def generate_train_and_test_dataset(dataset, keep_in_memory=False, chunk_size=512, seam=5):
    dataset = pretokenize_dataset(get_tokenizer(config), dataset, keep_in_memory=keep_in_memory)
    dataset = segmentation_dataset(dataset, chunk_size=chunk_size)
    train_dataset, test_dataset = split_dataset(dataset, keep_in_memory=keep_in_memory)
    return train_dataset, test_dataset


@dataset_cache
def generate_extract_tail_dataset(dataset, keep_in_memory=False, chunk_size=512):
    dataset = pretokenize_dataset(get_tokenizer(config), dataset, keep_in_memory=keep_in_memory)
    dataset = extract_tail_dataset(dataset, chunk_size=chunk_size)
    train_dataset, test_dataset = split_dataset(dataset, keep_in_memory=keep_in_memory)
    return train_dataset, test_dataset


source_dataset = get_merge_custom_answer_dataset()

pr_dataset = generate_pretokenize_dataset(source_dataset)
train_dataset, test_dataset = generate_extract_tail_dataset(source_dataset, chunk_size=512)


def start():
    # from pandora.CBOW import train_embedding, build_embedding
    # train_embedding(get_main_dataset(keep_in_memory=True), './data/embedding2')
    # build_embedding('./data/embedding2', '/root/autodl-fs/embedding.pt')

    from pandora.kernel import train_transformer, check_transformer
    for i in range(1):
        train_transformer(
            config,
            train_dataset,
            './model/transformer',
        )

    check_transformer(
        config,
        r'''[I'm]
一名奉行实用主义的结果导向型导师，总是简单直接的给出可执行且可测试的回答。研究领域为：生物信息学。
[SEP]
[Input]
如何整合和分析不同生物信息学数据库中的异构数据？
[SEP]
[Output]
''',
        './model/transformer'
    )


import accelerate

num_processes = 1
if num_processes > 1:
    accelerate.notebook_launcher(start, (), num_processes=num_processes, mixed_precision='fp16')
else:
    start()
