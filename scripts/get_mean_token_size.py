# -*-coding:utf-8 -*-
"""
:PROJECT_NAME: ret_t
:File: get_mean_token_size.py
:Time: 2024/2/19 18:52
:Author: 张隆鑫
"""
from __future__ import unicode_literals, print_function, division

if False:
    from typing import *
import torch
from dataproc.post_processe import get_train_dataset
from pandora.tokenizer_ import get_tokenizer
from pandora.config import Config


def get_mean_token_size():
    tokenizer = get_tokenizer(Config())
    dataset = get_train_dataset(keep_in_memory=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1024 * 800,
        shuffle=True,
        pin_memory=True,
    )
    text_data = next(iter(data_loader))

    split_data = tokenizer.encode_batch(text_data['text'])
    mean_size = torch.mean(torch.tensor([float(len(i.ids)) for i in split_data]))
    print('dataset mean size is ', mean_size)
    return int(mean_size.item())


if __name__ == '__main__':
    get_mean_token_size()
