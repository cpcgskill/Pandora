# -*-coding:utf-8 -*-
"""
:创建时间: 2023/9/29 22:46
:作者: 苍之幻灵
:我的主页: https://cpcgskill.com
:Github: https://github.com/cpcgskill
:QQ: 2921251087
:aboutcg: https://www.aboutcg.org/teacher/54335
:bilibili: https://space.bilibili.com/351598127
:爱发电: https://afdian.net/@Phantom_of_the_Cang

"""
from __future__ import unicode_literals, print_function, division

import functools
import json
import sys
from typing import *

import math
import os

import tokenizers
import torch
import torch.nn as nn

import tqdm

import accelerate
from accelerate.local_sgd import LocalSGD

from pandora.tokenizer_ import get_tokenizer
from pandora.SkipGram import get_embedding
from pandora.utils import WarmupScheduler, TrainCtx

from pandora.modules import MMModel


def new_model(config, tokenizer):
    # type: (config.Config, tokenizers.Tokenizer) -> MMModel
    return MMModel(
        tokenizer.get_vocab_size(),
        config.embed_size,
        config.feedforward_dim,
        config.num_layers,
        config.num_heads,
        config.dropout,
    )


def generate_mask(seq_len, device):
    """
    mask:
    [
     [0, -inf, -inf, ..., -inf, -inf, -inf],
     [0, 0, -inf, ..., -inf, -inf, -inf],
     [0, 0, 0, ...,    -inf, -inf, -inf],
     ...
     [0, 0,    0, ...,    0,    0,    0]
    ]
    ps: right mask need hide now token

    :param seq_len:
    :param device:
    :return:
    """
    return nn.Transformer.generate_square_subsequent_mask(seq_len, device)


def train_transformer(config, dataset, train_dir):
    # type: (config.Config, datasets.Dataset, str) -> None
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps)

    tokenizer = get_tokenizer(config)

    transformer = new_model(config, tokenizer)
    train_ctx = TrainCtx()

    optimizer = torch.optim.Adam(transformer.parameters(),
                                 lr=config.init_lr,
                                 )
    scheduler = WarmupScheduler(optimizer,
                                config.warmup_epochs / config.gradient_accumulation_steps,
                                init_lr=config.init_lr,
                                max_lr=config.max_lr,
                                gamma=config.gamma,
                                )
    loss_function = nn.CrossEntropyLoss(ignore_index=4)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
        collate_fn=lambda x: [i['ids'] for i in x],
    )

    transformer, optimizer, scheduler, loss_function = accelerator.prepare(
        transformer, optimizer, scheduler, loss_function
    )

    train_ctx = accelerator.prepare(train_ctx)
    accelerator.register_for_checkpointing(train_ctx)

    data_loader = accelerator.prepare(data_loader)

    if os.path.isdir(os.path.join(train_dir, 'model')):
        accelerator.print("load model")
        accelerator.load_state(os.path.join(train_dir, 'model'))

    # print model info
    accelerator.print(
        f"model structure: {transformer}",
        f"model parameters: {sum(p.numel() for p in transformer.parameters()) / (10 ** 9)}",
        f"optimizer: {optimizer}",
        f"scheduler: {scheduler}",
        f"loss_function: {loss_function}",
        sep='\n'
    )
    # prepare training
    transformer.train()
    # transformer.compute()
    optimizer.zero_grad()

    # make embedding
    embedding = get_embedding(config, tokenizer).to(accelerator.device)
    embedding.eval()

    with LocalSGD(accelerator=accelerator,
                  model=transformer,
                  local_sgd_steps=config.gradient_accumulation_steps * 16,
                  enabled=True) as local_sgd:
        for ids_data in tqdm.tqdm(data_loader, disable=not accelerator.is_local_main_process):
            train_ctx.step += 1
            with accelerator.accumulate(transformer):
                # process data
                with torch.no_grad():
                    label = torch.tensor(ids_data).to(accelerator.device)
                    data = embedding(label)
                    # data = data + torch.randn_like(data) * 0.01

                    mask = generate_mask(data.size(1), accelerator.device)
                if accelerator.is_main_process and accelerator.num_processes > 1:
                    prediction_length = config.main_prediction_length
                else:
                    prediction_length = config.prediction_length
                for i in range(-prediction_length, 0):
                    with torch.no_grad():
                        sub_data = data[:, :data.size(1) + i]
                        sub_mask = mask[:mask.size(0) + i, :mask.size(0) + i]
                    sub_output = transformer(sub_data, mask=sub_mask)
                sub_label = label[:, 1:label.size(1) + i + 1]
                loss = loss_function(sub_output.view(-1, tokenizer.get_vocab_size()), sub_label.view(-1))
                if accelerator.is_main_process:
                    train_ctx.loss_list.append(loss.item())
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                local_sgd.step()

                if not accelerator.is_main_process:
                    continue
                # in main process
                if (train_ctx.step + 1) % 1000 == 0:
                    accelerator.save_state(os.path.join(train_dir, 'model'))
                    train_ctx.export_loss_data(train_dir)
                    with open(os.path.join(train_dir, 'config.json'), 'w') as f:
                        json.dump(
                            config.to_dict(), f,
                            indent=4, ensure_ascii=False,
                        )

    if not accelerator.is_main_process:
        return
    # in main process
    accelerator.save_state(os.path.join(train_dir, 'model_backup'))
    accelerator.save_state(os.path.join(train_dir, 'model'))
    train_ctx.export_loss_data(train_dir)
    with open(os.path.join(train_dir, 'config.json'), 'w') as f:
        json.dump(
            config.to_dict(), f,
            indent=4, ensure_ascii=False,
        )


def check_transformer(config, input_str, train_dir='./data/transformer'):
    # type: (config.Config, str, str) -> None
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps)

    tokenizer = get_tokenizer(config)

    transformer = new_model(config, tokenizer)
    train_ctx = TrainCtx()

    optimizer = torch.optim.Adam(transformer.parameters(),
                                 lr=config.init_lr,
                                 )
    scheduler = WarmupScheduler(optimizer,
                                config.warmup_epochs / config.gradient_accumulation_steps,
                                init_lr=config.init_lr,
                                max_lr=config.max_lr,
                                gamma=config.gamma,
                                )
    loss_function = nn.CrossEntropyLoss(ignore_index=3)

    transformer, optimizer, scheduler, loss_function = accelerator.prepare(
        transformer, optimizer, scheduler, loss_function
    )

    train_ctx = accelerator.prepare(train_ctx)
    accelerator.register_for_checkpointing(train_ctx)

    if os.path.isdir(os.path.join(train_dir, 'model')):
        accelerator.print("load model")
        accelerator.load_state(os.path.join(train_dir, 'model'))

    # print model info
    accelerator.print(
        f"model structure: {transformer}",
        f"model parameters: {sum(p.numel() for p in transformer.parameters())}",
        f"optimizer: {optimizer}",
        f"scheduler: {scheduler}",
        f"loss_function: {loss_function}",
        sep='\n'
    )

    # prepare training
    transformer.eval()

    # make embedding
    embedding = get_embedding(config, tokenizer).to(accelerator.device)
    embedding.eval()

    sys.stdout.write(input_str)
    # process data
    with torch.no_grad():
        split_data = tokenizer.encode_batch([input_str])
        label = torch.tensor([i.ids for i in split_data]).to(accelerator.device)
        label = label[:, :-1]
        for _ in range(4096):
            data = embedding(label)

            mask = generate_mask(data.size(0), accelerator.device)

            output = transformer(data, mask=mask)

            top_probs, top_idx = torch.topk(output, k=2, dim=-1)
            softmaxed_output = torch.softmax(top_probs, dim=-1).reshape(top_probs.shape[0] * top_probs.shape[1], -1)

            result = torch.multinomial(softmaxed_output, num_samples=1).reshape(top_probs.shape[0], top_probs.shape[1])
            result = top_idx.gather(dim=-1, index=result.reshape(result.shape[0], result.shape[1], 1))
            result = result.reshape(top_idx.shape[0], top_idx.shape[1])

            # output text
            old_str = tokenizer.decode_batch(label.tolist())[0]
            # add next token mask
            label = torch.cat([label, result[:, -1:]], dim=-1)
            #
            new_str = tokenizer.decode_batch(label.tolist())[0]
            sys.stdout.write(new_str[len(old_str):])
            if tokenizer.decode_batch(label.tolist())[0].rfind('[EOS]') != -1:
                break
