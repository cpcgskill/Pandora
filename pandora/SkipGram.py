# -*-coding:utf-8 -*-
"""
:创建时间: 2023/9/26 3:20
:作者: 苍之幻灵
:我的主页: https://cpcgskill.com
:Github: https://github.com/cpcgskill
:QQ: 2921251087
:aboutcg: https://www.aboutcg.org/teacher/54335
:bilibili: https://space.bilibili.com/351598127
:爱发电: https://afdian.net/@Phantom_of_the_Cang

"""
from __future__ import unicode_literals, print_function, division

import os

import torch
import torch.nn as nn
import tqdm

import accelerate
from accelerate.local_sgd import LocalSGD

from pandora.utils import WarmupScheduler, TrainCtx
from pandora.tokenizer_ import get_tokenizer

from pandora.modules import SkipGram


def make_skip_gram(config, tokenizer):
    return SkipGram(tokenizer.get_vocab_size(), config.embed_size).normal_embedding()


def train_skip_gram(config, dataset, train_dir):
    # accelerate
    accelerator = accelerate.Accelerator()
    print('device:', accelerator.device)

    # make
    tokenizer = get_tokenizer(config)
    model = make_skip_gram(config, tokenizer)
    # train
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        # shuffle=True,
        pin_memory=True,
    )
    optimizer = torch.optim.Adagrad(model.parameters(), lr=1.0 / config.embed_size, weight_decay=0.01)
    scheduler = WarmupScheduler(optimizer,
                                warmup_epochs=240,
                                init_lr=1.0 / config.embed_size / 100,
                                max_lr=1.0 / config.embed_size,
                                gamma=0.98
                                )
    loss_function = nn.CrossEntropyLoss()
    train_ctx = TrainCtx()

    model, optimizer, scheduler, loss_function = accelerator.prepare(model, optimizer, scheduler, loss_function)

    train_ctx = accelerator.prepare(train_ctx)
    accelerator.register_for_checkpointing(train_ctx)

    data_loader = accelerator.prepare(data_loader)

    if os.path.isdir(os.path.join(train_dir, 'model')):
        accelerator.print("load model")
        accelerator.load_state(os.path.join(train_dir, 'model'))

    model.train()

    #  启用填充，最大长度为1024， 对于长度不足1024的序列，用3填充。 对于长度超过1024的序列，进行截断
    tokenizer.enable_padding(length=1024, pad_id=3, pad_token="[PAD]")
    #  启用截断，最大长度为1024
    tokenizer.enable_truncation(max_length=1024)
    with LocalSGD(accelerator=accelerator, model=model, local_sgd_steps=8, enabled=True) as local_sgd:
        for epoch in range(2):
            for text_data in tqdm.tqdm(data_loader, disable=not accelerator.is_local_main_process):
                train_ctx.step += 1

                tokens = tokenizer.encode_batch(text_data["text"])
                token_ids = [i.ids for i in tokens]
                token_ids = torch.Tensor(token_ids).long()
                token_ids = token_ids.to(accelerator.device)

                optimizer.zero_grad()
                for i in range(token_ids.shape[1] - 1):
                    target_idx = token_ids[:, i]
                    context_idx = token_ids[:, i + 1]
                    output = model(target_idx)
                    loss = loss_function(output, context_idx)
                    accelerator.backward(loss / (token_ids.shape[1] - 1))

                optimizer.step()
                scheduler.step()
                local_sgd.step()
                train_ctx.loss_list.append(loss.item())

                if train_ctx.step % 8 == 0:
                    if accelerator.num_processes > 1:
                        model.module.normal_embedding()
                    else:
                        model.normal_embedding()

                if not accelerator.is_main_process:
                    continue

                if train_ctx.step % 12 == 0:
                    train_ctx.export_loss_data(train_dir)
                    if min(train_ctx.loss_list[:-1]) > loss.item():
                        tqdm.tqdm.write(f"save model")
                        accelerator.save_state(os.path.join(train_dir, 'best_model'))
                    if (train_ctx.loss_list[-1] - train_ctx.loss_list[-2]) > \
                            abs(train_ctx.loss_list[-2] - train_ctx.loss_list[-3]):
                        tqdm.tqdm.write(f"backtrace model")
                        accelerator.load_state(os.path.join(train_dir, 'best_model'))
                        # save loss graph
                        with open(os.path.join(train_dir, 'loss.png'), 'rb') as f:
                            with open(os.path.join(train_dir, f'backtrace_loss_{len(train_ctx.loss_list)}.png'),
                                      'wb') as f2:
                                f2.write(f.read())
                # if train_ctx.step % 120 == 0:
                #     # save skip_gram
                #     accelerator.save_state(os.path.join(train_dir, 'model_backup'))
                #     accelerator.save_state(os.path.join(train_dir, 'model'))


def test_skip_gram(train_dir):
    # accelerate
    accelerator = accelerate.Accelerator()
    print('device:', accelerator.device)

    # make embedding
    tokenizer = get_tokenizer()
    skip_gram = make_skip_gram(tokenizer)
    # train embedding
    dataset = get_base_dataset(keep_in_memory=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2048,
        shuffle=True,
        pin_memory=True,
    )
    optimizer = torch.optim.Adagrad(skip_gram.parameters(), lr=1.0 / config.embed_size, weight_decay=0.01)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    scheduler = WarmupScheduler(optimizer,
                                warmup_epochs=10,
                                init_lr=1.0 / config.embed_size / 10,
                                max_lr=1.0 / config.embed_size,
                                gamma=0.97
                                )
    loss_function = nn.CrossEntropyLoss()
    train_ctx = TrainCtx()

    skip_gram, optimizer, scheduler, loss_function, train_ctx = accelerator.prepare(
        data_loader, skip_gram, optimizer, scheduler, loss_function, train_ctx
    )
    accelerator.register_for_checkpointing(train_ctx)

    if os.path.isdir(os.path.join(train_dir, 'model')):
        accelerator.print("load model")
        accelerator.load_state(os.path.join(train_dir, 'model'))

    # test skip_gram
    print("test skip_gram")
    skip_gram.eval()
    text = "Hello Hi Bad 你好 嗨 How"
    tokens = tokenizer.encode(text)
    print(tokens.tokens)
    print(tokens.ids)
    tokens = torch.Tensor([tokens.ids]).long()
    output = skip_gram(tokens.to(accelerator.device))
    print(output.shape)
    print(output)
    # num dot
    for i in range(2, output.shape[1]):
        print(output[0, 1].dot(output[0, i]))


build_pt_path = './model/embedding.pt'


def build_embedding_from_skip_gram(config, train_dir):
    # accelerate
    accelerator = accelerate.Accelerator()
    print('device:', accelerator.device)

    # make
    tokenizer = get_tokenizer(config)
    model = make_skip_gram(config, tokenizer)
    # train
    optimizer = torch.optim.Adagrad(model.parameters(), lr=1.0 / config.embed_size, weight_decay=0.01)
    scheduler = WarmupScheduler(optimizer,
                                warmup_epochs=30,
                                init_lr=1.0 / config.embed_size / 10,
                                max_lr=1.0 / config.embed_size,
                                gamma=0.97
                                )
    loss_function = nn.CrossEntropyLoss()
    train_ctx = TrainCtx()

    model, optimizer, scheduler, loss_function = accelerator.prepare(model, optimizer, scheduler, loss_function)

    train_ctx = accelerator.prepare(train_ctx)
    accelerator.register_for_checkpointing(train_ctx)

    if os.path.isdir(os.path.join(train_dir, 'model')):
        accelerator.print("load model")
        accelerator.load_state(os.path.join(train_dir, 'model'))

    # make embedding
    if not os.path.isdir(os.path.dirname(build_pt_path)):
        os.makedirs(os.path.dirname(build_pt_path))
    torch.save(model.in_embed.state_dict(), build_pt_path)


def make_embedding(config, tokenizer):
    # make embedding
    return nn.Embedding(tokenizer.get_vocab_size(), config.embed_size)


def get_embedding(config, tokenizer):
    # load embedding
    embedding = make_embedding(config, tokenizer)
    # state_dict = accelerator.load_state("/root/autodl-fs/skip_gram.in_embed.pth")
    state_dict = torch.load(build_pt_path)
    embedding.load_state_dict(state_dict)
    return embedding


def test_embedding():
    tokenizer = get_tokenizer()
    embedding = get_embedding(tokenizer)
    # test skip_gram
    print("test skip_gram")
    embedding.eval()
    text = "Hello Hi Bad 你好"
    tokens = tokenizer.encode(text)
    print(tokens.tokens)
    print(tokens.ids)
    tokens = torch.Tensor([tokens.ids]).long()
    output = embedding(tokens)
    print(output.shape)
    print(output)
    # num dot
    for i in range(2, output.shape[1]):
        print(output[0, 1].dot(output[0, i]))
