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

from pandora import config
from pandora.compile_dataset import get_base_dataset
from utils import WarmupScheduler, save_loss_list_graph
from tokenizer_ import get_tokenizer


# 定义SkipGram模型
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(SkipGram, self).__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_size)
        self.out_embed = nn.Embedding(vocab_size, embed_size)

    def forward(self, target):
        if self.training:
            in_embeds = self.in_embed(target)
            scores = torch.matmul(in_embeds, self.out_embed.weight.t())
            return scores
        else:
            in_embeds = self.in_embed(target)
            return in_embeds


class TrainCtx:
    def __init__(self):
        self.step = 0
        self.loss_list = []

    def state_dict(self):
        return {
            'global_step': self.step,
            'loss_list': self.loss_list,
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict['global_step']
        self.loss_list = state_dict['loss_list']


def make_skip_gram(tokenizer):
    # make embedding
    skip_gram = SkipGram(tokenizer.get_vocab_size(), config.module['embed_size'])
    with torch.no_grad():
        # normalize embeddings. ps: to 1
        skip_gram.in_embed.weight.div_(skip_gram.in_embed.weight.norm(dim=1, keepdim=True))
    return skip_gram


def train_skip_gram():
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
    optimizer = torch.optim.Adagrad(skip_gram.parameters(), lr=1.0 / config.module['embed_size'], weight_decay=0.01)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    scheduler = WarmupScheduler(optimizer,
                                warmup_epochs=10,
                                init_lr=1.0 / config.module['embed_size'] / 10,
                                max_lr=1.0 / config.module['embed_size'],
                                gamma=0.97
                                )
    loss_function = nn.CrossEntropyLoss()
    train_ctx = TrainCtx()

    data_loader, skip_gram, optimizer, scheduler, loss_function, train_ctx = accelerator.prepare(
        data_loader, skip_gram, optimizer, scheduler, loss_function, train_ctx
    )
    accelerator.register_for_checkpointing(train_ctx)

    if os.path.isdir('./data/skip_gram'):
        accelerator.print("load skip_gram")
        accelerator.load_state('./data/skip_gram')

    skip_gram.train()

    #  启用填充，最大长度为1024， 对于长度不足1024的序列，用3填充。 对于长度超过1024的序列，进行截断
    tokenizer.enable_padding(length=1024, pad_id=3, pad_token="[PAD]")
    #  启用截断，最大长度为1024
    tokenizer.enable_truncation(max_length=1024)
    with LocalSGD(accelerator=accelerator, model=skip_gram, local_sgd_steps=8, enabled=True) as local_sgd:
        for epoch in range(99):
            for text_data in tqdm.tqdm(data_loader, disable=not accelerator.is_local_main_process):
                train_ctx.step += 1

                tokens = tokenizer.encode_batch(text_data["text"])
                token_ids = [i.ids for i in tokens]
                token_ids = torch.Tensor(token_ids).long()
                token_ids = token_ids.to(accelerator.device)

                optimizer.zero_grad()
                # 向前向后各取2个词，共4个词作为上下文
                for i in range(token_ids.shape[1] - 1):
                    target_idx = token_ids[:, i]
                    context_idx = token_ids[:, i + 1]
                    output = skip_gram(target_idx)
                    loss = loss_function(output, context_idx)
                    accelerator.backward(loss / (token_ids.shape[1] - 1))
                optimizer.step()
                scheduler.step()
                local_sgd.step()
                train_ctx.loss_list.append(loss.item())

                if train_ctx.step % 10 == 0:
                    with torch.no_grad():
                        # normalize embeddings. ps: to 1
                        skip_gram.in_embed.weight.div_(skip_gram.in_embed.weight.norm(dim=1, keepdim=True))

                if not accelerator.is_main_process:
                    continue

                if train_ctx.step % 30 == 0:
                    # save skip_gram
                    tqdm.tqdm.write(f"save skip_gram: {epoch}")
                    accelerator.save_state('./data/skip_gram')
                    accelerator.save_state(f"./data/skip_gram_{train_ctx.step}_{loss.item():.2f}")
                    save_loss_list_graph(train_ctx.loss_list, "./data/skip_gram.png")


def test_skip_gram():
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
    optimizer = torch.optim.Adagrad(skip_gram.parameters(), lr=1.0 / config.module['embed_size'], weight_decay=0.01)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    scheduler = WarmupScheduler(optimizer,
                                warmup_epochs=10,
                                init_lr=1.0 / config.module['embed_size'] / 10,
                                max_lr=1.0 / config.module['embed_size'],
                                gamma=0.97
                                )
    loss_function = nn.CrossEntropyLoss()
    train_ctx = TrainCtx()

    data_loader, skip_gram, optimizer, scheduler, loss_function, train_ctx = accelerator.prepare(
        data_loader, skip_gram, optimizer, scheduler, loss_function, train_ctx
    )
    accelerator.register_for_checkpointing(train_ctx)

    if os.path.isdir('./data/skip_gram'):
        accelerator.print("load skip_gram")
        accelerator.load_state('./data/skip_gram')
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


def build_embedding_from_skip_gram():
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
    optimizer = torch.optim.Adagrad(skip_gram.parameters(), lr=1.0 / config.module['embed_size'], weight_decay=0.01)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    scheduler = WarmupScheduler(optimizer,
                                warmup_epochs=10,
                                init_lr=1.0 / config.module['embed_size'] / 10,
                                max_lr=1.0 / config.module['embed_size'],
                                gamma=0.97
                                )
    loss_function = nn.CrossEntropyLoss()
    train_ctx = TrainCtx()

    data_loader, skip_gram, optimizer, scheduler, loss_function, train_ctx = accelerator.prepare(
        data_loader, skip_gram, optimizer, scheduler, loss_function, train_ctx
    )
    accelerator.register_for_checkpointing(train_ctx)

    if os.path.isdir('./data/skip_gram'):
        accelerator.print("load skip_gram")
        accelerator.load_state('./data/skip_gram')

    # make embedding
    torch.save(skip_gram.in_embed.state_dict(), "/root/autodl-fs/skip_gram.in_embed.pt")

    print('test get embedding')
    get_embedding(get_tokenizer())


def make_embedding(tokenizer):
    # make embedding
    return nn.Embedding(tokenizer.get_vocab_size(), config.module['embed_size'])


def get_embedding(tokenizer):
    # load embedding
    embedding = make_embedding(tokenizer)
    # state_dict = accelerator.load_state("/root/autodl-fs/skip_gram.in_embed.pth")
    state_dict = torch.load("/root/autodl-fs/skip_gram.in_embed.pt")
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


if __name__ == '__main__':
    # train_tokenizer()
    # build_embedding_from_skip_gram()
    # test_skip_gram()
    # train_skip_gram()
    # accelerator = accelerate.Accelerator()
    # e = get_embedding(accelerator, get_tokenizer())
    # accelerator.save_state("/root/autodl-fs/skip_gram.in_embed")
    #
    test_embedding()
