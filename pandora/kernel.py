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

from dataproc.post_processe import get_train_dataset
from pandora.tokenizer_ import get_tokenizer
from pandora.CBOW import get_embedding
from pandora.utils import WarmupScheduler, TrainCtx
from aidevkit.component import GradientLayer


class ResNet(nn.Module):
    def __init__(self, dim, block_num, block_layer_num, bias=True, make_activation_function=lambda: nn.ReLU()):
        super(ResNet, self).__init__()
        self.blocks = [
            GradientLayer(dim, block_layer_num, dim, bias=bias, make_activation_function=make_activation_function)
            for _ in range(block_num)
        ]
        for idx, i in enumerate(self.blocks):
            self.add_module(f"{idx}", i)

    def forward(self, x):
        first_two_x = x
        first_one_x = self.blocks[0](x)
        for block in self.blocks[1:]:
            tmp_x = first_one_x
            first_one_x = block(first_one_x + first_two_x)
            first_two_x = tmp_x
        return x


class SelfAttention(nn.Module):
    """一个单头的自注意力层"""

    def __init__(self, embed_size, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.dropout = dropout

        self.q = nn.Linear(embed_size, embed_size, bias=False)
        self.k = nn.Linear(embed_size, embed_size, bias=False)
        self.v = nn.Linear(embed_size, embed_size, bias=False)

        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)

    def forward(self, x, attn_mask=None, pos_embed=None):
        """
        :param x: [seq_len, batch_size, embed_size]
        :param attn_mask: [seq_len, seq_len]
        :param pos_embed: [seq_len, batch_size, embed_size]
        :return:
        """
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = self.layer_norm1(q + x) + pos_embed
        k = self.layer_norm1(k + x) + pos_embed
        v = self.layer_norm1(v + x)

        # [seq_len, batch_size, embed_size] * [seq_len, batch_size, embed_size] -> [seq_len, batch_size, seq_len]
        attn = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

        return self.layer_norm2(attn + x)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads

        self.self_attention_list = nn.ModuleList([
            SelfAttention(embed_size, dropout) for _ in range(heads)
        ])

        self.fc = nn.Linear(embed_size * heads, embed_size, bias=False)
        self.layer_norm = nn.LayerNorm(embed_size)

    def forward(self, x, mask=None, pos_embed=None):
        """
        :param x: [seq_len, batch_size, embed_size]
        :param mask: [seq_len, seq_len]
        :param pos_embed: [seq_len, batch_size, embed_size]
        :return:
        """
        out = torch.cat([i(x, mask, pos_embed) for i in self.self_attention_list], dim=-1)
        out = self.fc(out)
        return self.layer_norm(out + x)


class FeedForward(nn.Module):
    def __init__(self, embed_size, feedforward_dim):
        super(FeedForward, self).__init__()
        self.embed_size = embed_size
        self.feedforward_dim = feedforward_dim

        self.fc1 = nn.Linear(embed_size, feedforward_dim, bias=False)
        self.silu = nn.SiLU()
        self.fc2 = nn.Linear(feedforward_dim, embed_size, bias=False)

        self.layer_norm = nn.LayerNorm(embed_size)

    def forward(self, x):
        """
        :param x: [seq_len, batch_size, embed_size]
        :return:
        """
        out = self.fc1(x)
        out = self.silu(out)
        out = self.fc2(out)
        return self.layer_norm(out + x)


class GPTBlock(nn.Module):
    def __init__(self, feature_dim, num_heads, feedforward_dim, dropout=0.1):
        super(GPTBlock, self).__init__()
        self.self_attention = MultiHeadAttention(feature_dim, num_heads, dropout)

        self.feed_forward = FeedForward(feature_dim, feedforward_dim)

    def forward(self, x, attn_mask=None, pos_embed=None):
        x = self.self_attention(x, attn_mask, pos_embed)
        x = self.feed_forward(x)
        return x


# MM Model
class MMModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_size,
                 feedforward_dim,
                 num_layers,
                 num_heads,
                 dropout=0.1,
                 ):
        super(MMModel, self).__init__()
        self.blocks = nn.ModuleList([
            GPTBlock(embed_size, num_heads, feedforward_dim, dropout) for _ in range(num_layers)
        ])
        # self.dropout = nn.Dropout(dropout)
        # self.res_net = ResNet(embed_size, res_net_block_num, res_net_block_layer_num, bias=False)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x, mask=None):
        """

        :param x:  [seq_len, N, embed_size]
        :param mask: [seq_len, seq_len]
        :return:
        """
        with torch.no_grad():
            pos_embed = rope_position_encoding(x.shape[1], x.shape[-1], x.device, theta=10000.0).unsqueeze(0)
        for block in self.blocks:
            x = block(x, mask, pos_embed)
        x = x + pos_embed
        out = self.fc_out(x)
        return out


def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    """使用llama的函数式ROPE实现取代原来的RotateEmbedding的实现"""
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [seq_len, dim // 2]
    freqs = torch.outer(t, freqs).float()
    # torch.polar 的文档
    # https://pytorch.org/docs/stable/generated/torch.polar.html
    # 计算结果是个复数向量
    # 假设 freqs = [x, y]
    # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


@functools.lru_cache(maxsize=3)
def rope_position_encoding(seq_len, embed_size, device, theta=10000.0):
    """
    生成旋转位置编码
    :param seq_len: 序列长度
    :param embed_size: 嵌入维度
    :param theta: 旋转位置编码参数
    :return: 旋转位置编码张量
    """
    # # make tensor
    # t = torch.zeros(seq_len, embed_size)
    #
    # # init tensor
    # for i in range(seq_len):
    #     for j in range(0, embed_size, 2):
    #         t[i, j] = math.sin(i / (theta ** (j / embed_size)))
    #         t[i, j + 1] = math.cos(i / (theta ** (j / embed_size)))
    #
    # return t

    # a speed up version

    # sin_off_vec = [0, 0.5pi, 0, 0.5pi, 0, 0.5pi, ...]
    # 这个向量用于偏移 sin 函数， 令其的效果为 cos 函数
    sin_off_vec = torch.zeros(embed_size).to(device)
    sin_off_vec[torch.cos(torch.arange(0, embed_size).to(device) * torch.pi) < 0] = torch.pi * 0.5

    # base_vec = (theta ** (j / embed_size)
    # 这个向量用于计算旋转位置编码的基础值
    base_vec = theta ** (torch.arange(0, embed_size, 2).to(device).float() / embed_size)
    base_vec = torch.stack([base_vec, base_vec], dim=-1).reshape(-1)

    idx = torch.arange(seq_len).to(device)

    return torch.sin(idx.unsqueeze(-1) / base_vec + sin_off_vec)


# 生成旋转矩阵
def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度\theta_i
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [seq_len, dim // 2]
    freqs = torch.outer(t, freqs).float()  # 计算m * \theta

    # 计算结果是个复数向量
    # 假设 freqs = [x, y]
    # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


# 旋转位置编码计算
def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq.shape = [batch_size, seq_len, dim]
    # xq_.shape = [batch_size, seq_len, dim // 2, 2]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)

    # 转为复数域
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)

    # 应用旋转操作，然后将结果转回实数域
    # xq_out.shape = [batch_size, seq_len, dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def get_mean_token_size():
    tokenizer = get_tokenizer()
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
                    accelerator.save_state(os.path.join(train_dir, 'model_backup'))
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




