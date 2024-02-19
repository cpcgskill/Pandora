# -*-coding:utf-8 -*-
"""
:PROJECT_NAME: ret_t
:File: modules.py
:Time: 2024/2/19 17:04
:Author: 张隆鑫
"""
from __future__ import unicode_literals, print_function, division

from typing import *

import functools

import torch
import torch.nn as nn
from aidevkit.component import GradientLayer


class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(CBOW, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.linear = nn.Linear(embed_size, vocab_size)

    def forward(self, context):
        # if is train mode
        if self.training:
            embeds = self.embed(context)
            embeds_sum = torch.sum(embeds, dim=1)
            out = self.linear(embeds_sum)
            return out
        else:
            # if is eval mode
            embeds = self.embed(context)
            return embeds

    def normal_embedding(self):
        with torch.no_grad():
            # normalize embeddings. ps: to 1
            self.embed.weight.div_(self.embed.weight.norm(dim=1, keepdim=True))
            self.linear.weight.div_(self.linear.weight.norm(dim=1, keepdim=True))
        return self


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

    def normal_embedding(self):
        with torch.no_grad():
            # normalize embeddings. ps: to 1
            self.in_embed.weight.div_(self.in_embed.weight.norm(dim=1, keepdim=True))
            self.out_embed.weight.div_(self.out_embed.weight.norm(dim=1, keepdim=True))
        return self


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
