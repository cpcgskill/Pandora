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
from typing import *

import math
import os

import tokenizers
import torch
import torch.nn as nn

import tqdm

import accelerate
from accelerate.local_sgd import LocalSGD

from post_processe import get_train_dataset
from SkipGram import get_tokenizer, get_embedding
import config
from utils import add_random_value_by_weights, WarmupScheduler, save_loss_list_graph
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


class GPTBlock(nn.Module):
    def __init__(self, feature_dim, num_heads, feedforward_dim):
        super(GPTBlock, self).__init__()
        self.self_attention = nn.MultiheadAttention(feature_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        self.layer_norm1 = nn.LayerNorm(feature_dim)
        self.layer_norm2 = nn.LayerNorm(feature_dim)

    def forward(self, x, attn_mask=None):
        attn_output, _ = self.self_attention(x, x, x, attn_mask=attn_mask)
        x = x + attn_output
        x = self.layer_norm1(x)

        ff_output = self.feed_forward(x)
        x = x + ff_output
        x = self.layer_norm2(x)

        return x


class GPT(nn.Module):
    def __init__(self, feature_dim, num_heads, feedforward_dim, num_layers):
        super(GPT, self).__init__()
        self.blocks = nn.ModuleList([
            GPTBlock(feature_dim, num_heads, feedforward_dim) for _ in range(num_layers)
        ])

    def forward(self, x, attn_mask=None):
        for block in self.blocks:
            x = block(x, attn_mask)
        return x


# MM Model
class MMModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_size,
                 num_layers,
                 heads,
                 dropout=0.1,
                 res_net_block_num=192,
                 res_net_block_layer_num=6,
                 ):
        super(MMModel, self).__init__()
        self.gpt = GPT(embed_size, heads, embed_size * 4, num_layers)
        self.dropout = nn.Dropout(dropout)
        self.res_net = ResNet(embed_size, res_net_block_num, res_net_block_layer_num, bias=False)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x, mask=None):
        """

        :param x:  [N, seq_len, embed_size]
        :param mask: [seq_len, seq_len]
        :return:
        """
        x = self.dropout(x)
        out = self.gpt(x, attn_mask=mask)
        out = self.res_net(out)
        out = self.fc_out(out)
        return out


class RotateEmbedding(nn.Module):
    """
    一个用旋转位置编码初始化的嵌入层
    并且使用旋转位置编码初始化这个张量
    """

    def __init__(self, seq_len, embed_size):
        """

        :param seq_len: 序列长度
        :param embed_size: 嵌入维度
        """
        assert embed_size % 2 == 0, "embed_size must be even"

        super(RotateEmbedding, self).__init__()
        self.seq_len = seq_len
        self.embed_size = embed_size

        # make tensor
        self.t = nn.Embedding(seq_len, embed_size)

        # init tensor
        for i in range(seq_len):
            for j in range(0, embed_size, 2):
                self.t.weight.data[i, j] = math.sin(i / (10000 ** (j / embed_size)))
                self.t.weight.data[i, j + 1] = math.cos(i / (10000 ** (j / embed_size)))

    def forward(self, x):
        """

        :param x: 输入张量
        :return: 旋转位置编码后的张量
        """
        N, seq_len, embed_size = x.shape
        assert seq_len <= self.seq_len, "seq_len must be less than RotateEmbedding.seq_len"
        assert embed_size == self.embed_size, "embed_size must be equal to RotateEmbedding.embed_size"

        idx = torch.arange(seq_len).to(x.device)

        return x + self.t(idx).unsqueeze(0).expand(N, seq_len, embed_size)


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


def new_model(tokenizer):
    return MMModel(
        tokenizer.get_vocab_size(),
        **config.module
    )


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


def init_all_object(root='./data/transformer') -> Tuple[
    accelerate.Accelerator,
    tokenizers.Tokenizer,
    MMModel,
    TrainCtx,
    torch.optim.Optimizer,
    WarmupScheduler,
    nn.CrossEntropyLoss,
    torch.utils.data.DataLoader,
]:
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=config.train['gradient_accumulation_steps'])

    tokenizer = get_tokenizer()
    dataset = get_train_dataset(keep_in_memory=True)

    transformer = new_model(tokenizer)
    train_ctx = TrainCtx()

    optimizer = torch.optim.Adam(transformer.parameters(),
                                 lr=config.train['init_lr'],
                                 )
    scheduler = WarmupScheduler(optimizer,
                                config.train['warmup_epochs'] / config.train['gradient_accumulation_steps'],
                                init_lr=config.train['init_lr'],
                                max_lr=config.train['max_lr'],
                                gamma=config.train['gamma'],
                                )
    loss_function = nn.CrossEntropyLoss(ignore_index=3)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
        collate_fn=lambda x: [i['ids'] for i in x],
    )

    data_loader, transformer, optimizer, scheduler, loss_function = accelerator.prepare(
        data_loader, transformer, optimizer, scheduler, loss_function
    )
    accelerator.register_for_checkpointing(train_ctx)

    if os.path.isdir(root):
        accelerator.print("load transformer")
        accelerator.load_state(root)

    # print model info
    accelerator.print(
        f"model structure: {transformer}",
        f"model parameters: {sum(p.numel() for p in transformer.parameters())}",
        f"optimizer: {optimizer}",
        f"scheduler: {scheduler}",
        f"loss_function: {loss_function}",
        sep='\n'
    )
    return accelerator, tokenizer, transformer, train_ctx, optimizer, scheduler, loss_function, data_loader


def train_transformer():
    accelerator, tokenizer, transformer, train_ctx, optimizer, scheduler, loss_function, data_loader = init_all_object()

    # prepare training
    transformer.train()
    # transformer.compute()
    optimizer.zero_grad()

    # #  启用填充，最大长度为1024， 对于长度不足1024的序列，用3填充。 对于长度超过1024的序列，进行截断
    # tokenizer.enable_padding(length=max_seq_len, pad_id=3, pad_token="[PAD]")

    # 使用数据集的平均长度作为最大长度， 这个长度可以通过 get_mean_token_size() 函数获取
    max_seq_len = 698
    #  启用截断
    tokenizer.enable_truncation(max_length=max_seq_len)
    # make embedding
    embedding = get_embedding(tokenizer).to(accelerator.device)
    embedding.eval()
    # 启用旋转位置编码
    position_encoding = RotateEmbedding(max_seq_len, 768).to(accelerator.device)

    with LocalSGD(accelerator=accelerator,
                  model=transformer,
                  local_sgd_steps=config.train["gradient_accumulation_steps"] * 16,
                  enabled=True) as local_sgd:
        for ids_data in tqdm.tqdm(data_loader, disable=not accelerator.is_local_main_process):
            train_ctx.step += 1
            with accelerator.accumulate(transformer):
                # process data
                with torch.no_grad():
                    # split_data = tokenizer.encode_batch(text_data['text'])
                    # label = torch.tensor([i.ids for i in split_data]).to(accelerator.device)
                    label = torch.tensor(ids_data).to(accelerator.device)
                    data = embedding(label[:, :-1])
                    data = position_encoding(data)
                    data = data + torch.randn_like(data) * 0.1

                    # shape[batch_size, seq_len, embed_size] -> [seq_len, batch_size, embed_size]
                    data = data.permute(1, 0, 2)
                    label = label[:, 1:].permute(1, 0)

                    mask = generate_mask(data.size(0), accelerator.device)

                output = transformer(data, mask=mask)
                loss = loss_function(output.view(-1, tokenizer.get_vocab_size()), label.view(-1))
                if accelerator.is_main_process:
                    train_ctx.loss_list.append(loss.item())
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                local_sgd.step()
                if train_ctx.step > config.train['warmup_epochs'] and (train_ctx.step + 1) % config.train[
                    'random_epoch'] == 0:
                    with torch.no_grad():
                        # add random to model
                        add_random_value_by_weights(transformer, config.train['random_effect'] * (
                                config.train['warmup_epochs'] / train_ctx.step))
                if not accelerator.is_main_process:
                    continue
                # in main process
                if (train_ctx.step + 1) % 20000 == 0:
                    accelerator.print(f"\nstep: {train_ctx.step}, loss: {loss.item()}")
                    # save skip_gram
                    accelerator.print(f"save transformer: step: {train_ctx.step}")
                    accelerator.save_state("./data/transformer")
                    save_loss_list_graph(train_ctx.loss_list, f"./data/transformer.png")
                    accelerator.save_state(f"./data/transformer_{train_ctx.step}_{loss.item():.2f}")

                    # print original text
                    accelerator.print('original text:', tokenizer.decode_batch(ids_data)[0][:100])
                    # get result, get max index
                    result = torch.argmax(output, dim=-1)
                    accelerator.print('result ids:', result[:, 0][:100])
                    # to text token list
                    token_result = [tokenizer.id_to_token(i) for i in result[:, 0].tolist()]
                    accelerator.print('result token list:', token_result[:100])
                    # to text
                    text_result = tokenizer.decode_batch(result.permute(1, 0).tolist())
                    accelerator.print('result text:', text_result[0])


def check_transformer(input_str, root='./data/transformer'):
    (
        accelerator,
        tokenizer,
        transformer,
        train_ctx,
        optimizer,
        scheduler,
        loss_function,
        data_loader
    ) = init_all_object(root)

    # prepare training
    transformer.eval()

    #
    max_seq_len = 698
    # #  启用填充，最大长度为1024， 对于长度不足1024的序列，用3填充。 对于长度超过1024的序列，进行截断
    # tokenizer.enable_padding(length=max_seq_len, pad_id=3, pad_token="[PAD]")
    #  启用截断，最大长度为1024
    tokenizer.enable_truncation(max_length=max_seq_len)
    # make embedding
    embedding = get_embedding(tokenizer).to(accelerator.device)
    embedding.eval()
    # 启用旋转位置编码
    position_encoding = RotateEmbedding(max_seq_len, 768).to(accelerator.device)
    position_encoding.eval()

    text_data = {
        'text': [
            input_str,
        ],
    }

    # process data
    with torch.no_grad():
        split_data = tokenizer.encode_batch(text_data['text'])
        label = torch.tensor([i.ids for i in split_data]).to(accelerator.device)
        label = label[:, :-1]
        for _ in range(500):
            data = embedding(label)
            data = position_encoding(data)
            # data = data + torch.randn_like(data) * 0.1

            data = data.permute(1, 0, 2)

            mask = generate_mask(data.size(0), accelerator.device)

            output = transformer(data, mask=mask)
            # output = output + torch.randn_like(output) * 0.4

            # print original text
            accelerator.print('original text:', text_data['text'][0][:100])
            # get result, get max index
            result = torch.argmax(output, dim=-1)
            accelerator.print('result ids:', result[:, 0][:100])
            # to text token list
            token_result = [tokenizer.id_to_token(i) for i in result[:, 0].tolist()]
            accelerator.print('result token list:', token_result[:100])
            # to text
            text_result = tokenizer.decode_batch(result.permute(1, 0).tolist())
            accelerator.print('result text:', text_result[0])

            # add next token mask
            label = torch.cat([label, result.permute(1, 0)[:, -1:]], dim=-1)

            # output text
            print('output text:', tokenizer.decode_batch(label.tolist())[0])
            if tokenizer.decode_batch(label.tolist())[0].rfind('</ text >') != -1:
                break


if __name__ == '__main__':
    # check_transformer('/root/autodl-tmp/project/data_v1/transformer')
    #     check_transformer('''title:
    # Kopylovo, Vologodsky District, Vologda Oblast
    # text:
    # Kopylovo () is a''')
    check_transformer('''question: what is your name?
answer:''')
    # train_transformer()
