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

if False:
    from typing import *

import torch
import torch.nn as nn
import tqdm

import accelerate

from tokenizers import Tokenizer
from tokenizers import normalizers
from tokenizers.trainers import WordPieceTrainer
from tokenizers.models import WordPiece
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers import decoders

from accelerate.local_sgd import LocalSGD

def make_tokenizer():
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", 1),
            ("[SEP]", 2),
        ],
    )
    tokenizer.decoder = decoders.WordPiece()
    return tokenizer


def train_tokenizer():
    tokenizer = make_tokenizer()
    trainer = WordPieceTrainer(
        vocab_size=30522 * 5,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    )

    data_admin = DatasetAdmin()
    dataset = data_admin.get_train_dataset()

    tokenizer.train_from_iterator(
        iterator=(dataset[i: i + 1000]["text"] for i in range(0, len(dataset), 1000)),
        trainer=trainer,
        length=len(dataset),
    )
    tokenizer.save("tokenizer.json")

    test_tokenizer()


def test_tokenizer():
    tokenizer = Tokenizer.from_file("tokenizer.json")
    # tokenizer.model = WordPiece.from_file("tokenizer.json")
    print(tokenizer.encode("Hello, y'all! How are you 😁 ?").tokens)
    print(tokenizer.encode("你好， 你还好吗？").tokens)


def get_tokenizer():
    tokenizer = Tokenizer.from_file("tokenizer.json")
    return tokenizer


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


def make_skip_gram(tokenizer):
    # make embedding
    return SkipGram(tokenizer.get_vocab_size(), 768)


def train_skip_gram():
    # accelerate
    accelerator = accelerate.Accelerator()
    print('device:', accelerator.device)

    # check tokenizer
    print("check tokenizer")
    test_tokenizer()
    # make embedding
    tokenizer = get_tokenizer()
    skip_gram = make_skip_gram(tokenizer)
    # train embedding
    data_admin = DatasetAdmin()
    dataset = data_admin.get_train_dataset()
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=6144,
        shuffle=True,
        pin_memory=True,
    )
    optimizer = torch.optim.Adagrad(skip_gram.parameters(), lr=1.0 / 768, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    loss_function = nn.CrossEntropyLoss()

    data_loader, skip_gram, optimizer, scheduler, loss_function = accelerator.prepare(
        data_loader, skip_gram, optimizer, scheduler, loss_function
    )

    if os.path.isdir("/root/autodl-fs/skip_gram"):
        accelerator.load_state("/root/autodl-fs/skip_gram")

    skip_gram.train()

    #  启用填充，最大长度为1024， 对于长度不足1024的序列，用3填充。 对于长度超过1024的序列，进行截断
    tokenizer.enable_padding(length=1024, pad_id=3, pad_token="[PAD]")
    #  启用截断，最大长度为1024
    tokenizer.enable_truncation(max_length=1024)
    with LocalSGD(accelerator=accelerator, model=skip_gram, local_sgd_steps=8, enabled=True) as local_sgd:
        for epoch in range(10):
            for step, text_data in enumerate(tqdm.tqdm(data_loader, disable=not accelerator.is_local_main_process)):
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

                if accelerator.is_main_process:
                    tqdm.tqdm.write(f"epoch: {epoch}, loss: {loss.item()}")
                    if step % 10 == 0:
                        # save skip_gram
                        tqdm.tqdm.write(f"save skip_gram: {epoch}")
                        accelerator.save_state("/root/autodl-fs/skip_gram")

            # must save skip_gram
            print(f"save skip_gram: {epoch}")
            accelerator.save_state(f"/root/autodl-fs/skip_gram_{epoch}")


def test_skip_gram():
    # accelerate
    accelerator = accelerate.Accelerator()
    print('device:', accelerator.device)

    # check tokenizer
    print("check tokenizer")
    test_tokenizer()
    # make skip_gram
    tokenizer = get_tokenizer()
    skip_gram = make_skip_gram(tokenizer)
    skip_gram = accelerator.prepare(skip_gram)
    # load skip_gram
    if os.path.isdir("/root/autodl-fs/skip_gram"):
        accelerator.load_state("/root/autodl-fs/skip_gram")
    # test skip_gram
    print("test skip_gram")
    skip_gram.eval()
    text = "Hello Hi Bad 你好"
    tokens = tokenizer.encode(text)
    print(tokens.tokens)
    print(tokens.ids)
    tokens = torch.Tensor([tokens.ids]).long()
    output = skip_gram(tokens.to(accelerator.device))
    print(output.shape)
    print(output)
    # num dot
    print(output[0, 1].dot(output[0, 2]))
    print(output[0, 1].dot(output[0, 3]))
    print(output[0, 1].dot(output[0, 4]))


def build_embedding_from_skip_gram():
    # accelerate
    accelerator = accelerate.Accelerator()
    print('device:', accelerator.device)

    # check tokenizer
    print("check tokenizer")
    test_tokenizer()
    # make skip_gram
    tokenizer = get_tokenizer()
    skip_gram = make_skip_gram(tokenizer)
    skip_gram = accelerator.prepare(skip_gram)
    # load skip_gram
    if os.path.isdir("/root/autodl-fs/skip_gram"):
        accelerator.load_state("/root/autodl-fs/skip_gram")

    # make embedding
    torch.save(skip_gram.in_embed.state_dict(), "/root/autodl-fs/skip_gram.in_embed.pt")

    print('test get embedding')
    get_embedding(get_tokenizer())


def make_embedding(tokenizer):
    # make embedding
    return nn.Embedding(tokenizer.get_vocab_size(), 768)


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
    print(output[0, 1].dot(output[0, 2]))
    print(output[0, 1].dot(output[0, 3]))
    print(output[0, 1].dot(output[0, 4]))


if __name__ == '__main__':
    # build_embedding_from_skip_gram()
    test_skip_gram()
    # train_skip_gram()
    # accelerator = accelerate.Accelerator()
    # e = get_embedding(accelerator, get_tokenizer())
    # accelerator.save_state("/root/autodl-fs/skip_gram.in_embed")
    #
    # # test_embedding()
