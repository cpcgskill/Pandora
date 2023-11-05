# -*-coding:utf-8 -*-
"""
:创建时间: 2023/9/17 21:41
:作者: 苍之幻灵
:我的主页: https://cpcgskill.com
:Github: https://github.com/cpcgskill
:QQ: 2921251087
:aboutcg: https://www.aboutcg.org/teacher/54335
:bilibili: https://space.bilibili.com/351598127
:爱发电: https://afdian.net/@Phantom_of_the_Cang

"""
from __future__ import unicode_literals, print_function, division

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

from data_admin import DatasetAdmin


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


# 定义CBOW模型
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


def make_embedding(tokenizer):
    # make embedding
    return CBOW(tokenizer.get_vocab_size(), 768)


def train_embedding():
    # accelerate
    accelerator = accelerate.Accelerator()
    print('device:', accelerator.device)

    # check tokenizer
    print("check tokenizer")
    test_tokenizer()
    # make embedding
    tokenizer = Tokenizer.from_file("tokenizer.json")
    embedding = make_embedding(tokenizer)
    embedding.train()
    # train embedding
    data_admin = DatasetAdmin()
    dataset = data_admin.get_train_dataset()
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4096,
        shuffle=True,
        pin_memory=True,
    )
    optimizer = torch.optim.Adagrad(embedding.parameters(), lr=1.0 / 768, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    loss_function = nn.CrossEntropyLoss()

    data_loader, embedding, optimizer, scheduler, loss_function = accelerator.prepare(
        data_loader, embedding, optimizer, scheduler, loss_function
    )

    #  启用填充，最大长度为1024， 对于长度不足1024的序列，用3填充。 对于长度超过1024的序列，进行截断
    tokenizer.enable_padding(length=1024, pad_id=3, pad_token="[PAD]")
    #  启用截断，最大长度为1024
    tokenizer.enable_truncation(max_length=1024)
    for epoch in range(10):
        for step, text_data in enumerate(tqdm.tqdm(data_loader, disable=not accelerator.is_local_main_process)):
            tokens = tokenizer.encode_batch(text_data["text"])
            token_ids = [i.ids for i in tokens]
            token_ids = torch.Tensor(token_ids).long()
            token_ids = token_ids.to(accelerator.device)

            optimizer.zero_grad()
            # 向前向后各取2个词，共4个词作为上下文
            for i in range(2, token_ids.shape[1] - 2):
                context = torch.stack(
                    [
                        token_ids[:, i - 2],
                        token_ids[:, i - 1],
                        token_ids[:, i + 1],
                        token_ids[:, i + 2],
                    ],
                    dim=1,
                )
                target = token_ids[:, i]
                output = embedding(context)
                loss = loss_function(output, target)
                accelerator.backward(loss / (token_ids.shape[1] - 4))
            optimizer.step()
            scheduler.step()

            if accelerator.is_main_process:
                tqdm.tqdm.write(f"epoch: {epoch}, loss: {loss.item()}")
                if step % 1000 == 0:
                    # save embedding
                    tqdm.tqdm.write(f"save embedding: {epoch}")
                    torch.save(embedding.state_dict(), f"/root/autodl-fs/embedding_{epoch}.pt")

        # must save embedding
        print(f"save embedding: {epoch}")
        torch.save(embedding.state_dict(), f"/root/autodl-fs/embedding_{epoch}.pt")

    # test embedding
    test_embedding()


def test_embedding():
    # load embedding
    tokenizer = Tokenizer.from_file("tokenizer.json")
    embedding = make_embedding(tokenizer)
    embedding.load_state_dict(torch.load("embedding.pt"))
    # test embedding
    print("test embedding")
    embedding.eval()
    text = "Hello, y'all! How are you 😁 ?"
    tokens = tokenizer.encode(text)
    tokens = torch.Tensor([tokens.ids]).long()
    output = embedding(tokens)
    print(output.shape)
    print(output[0, 0, :10])


if __name__ == '__main__':
    train_embedding()
