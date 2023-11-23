# -*-coding:utf-8 -*-
"""
:创建时间: 2023/11/18 3:25
:作者: 苍之幻灵
:我的主页: https://cpcgskill.com
:Github: https://github.com/cpcgskill
:QQ: 2921251087
:aboutcg: https://www.aboutcg.org/teacher/54335
:bilibili: https://space.bilibili.com/351598127
:爱发电: https://afdian.net/@Phantom_of_the_Cang

"""
from __future__ import unicode_literals, print_function, division

from tokenizers import Tokenizer
from tokenizers import normalizers
from tokenizers.trainers import WordPieceTrainer
from tokenizers.models import WordPiece
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers import decoders

from pandora import config

def make_tokenizer():
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [EOS]",
        pair="[CLS] $A [SEP] $B:1 [EOS]",
        special_tokens=[
            ("[UNK]", 0),
            ("[CLS]", 1),
            ("[SEP]", 2),
            ("[EOS]", 3),
            ("[PAD]", 4),
            ("[MASK]", 5),
        ],
    )
    tokenizer.decoder = decoders.WordPiece()
    return tokenizer



def train_tokenizer(dataset):
    tokenizer = make_tokenizer()
    trainer = WordPieceTrainer(
        vocab_size=30522 * 5,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[EOS]", "[PAD]", "[MASK]"],
    )

    tokenizer.train_from_iterator(
        iterator=(dataset[i: i + 1000]["text"] for i in range(0, len(dataset), 1000)),
        trainer=trainer,
        length=len(dataset),
    )
    tokenizer.save(config.tokenizer_path)

    test_tokenizer()


def test_tokenizer():
    tokenizer = Tokenizer.from_file(config.tokenizer_path)
    # tokenizer.model = WordPiece.from_file(config.tokenizer_path)
    print(tokenizer.encode("Hello, y'all! How are you 😁 ?").tokens)
    print(tokenizer.encode("你好， 你还好吗？").tokens)


def get_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer.from_file(config.tokenizer_path)
    return tokenizer

if __name__ == '__main__':
    test_tokenizer()