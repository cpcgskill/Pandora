# -*-coding:utf-8 -*-
"""
:创建时间: 2023/9/24 23:10
:作者: 苍之幻灵
:我的主页: https://cpcgskill.com
:Github: https://github.com/cpcgskill
:QQ: 2921251087
:aboutcg: https://www.aboutcg.org/teacher/54335
:bilibili: https://space.bilibili.com/351598127
:爱发电: https://afdian.net/@Phantom_of_the_Cang

"""
from __future__ import unicode_literals, print_function, division

from dataproc.compile_dataset import get_merge_custom_answer_dataset
from pandora.config import Config
from pandora.tokenizer_ import train_tokenizer, get_tokenizer

config = Config()
source_dataset = get_merge_custom_answer_dataset()

print('=' * 30, 'train_tokenizer', '=' * 30)

train_tokenizer(
    config,
    source_dataset
)

tokenizer = get_tokenizer(config)
# tokenizer.model = WordPiece.from_file(config.tokenizer_path)
print(tokenizer.encode("Hello, y'all! How are you 😁 ?").tokens)
print(tokenizer.encode(source_dataset[0]['text']).tokens)
