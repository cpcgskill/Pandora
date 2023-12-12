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

if False:
    pass


# from pandora.data.compile_dataset import get_main_dataset
# from pandora.tokenizer_ import train_tokenizer
# train_tokenizer(get_main_dataset(keep_in_memory=True))

# from pandora.data.post_processe import generate_train_and_test_dataset, generate_pretokenize_dataset
# generate_pretokenize_dataset()
# generate_train_and_test_dataset(chunk_size=512)

def start():
    # from pandora.data.post_processe import get_pretokenize_dataset, get_main_dataset
    # from pandora.CBOW import train_embedding, build_embedding
    # train_embedding(get_main_dataset(keep_in_memory=True), './data/embedding2')
    # build_embedding('./data/embedding2', '/root/autodl-fs/embedding.pt')
    # from pandora.SkipGram import train_skip_gram, build_embedding_from_skip_gram
    # train_skip_gram(
    #     get_main_dataset().shuffle(keep_in_memory=True),
    #     './data/skip_gram',
    # )
    # build_embedding_from_skip_gram('./data/skip_gram')
    from pandora.kernel import train_transformer,check_transformer
    from pandora.data.post_processe import get_train_dataset
    train_transformer(get_train_dataset(keep_in_memory=False), './data/transformer24')
    # check_transformer('give me a python example.\n', './data/transformer22')
    # check_transformer('give me a python example.\n', './data/transformer3')



import accelerate

num_processes = 1
if num_processes > 1:
    accelerate.notebook_launcher(start, (), num_processes=num_processes, mixed_precision='fp16')
else:
    start()
