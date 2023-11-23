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


def start():

    from pandora.kernel import train_transformer, check_transformer
    from pandora.data.post_processe import get_train_dataset
    train_transformer(get_train_dataset(keep_in_memory=True), './data/transformer')

import accelerate
num_processes=1
if num_processes > 1:
    accelerate.notebook_launcher(start, (), num_processes=num_processes, mixed_precision='fp16')
else:
    start()
