# -*-coding:utf-8 -*-
"""
:创建时间: 2023/12/24 18:59
:作者: 苍之幻灵
:我的主页: https://cpcgskill.com
:Github: https://github.com/cpcgskill
:QQ: 2921251087
:aboutcg: https://www.aboutcg.org/teacher/54335
:bilibili: https://space.bilibili.com/351598127
:爱发电: https://afdian.net/@Phantom_of_the_Cang

"""
from __future__ import unicode_literals, print_function, division

import os.path

if False:
    from typing import *
import functools
import datasets
import hashlib

def dataset_cache(fn):
    """对dataset进行缓存"""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        cache_id = hashlib.sha1('{}_{}'.format(args, kwargs).encode('utf-8')).hexdigest()
        cache_name = fn.__name__ + '_' + cache_id
        cache_dir_path = './dataset_cache/{}'.format(cache_name)
        if os.path.isdir(cache_dir_path):
            if os.path.isdir('{}/0'.format(cache_dir_path)):
                return tuple(
                    datasets.load_from_disk('{}/{}'.format(cache_dir_path, idx))
                    for idx in range(len(os.listdir(cache_dir_path)))
                )
            else:
                return datasets.load_from_disk(cache_dir_path)
        else:
            dataset = fn(*args, **kwargs)
            if isinstance(dataset, tuple):
                for idx, i in enumerate(dataset):
                    i.save_to_disk('{}/{}'.format(cache_dir_path, idx))
            else:
                dataset.save_to_disk(cache_dir_path)
            return dataset

    return wrapper
