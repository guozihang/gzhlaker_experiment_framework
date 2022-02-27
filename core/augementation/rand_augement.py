# -*- encoding: utf-8 -*-
"""
@File    :   rand_augement.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/22 3:46 PM   Gzhlaker      1.0         None
"""
from core.transfrom.group_transfrom import GroupTransform


class RandAugment:
    def __call__(self, transform_train, config):
        print('Using RandAugment!')
        transform_train.transforms.insert(0, GroupTransform(RandAugment(config["data"]["randaug"]["N"], config["data"]["randaug"]["M"])))
        return transform_train
