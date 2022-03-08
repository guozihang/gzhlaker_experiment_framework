# -*- encoding: utf-8 -*-
"""
@File    :   group_normalize_1.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/22 3:56 PM   Gzhlaker      1.down.sh         None
"""
import torchvision


class GroupNormalize1(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.worker = torchvision.transforms.Normalize(mean, std)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]