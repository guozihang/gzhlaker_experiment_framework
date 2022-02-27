# -*- encoding: utf-8 -*-
"""
@File    :   group_solarization.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/22 3:31 PM   Gzhlaker      1.0         None
"""
import random

from PIL import ImageOps


class GroupSolarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img_group):
        if random.random() < self.p:
            return [ImageOps.solarize(img)  for img in img_group]
        else:
            return img_group