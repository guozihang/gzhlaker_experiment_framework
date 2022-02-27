# -*- encoding: utf-8 -*-
"""
@File    :   group_gaussian_blur.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/22 3:42 PM   Gzhlaker      1.0         None
"""
import random

from PIL import ImageFilter


class GroupGaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img_group):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return [img.filter(ImageFilter.GaussianBlur(sigma)) for img in img_group]
        else:
            return img_group
