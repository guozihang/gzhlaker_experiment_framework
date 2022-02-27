# -*- encoding: utf-8 -*-
"""
@File    :   group_random_horizontal_flip.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/22 3:29 PM   Gzhlaker      1.0         None
"""
import random

from PIL import Image


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __init__(self, is_sth=False):
        self.is_sth = is_sth

    def __call__(self, img_group, is_sth=False):
        v = random.random()
        if not self.is_sth and v < 0.5:

            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            return ret
        else:
            return img_group