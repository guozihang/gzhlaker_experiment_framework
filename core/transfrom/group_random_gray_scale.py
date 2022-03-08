# -*- encoding: utf-8 -*-
"""
@File    :   group_random_gray_scale.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/22 3:32 PM   Gzhlaker      1.down.sh         None
"""
import random

import torchvision


class GroupRandomGrayscale(object):
    """Randomly Grayscale flips the given PIL.Image with a probability
    """

    def __init__(self, p=0.2):
        self.p = p
        self.worker = torchvision.transforms.Grayscale(num_output_channels=3)

    def __call__(self, img_group):

        v = random.random()
        if v < self.p:
            ret = [self.worker(img) for img in img_group]

            return ret
        else:
            return img_group
