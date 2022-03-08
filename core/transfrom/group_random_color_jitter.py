# -*- encoding: utf-8 -*-
"""
@File    :   group_random_color_jitter.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/22 3:33 PM   Gzhlaker      1.down.sh         None
"""
import random

import torchvision


class GroupRandomColorJitter(object):
    """Randomly ColorJitter the given PIL.Image with a probability
    """

    def __init__(self, p=0.8, brightness=0.4, contrast=0.4,
                 saturation=0.2, hue=0.1):
        self.p = p
        self.worker = torchvision.transforms.ColorJitter(brightness=brightness, contrast=contrast,
                                                         saturation=saturation, hue=hue)

    def __call__(self, img_group):

        v = random.random()
        if v < self.p:
            ret = [self.worker(img) for img in img_group]

            return ret
        else:
            return img_group
