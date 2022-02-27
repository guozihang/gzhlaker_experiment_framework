# -*- encoding: utf-8 -*-
"""
@File    :   stack_1.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/22 3:55 PM   Gzhlaker      1.0         None
"""
import numpy as np
import torch


class Stack1(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):

        if self.roll:
            return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
        else:

            rst = np.concatenate(img_group, axis=0)
            # plt.imshow(rst[:,:,3:6])
            # plt.show()
            return torch.from_numpy(rst)