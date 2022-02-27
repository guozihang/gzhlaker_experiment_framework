# -*- encoding: utf-8 -*-
"""
@File    :   quick_GELU.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/20 10:43 AM   Gzhlaker      1.0         None
"""
import torch
from torch import nn


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
