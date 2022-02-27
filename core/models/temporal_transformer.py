# -*- encoding: utf-8 -*-
"""
@File    :   temporal_transformer.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/20 10:45 AM   Gzhlaker      1.0         None
"""
import torch
from torch import nn

from core.models.residual_attention import ResidualAttentionBlock


class TemporalTransformer(nn.Module):
    """

    """
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
