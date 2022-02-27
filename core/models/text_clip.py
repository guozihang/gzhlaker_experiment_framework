# -*- encoding: utf-8 -*-
"""
@File    :   text_clip.py
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/20 10:13 AM   Gzhlaker      1.0         None
"""
from torch import nn


class TextCLIP(nn.Module):
    """
    use clip-encoder encode the text-token
    """

    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)
