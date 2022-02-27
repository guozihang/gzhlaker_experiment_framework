# -*- encoding: utf-8 -*-
"""
@File    :   image_clip.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/20 10:20 AM   Gzhlaker      1.0         None
"""
from torch import nn


class ImageCLIP(nn.Module):
    """
    use clip-encoder encode the image
    """

    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)
