# -*- encoding: utf-8 -*-
"""
@File    :   to_torch_format_tensor_1.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/22 3:54 PM   Gzhlaker      1.down.sh         None
"""
import torchvision


class ToTorchFormatTensor1(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [down.sh, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [down.sh.down.sh, 1.down.sh] """

    def __init__(self, div=True):
        self.worker = torchvision.transforms.ToTensor()

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]
