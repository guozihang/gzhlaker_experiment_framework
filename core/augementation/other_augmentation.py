# -*- encoding: utf-8 -*-
"""
@File    :   other_augmentation.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/22 3:49 PM   Gzhlaker      1.down.sh         None
"""
import torchvision

from core.transfrom.group_center_crop import GroupCenterCrop
from core.transfrom.group_gaussian_blur import GroupGaussianBlur
from core.transfrom.group_multi_scale_group import GroupMultiScaleCrop
from core.transfrom.group_normalize import GroupNormalize
from core.transfrom.group_random_color_jitter import GroupRandomColorJitter
from core.transfrom.group_random_gray_scale import GroupRandomGrayscale
from core.transfrom.group_random_horizontal_flip import GroupRandomHorizontalFlip
from core.transfrom.group_scale import GroupScale
from core.transfrom.group_solarization import GroupSolarization
from core.transfrom.stack import Stack
from core.transfrom.to_torch_format_tensor import ToTorchFormatTensor


class OtherAugmentation:
    def __call__(self, training, config):
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        scale_size = config["data"]["input_size"] * 256 // 224
        if training:
            unique = torchvision.transforms.Compose(
                [
                    GroupMultiScaleCrop(config["data"]["input_size"], [1, .875, .75, .66]),
                    GroupRandomHorizontalFlip(is_sth='some' in config["data"]["dataset"]),
                    GroupRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4,saturation=0.2, hue=0.1),
                    GroupRandomGrayscale(p=0.2),
                    GroupGaussianBlur(p=0.0),
                    GroupSolarization(p=0.0)
                ]
            )
        else:
            unique = torchvision.transforms.Compose(
                [
                    GroupScale(scale_size),
                    GroupCenterCrop(config["data"]["input_size"])
                ]
            )

        common = torchvision.transforms.Compose(
            [
                Stack(roll=False),
                ToTorchFormatTensor(div=True),
                GroupNormalize(input_mean, input_std)
            ]
        )
        return torchvision.transforms.Compose([unique, common])