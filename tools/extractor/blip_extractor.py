# -*- encoding: utf-8 -*-
"""
@File    :   blip_extractor.py.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/3/14 10:31 上午   Gzhlaker      1.0         None
"""
import sys

sys.path.append(".")

from base_extractor import BaseExtractor

from three.blip.models.blip import blip_feature_extractor


class BlipExtractor(BaseExtractor):
    def __init__(self, pretrain_path):
        self.pretrain_path = pretrain_path
        self.model = None

    def get_model(self):
        """
        读取预训练模型
        Returns:

        """
        self.model = blip_feature_extractor(self.pretrain_path)
