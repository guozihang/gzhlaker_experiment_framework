# -*- encoding: utf-8 -*-
"""
@File    :   clip_extractor.py.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/3/17 6:12 下午   Gzhlaker      1.0         None
"""
import sys
sys.path.append(".")

from base_extractor import BaseExtractor

class ClipExtractor(BaseExtractor):
    def __init__(self, annotation_file, slice_file, video_path, out_path):
        self.annotation_file = annotation_file
        self.slice_file = slice_file
        self.video_path = video_path
        self.out_path = out_path