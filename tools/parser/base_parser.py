# -*- encoding: utf-8 -*-
"""
@File    :   base_parser.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/20 3:37 PM   Gzhlaker      1.down.sh         None
"""
import os
from abc import ABC

import cv2


class BaseParser:
    """
    所有标注解析类的基类，实现了一些工具方法
    """

    def get_file_count(self, path):
        """
        获取一个文件夹下的文件数量
        Args:
            path

        Returns:
            file number
        """
        count = 0
        for file in os.listdir(path):
            count = count + 1
        return count

    def get_file_name(self, path):
        """
        获取一个文件夹下的所有名称
        Args:
            path:

        Returns:

        """
        files = []
        for file in os.listdir(path):
            files.append(file)
        return files

    def get_file_info(self, filepath):
        cap = cv2.VideoCapture(filepath)
        # get video info
        video_info = {
            "video_width": cap.get(3),
            "video_height": cap.get(4),
            "video_fps": int(cap.get(5)),
            "video_frame_num": int(cap.get(7)),
            "video_format": cap.get(8)

        }
        cap.release()
        return video_info
