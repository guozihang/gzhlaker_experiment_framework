# -*- encoding: utf-8 -*-
"""
@File    :   base_parser.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/20 3:37 PM   Gzhlaker      1.0         None
"""
import os


class BaseParser:

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


