# -*- encoding: utf-8 -*-
"""
@File    :   abstract_parser.py.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/4/18 8:50 下午   Gzhlaker      1.0         None
"""
import abc


class AbstractParser:

    @abc.abstractmethod
    def parse_raw_annotation_data(self):
        """
        解析原始的标注数据
        Returns:

        """
        pass

    @abc.abstractmethod
    def process(self):
        """
        处理和重新组合数据
        Returns:

        """
        pass

    @abc.abstractmethod
    def save(self):
        """
        将数据存储为固定的格式
        Returns:

        """
        pass
