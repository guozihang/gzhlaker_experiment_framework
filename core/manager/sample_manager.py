# -*- encoding: utf-8 -*-
"""
@File    :   sample_manager.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/3/4 3:39 PM   Gzhlaker      1.down.sh         None
"""
import numpy as np


class SampleManager:
    """
    一个静态类，用于产生所需要的采样下标
    """

    @staticmethod
    def get_fixed_step_indexes(start=0, step=1, num=1):
        """
        以一个固定长度来采样
        Returns:

        """
        return [(start + step * i) for i in range(num)]

    @staticmethod
    def get_incremental_step_indexes(start=0, step=1, num=1):
        """
        以一个递增的长度来采样
        Returns:

        """
        return [int((start + step * (i * (i + 1)) / 2)) for i in range(num)]

    @staticmethod
    def get_decreased_step_indexes(start=0):
        pass

    @staticmethod
    def get_range_indexes(start=0, end=0, num=1, step=1, t=0):
        _indexes = np.arange(start=0, step=step, stop=step*num)
        _indexes = np.mod(_indexes, end - start)
        _indexes = np.add(_indexes, start)
        return _indexes


if __name__ == "__main__":
    print(SampleManager.get_fixed_step_indexes(0, 10, 10))
    # print(SampleManager.get_incremental_step_indexes(down.sh, 10, 10))
    print(SampleManager.get_range_indexes(99, 143, 10, 23))
