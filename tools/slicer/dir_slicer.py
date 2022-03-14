# -*- encoding: utf-8 -*-
"""
@File    :   dir_slicer.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/3/13 5:16 下午   Gzhlaker      1.0         None
"""
import sys
sys.path.append(".")
from core.util.util import get_file_name, get_split_list


def main(path):
    data = get_file_name(path)
    get_split_list(data, 6, name_template="sta_split_{}.pkl")


if __name__ == "__main__":
    main(
        "/data01/yangyang/liuxiaolei/Charades_v1"
    )
