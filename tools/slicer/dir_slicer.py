# -*- encoding: utf-8 -*-
"""
@File    :   dir_slicer.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/3/13 5:16 下午   Gzhlaker      1.0         None
"""
import os.path
import sys

sys.path.append(".")
from core.util.util import get_split_list


def get_file_name(path):
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


def main(in_path, check_path_1, check_path_2):
    in_files = get_file_name(in_path)
    out_files_1 = get_file_name(check_path_1)
    out_files_2 = get_file_name(check_path_2)
    _l = []
    for file in in_files:
        if ((file[:-4] + "_rgb.npy") not in out_files_1) and ((file[:-4] + "_rgb.npy") not in out_files_2):
            _l.append(os.path.join(in_path, file))
            print(os.path.join(in_path, file))
        else:
            print(file, " exists")
    get_split_list(_l, 6, name_template="activitynet_slipt_{}.txt")


def didemo_main(in_path, in_path_2, check_path, out_path):
    in_files = get_file_name(in_path)
    in_files_2 = get_file_name(in_path_2)
    out_files = get_file_name(check_path)
    with open(out_path, mode="w") as f:
        for file in in_files:
            if ((file[:-4] + "_rgb.npy") not in out_files) and (file[-4:] == ".mp4"):
                f.write(os.path.join(in_path, file) + "\n")
                print(os.path.join(in_path, file))
            else:
                print(file, " exists")
        for file in in_files_2:
            if ((file[:-4] + "_rgb.npy") not in out_files) and (file[-4:] == ".mp4"):
                f.write(os.path.join(in_path_2, file) + "\n")
                print(os.path.join(in_path_2, file))
            else:
                print(file, " exists")


def ac_main(in_path, out_path):
    in_files = get_file_name(in_path)
    _l = []
    for file in in_files:
        _l.append(os.path.join(in_path, file))
        print(os.path.join(in_path, file))
    get_split_list(_l, 6, name_template="activitynet__test_slipt_{}.txt")


if __name__ == "__main__":
    ac_main(
        in_path="/data02/yangyang/VTR/datasets/ActivityNetDataset/video/val2",
        out_path="/data02/gzh/acnet/"
    )
