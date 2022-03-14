# -*- encoding: utf-8 -*-
"""
@File    :   path_manager.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/28 11:05 AM   Gzhlaker      1.down.sh         None
"""
import os
import time
import sys

sys.path.append(".")
from core.util.util import Util


class PathManager:
    log_path = ""
    program_path = ""
    dataset_path = None
    pretrained_path = None

    @staticmethod
    def get_log_path():
        if PathManager.log_path == "":
            if not os.path.exists(os.path.join(PathManager.get_program_path(), "result")):
                os.mkdir(os.path.join(PathManager.get_program_path(), "result"))
            time_string = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            if not os.path.exists(os.path.join(PathManager.get_program_path(), "result", time_string)):
                os.mkdir(os.path.join(PathManager.get_program_path(), "result", time_string))
            PathManager.log_path = os.path.join(PathManager.get_program_path(), "result", time_string)
            return PathManager.log_path
        else:
            return PathManager.log_path

    @staticmethod
    def get_program_path():
        if PathManager.program_path == "":
            PathManager.program_path = os.getcwd()
            return PathManager.program_path
        else:
            return PathManager.program_path

    @staticmethod
    def get_dataset_path(name: tuple):
        if PathManager.dataset_path is None:
            PathManager.dataset_path = Util.get_yaml_data("config/path/dataset_path.yaml")
        _temp = PathManager.dataset_path
        for _path in name:
            if type(_temp) != str:
                _temp = _temp[_path]
        return _temp

    @staticmethod
    def get_pretrained_path(name: tuple):
        if PathManager.pretrained_path is None:
            PathManager.pretrained_path = Util.get_yaml_data("config/path/pretrained_path.yaml")
        _temp = PathManager.pretrained_path
        for _path in name:
            if type(_temp) != str:
                _temp = _temp[_path]
        return _temp


def test():
    print(PathManager.get_dataset_path(("HOW2SIGN", "VIDEO", "RAW", "TRAIN")))


if __name__ == "__main__":
    test()
