# -*- encoding: utf-8 -*-
"""
@File    :   path_manager.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/28 11:05 AM   Gzhlaker      1.0         None
"""
import os
import time


class PathManager:
    log_path = ""
    program_path = ""

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
    def get_data_path():
        pass
