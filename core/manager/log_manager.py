'''
Descripttion: 
version: 
Author: Gzhlaker
Date: 2022-02-11 22:17:25
LastEditors: Andy
LastEditTime: 2022-02-11 23:22:51
'''
import logging
import os
import time

from rich.logging import RichHandler

from core.manager.path_manager import PathManager


class log_manager:
    """this is a static method, """
    @staticmethod
    def get_logger():
        time_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(
            level="NOTSET",
            format=time_format,
            datefmt="[%X]",
            handlers=[
                RichHandler(
                    show_time=False,
                    show_path=False
                ),
                logging.FileHandler(
                    os.path.join(PathManager.get_log_path(), "train.log")
                )
            ]
        )
        log = logging.getLogger("rich")
        return log
