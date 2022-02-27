'''
Descripttion: 
version: 
Author: Gzhlaker
Date: 2022-02-11 22:17:25
LastEditors: Andy
LastEditTime: 2022-02-11 23:22:51
'''
import logging
from rich.logging import RichHandler


class log_manager:
    """this is a static method, """

    @staticmethod
    def get_logger(filename):
        FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(
            level="NOTSET",
            format=FORMAT,
            datefmt="[%X]",
            handlers=[
                RichHandler(
                    show_time = False,
                    show_path=False
                )
            ]
        )
        log = logging.getLogger("rich")
        formatter = logging.Formatter(FORMAT)
        file_handler = logging.FileHandler(filename, "w")
        file_handler.setLevel(logging.NOTSET)
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)

        return log
