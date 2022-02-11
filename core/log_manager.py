'''
Descripttion: 
version: 
Author: Gzhlaker
Date: 2022-02-11 22:17:25
LastEditors: Andy
LastEditTime: 2022-02-11 23:21:13
'''
import logging
from rich.logging import RichHandler
from rich.console import Console
class log_manager:
    @staticmethod
    def get_logger(filename):
        FORMAT = "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
        logging.basicConfig(
            level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
        )
        log = logging.getLogger("rich")
        formatter = logging.Formatter(
            "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
        )
        filehandler = logging.FileHandler(filename, "w")
        filehandler.setLevel(logging.NOTSET)
        filehandler.setFormatter(formatter)
        log.addHandler(filehandler)

        return log