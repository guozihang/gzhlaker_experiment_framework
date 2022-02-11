'''
Descripttion: 
version: 
Author: Gzhlaker
Date: 2022-02-11 22:17:25
LastEditors: Andy
LastEditTime: 2022-02-11 23:10:16
'''
import logging
from rich.logging import RichHandler
from rich.console import Console
class log_manager:
    @staticmethod
    def get_logger(filename):
        FORMAT = "%(message)s"
        logging.basicConfig(
            level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
        )
        log = logging.getLogger("rich")

        filehandler = logging.FileHandler(filename, "w")
        filehandler.setLevel(logging.NOTSET)
        filehandler.setFormatter(" %(message)s")
        log.addHandler(filehandler)
        
        return log