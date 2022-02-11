'''
Descripttion: 
version: 
Author: Gzhlaker
Date: 2022-02-11 22:17:25
LastEditors: Andy
LastEditTime: 2022-02-11 23:02:52
'''
import logging
from rich.logging import RichHandler

class log_manager:
    @staticmethod
    def get_logger(filename):
        log = logging.getLogger("rich")
        log.setLevel(logging.NOTSET)
        richhandler = RichHandler()
        richhandler.setLevel(logging.NOTSET)
        richhandler.setFormatter(" %(message)s")

        filehandler = logging.FileHandler(filename, "w")
        filehandler.setLevel(logging.NOTSET)
        filehandler.setFormatter(" %(message)s")

        log.addHandler(richhandler)
        log.addHandler(filehandler)
        return log