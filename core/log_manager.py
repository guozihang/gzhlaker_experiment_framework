'''
Descripttion: 
version: 
Author: Gzhlaker
Date: 2022-02-11 22:17:25
LastEditors: Andy
LastEditTime: 2022-02-11 22:57:12
'''
import logging
from rich.logging import RichHandler

class log_manager:
    @staticmethod
    def get_logger(filename):
        log = logging.getLogger("rich")
        log.setLevel(logging.INFO)
        streamhandler = RichHandler()
        streamhandler.setLevel(logging.INFO)
        streamhandler.setFormatter("[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s")

        filehandler = logging.FileHandler(filename=filename, filemode="w")
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter("[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s")

        log.addHandler(log_manager.get_filehandler())
        return log