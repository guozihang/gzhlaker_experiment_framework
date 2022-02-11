'''
Descripttion: 
version: 
Author: Gzhlaker
Date: 2022-02-11 22:17:25
LastEditors: Andy
LastEditTime: 2022-02-11 22:52:53
'''
import logging
from unicodedata import name
from rich.logging import RichHandler

class log_manager:
    level = None
    formater = None
    filehandler = None
    @staticmethod
    def get_logger(filename):
        log_manager.set_level()
        log_manager.set_fotmater()

        logging.basicConfig(
            level=logging.INFO,
            format=log_manager.get_formater(), 
            datefmt="[%X]", 
            handlers=[RichHandler()],
            # filename=filename,
            # filemode="w"
        )

        log = logging.getLogger("rich")
        log.setLevel(log_manager.get_level())
        return log
    
    @staticmethod
    def set_level(level={
            0: logging.DEBUG,
            1: logging.INFO,
            2: logging.WARNING,
            3: logging.ERROR
        }):
        log_manager.level = level
    
    @staticmethod
    def get_level():
        return log_manager.level

    @staticmethod
    def set_fotmater(formater = "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"):
        log_manager.formater = formater

    @staticmethod
    def get_formater():
        return log_manager.formater
        

    