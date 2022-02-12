'''
Descripttion: 
version: 
Author: Gzhlaker
Date: 2022-01-25 19:36:59
LastEditors: Andy
LastEditTime: 2022-02-12 12:54:14
'''
import functools
import os
import sys
import time

from rich.panel import Panel
from rich.progress import Progress
from rich.console import Console
from rich.traceback import install
from core.log_manager import log_manager
install(show_locals=False)

if os.path.exists("./result") == False:
    os.mkdir(os.getcwd() + "/result")
timestr = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
os.mkdir(os.getcwd() + "/result/" + timestr)
class Printer:
    '''
    rich-base logger
    '''
    timestr = timestr
    log = log_manager.get_logger("./result/{}/{}.txt".format(timestr, timestr))
    console = Console(color_system = 'auto')
    progress_list = {}

    @staticmethod
    def print_title(data:str)-> None:
        Printer.console.print(data, justify='center')
    
    @staticmethod
    def print_panle(str, title="None")-> None:
        Printer.console.print(Panel(str, title=title))

    @staticmethod
    def print_rule(data: str)-> None:
        Printer.console.rule(data)
    
    @staticmethod
    def print_log(str)-> None:
        Printer.log.info(str)
    @staticmethod
    def function_name(func):
        @functools.wraps(func)
        def wrapper(*args, **kward):
            Printer.log.info(func.__name__)
            return func(*args, **kward)
        return wrapper

    @staticmethod
    def function_log(func, str):
        @functools.wraps(func)
        def wrapper(*args, **kward):
            Printer.log.info(str)
            return func(*args, **kward)
        return wrapper
    @staticmethod
    def create_progressor(name:str="Wow", total:int=1000)-> None:
        Printer.progress_list[name] = {}
        Printer.progress_list[name]["progress"] = Progress(expand = True)
        Printer.progress_list[name]["task"] = Printer.progress_list[name]["progress"].add_task(description=name, total=total)

    @staticmethod
    def get_progressor(name:str="Wow"):
        return Printer.progress_list[name]["progress"]
    
    @staticmethod
    def update_progressor(name:str="Wow", advance:int=0.1)-> None:
        with Printer.progress_list[name]["progress"]:
            if not Printer.progress_list[name]["progress"].finished:
                Printer.progress_list[name]["progress"].update(Printer.progress_list[name]["task"], advance=advance)
                Printer.progress_list[name]["progress"].refresh()
    
    @staticmethod
    def is_progressor_finished(name:str="Wow")-> bool:
        return Printer.progress_list[name]["progress"].finished
    
    @staticmethod
    def update_progressor_without_progress(name:str="Wow", advance:int=0.1):
        Printer.progress_list[name]["progress"].update(Printer.progress_list[name]["task"], advance=advance)

    
        
    