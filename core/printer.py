'''
Descripttion: 
version: 
Author: Gzhlaker
Date: 2022-01-25 19:36:59
LastEditors: Andy
LastEditTime: 2022-01-26 12:29:02
'''


from rich.panel import Panel
from rich.progress import Progress

from core.util import Util

class Printer:
    '''
    rich-base logger
    '''
    console = Util.init_rich()
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
        Printer.console.log(str)

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



    