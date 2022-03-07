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
import time

from rich.panel import Panel
from rich.progress import Progress
from rich.console import Console
from rich.pretty import Pretty
from rich.traceback import install
from rich.rule import Rule
from core.manager.log_manager import log_manager
from core.manager.path_manager import PathManager
install(show_locals=False)


class Printer:
    """
    rich-base logger
    """
    log = log_manager.get_logger()
    console = Console(
        color_system='auto',
        log_time_format="[%Y-%m-%d %H:%M:%S]"
    )

    progress_list = {}

    @staticmethod
    def print_title(data: str) -> None:
        Printer.print_log(data, justify='center')

    @staticmethod
    def print_panle(str, title="None") -> None:
        pretty = Pretty(str)
        Printer.print_log(Panel(pretty, title=title))

    @staticmethod
    def print_rule(data: str, characters="*") -> None:
        Printer.print_log(Rule(data, characters=characters))

    @staticmethod
    def print_log(data) -> None:
        Printer.console.log(data)
        with open(os.path.join(PathManager.get_log_path(), "train.log"), "a") as report_file:
            _console = Console(
                color_system='auto',
                file=report_file,
                log_time_format="[%Y-%m-%d %H:%M:%S]"
            )
            _console.log(data)

    @staticmethod
    def function_name(func):
        @functools.wraps(func)
        def wrapper(*args, **kward):
            Printer.print_log(func.__name__)
            return func(*args, **kward)

        return wrapper

    @staticmethod
    def function_log(func, str):
        @functools.wraps(func)
        def wrapper(*args, **kward):
            Printer.print_log(str)
            return func(*args, **kward)

        return wrapper

    @staticmethod
    def create_progressor(name: str = "Wow", total: int = 1000) -> None:
        Printer.progress_list[name] = {}
        Printer.progress_list[name]["progress"] = Progress(expand=True)
        Printer.progress_list[name]["task"] = Printer.progress_list[name]["progress"].add_task(description=name,
                                                                                               total=total)

    @staticmethod
    def get_progressor(name: str = "Wow"):
        return Printer.progress_list[name]["progress"]

    @staticmethod
    def update_progressor(name: str = "Wow", advance: int = 0.1) -> None:
        with Printer.progress_list[name]["progress"]:
            if not Printer.progress_list[name]["progress"].finished:
                Printer.progress_list[name]["progress"].update(Printer.progress_list[name]["task"], advance=advance)
                Printer.progress_list[name]["progress"].refresh()

    @staticmethod
    def is_progressor_finished(name: str = "Wow") -> bool:
        return Printer.progress_list[name]["progress"].finished

    @staticmethod
    def update_progressor_without_progress(name: str = "Wow", advance: int = 0.1):
        Printer.progress_list[name]["progress"].update(Printer.progress_list[name]["task"], advance=advance)
