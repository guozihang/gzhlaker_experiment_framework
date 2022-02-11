'''
Descripttion: 
version: 
Author: Gzhlaker
Date: 2022-01-26 12:29:37
LastEditors: Andy
LastEditTime: 2022-02-11 13:26:40
'''
import sys
sys.path.append(".")
from core.printer import Printer
class base_runner:
    '''define basic hook'''
    def __init__(self):
        self.hook = {
            "on_init": self.on_init,
            "on_parse_argument": self.on_parse_argument,
            "on_get_config": self.on_get_config,
            "on_get_dataset": self.on_get_dataset,
            "on_get_dataLoader": self.on_get_dataLoader,
            "on_get_model": self.on_get_model,
            "on_get_loss": self.on_get_loss,
            "on_get_oprimizer": self.on_get_oprimizer,
            "on_start_train": self.on_start_train,
            "on_end_train": self.on_end_train,
            "on_start_epoch": self.on_start_epoch,
            "on_end_epoch": self.on_end_epoch
        }

    @Printer.function_name
    def on_init(self):
        pass

    @Printer.function_name
    def on_parse_argument(self):
        pass

    @Printer.function_name
    def on_get_config(self):
        pass

    @Printer.function_name
    def on_get_dataset(self):
        pass

    @Printer.function_name
    def on_get_dataLoader(self):
        pass

    @Printer.function_name
    def on_get_model(self):
        pass

    @Printer.function_name
    def on_get_loss(self):
        pass

    @Printer.function_name
    def on_get_oprimizer(self):
        pass

    @Printer.function_name
    def on_start_train(self):
        pass
    
    @Printer.function_name
    def on_start_train(self):
        pass

    @Printer.function_name
    def on_end_train(self):
        pass

    @Printer.function_name
    def on_start_epoch(self):
        pass

    @Printer.function_name
    def on_end_epoch(self):
        pass
