'''
Descripttion: 
version: 
Author: Gzhlaker
Date: 2022-01-26 12:29:37
LastEditors: Andy
LastEditTime: 2022-02-12 00:16:22
'''
import sys
sys.path.append(".")
from core.printer import Printer
class base_runner:
    '''define basic hook, state, config'''
    def __init__(self):
        self._init_hook()
        self._init_state()
        self._init_config()

    def _init_hook(self):
        self.hook = {
            "on_init": self.on_init,
            "on_parse_argument": self.on_parse_argument,
            "on_get_config": self.on_get_config,
            "on_get_dataset": self.on_get_dataset,
            "on_get_dataLoader": self.on_get_dataLoader,
            "on_get_model": self.on_get_model,
            "on_get_loss": self.on_get_loss,
            "on_get_oprimizer": self.on_get_oprimizer,
            # train
            "on_start_train": self.on_start_train,
            "on_train": self.on_train,
            "on_end_train": self.on_end_train,
            # epoch
            "on_start_epoch": self.on_start_epoch,
            "on_epoch": self.on_epoch,
            "on_end_epoch": self.on_end_epoch,
            #
            "on_set_grad": self.on_set_grad,
            "on_calculate_loss": self.on_calculate_loss,
            "on_calculate_back_grad": self.on_calculate_back_grad,
            "on_update_parameter": self.on_update_parameter,
            "on_calculate_matric": self.on_calculate_matric,
            # valid
            "on_start_valid": self.on_start_valid,
            "on_valid": self.on_valid,
            "on_end_valid": self.on_end_valid,
            # test
            "on_start_test": None,
            "on_test": None,
            "on_end_test": None
            
        }
    
    def _init_state(self):
        self.state = {
            "train_model": None,
            "valid_model": None,
            "train_iter": None,
            "valid_iter": None,
        }
    def _init_config(self):
        self.config = {}

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

    def on_train(self):
        pass

    @Printer.function_name
    def on_end_train(self):
        pass

    # @Printer.function_name
    def on_start_epoch(self):
        pass

    @Printer.function_name
    def on_epoch(self):
        pass

    # @Printer.function_name
    def on_end_epoch(self):
        pass

    # @Printer.function_name
    def on_start_valid(self):
        pass

    def on_valid(self):
        pass
    # @Printer.function_name
    def on_end_valid(self):
        pass

    @Printer.function_name
    def on_start_valid_epoch(self):
        pass

    @Printer.function_name
    def on_end_valid_epoch(self):
        pass

    def on_set_grad(self):
        pass

    def on_calculate_loss(self):
        pass

    def on_calculate_back_grad(self):
        pass

    def on_update_parameter(self):
        pass

    def on_calculate_matric(self):
        pass
