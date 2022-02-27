'''
Descripttion: 
version: 
Author: Gzhlaker
Date: 2022-01-26 12:29:37
LastEditors: Andy
LastEditTime: 2022-02-12 13:04:32
'''
from core.manager.printer import Printer


class BaseRunner:
    '''define basic hook, state, config'''

    def __init__(self):
        Printer.print_rule("Init...")
        self._init_hook()
        self._init_state()
        self._init_config()

    @Printer.function_name
    def _init_hook(self):
        """create all hooks"""
        self.hook = {
            "on_system_get_argument": self.on_system_get_argument,
            "on_system_get_config": self.on_system_get_config,
            "on_system_init": self.on_system_init,

            # train
            "on_system_start_train": self.on_system_start_train,
            "on_user_get_dataset": self.on_user_get_dataset,
            "on_user_get_dataLoader": self.on_user_get_dataLoader,
            "on_user_get_model": self.on_user_get_model,
            "on_user_get_loss": self.on_user_get_loss,
            "on_user_get_optimizer": self.on_user_get_optimizer,
            "on_user_get_lr_scheduler": self.on_user_get_lr_scheduler,
            "on_user_get_checkpoint": self.on_user_get_checkpoint,
            "on_user_get_start_epoch": self.on_user_get_start_epoch,
            "on_system_train": self.on_system_train,
            "on_system_end_train": self.on_system_end_train,
            # epoch
            "on_system_start_epoch": self.on_system_start_epoch,
            "on_user_epoch": self.on_user_epoch,
            "on_system_end_epoch": self.on_system_end_epoch,
            # details
            "on_user_set_grad": self.on_user_set_grad,
            "on_user_calculate_loss": self.on_user_calculate_loss,
            "on_user_calculate_back_grad": self.on_user_calculate_back_grad,
            "on_user_update_parameter": self.on_user_update_parameter,
            "on_user_calculate_matric": self.on_user_calculate_matric,
            "on_user_save_checkpoint": self.on_user_save_checkpoint,
            # valid
            "on_system_start_valid": self.on_system_start_valid,
            "on_user_valid": self.on_user_valid,
            "on_system_end_valid": self.on_system_end_valid,
            # test
            "on_user_start_test": None,
            "on_user_test": None,
            "on_user_end_test": None,
        }

    @Printer.function_name
    def _init_state(self):
        """create all experience variable"""
        self.state = {
            "train_model": None,
            "valid_model": None,
            "train_iter": None,
            "valid_iter": None,
            "epoch": None,
            "max_epoch": None
        }

    @Printer.function_name
    def _init_config(self):
        """create config data"""
        self.config = {}

    @Printer.function_name
    def on_system_init(self):
        """init """
        pass

    @Printer.function_name
    def on_system_get_argument(self):
        """parse argument from console"""
        pass

    @Printer.function_name
    def on_system_get_config(self):
        """parse config file"""
        pass

    @Printer.function_name
    def on_user_get_dataset(self):
        """create dataset object"""
        pass

    @Printer.function_name
    def on_user_get_dataLoader(self):
        """create dataloader object"""
        pass

    @Printer.function_name
    def on_user_get_model(self):
        """create model object"""
        pass

    @Printer.function_name
    def on_user_get_loss(self):
        """get loss function object"""
        pass

    @Printer.function_name
    def on_user_get_optimizer(self):
        """get oprimizer object"""
        pass

    @Printer.function_name
    def on_user_get_lr_scheduler(self):
        pass

    @Printer.function_name
    def on_user_get_start_epoch(self):
        pass

    @Printer.function_name
    def on_user_get_checkpoint(self):
        pass

    @Printer.function_name
    def on_system_start_train(self):
        pass

    @Printer.function_name
    def on_system_start_train(self):
        pass

    @Printer.function_name
    def on_system_train(self):
        pass

    @Printer.function_name
    def on_system_end_train(self):
        pass

    # @Printer.function_name
    def on_system_start_epoch(self):
        pass

    # @Printer.function_name
    def on_user_epoch(self):
        pass

    # @Printer.function_name
    def on_system_end_epoch(self):
        pass

    @Printer.function_name
    def on_system_start_valid(self):
        pass

    def on_user_valid(self):
        pass

    @Printer.function_name
    def on_system_end_valid(self):
        pass

    @Printer.function_name
    def on_user_start_valid_epoch(self):
        pass

    @Printer.function_name
    def on_user_end_valid_epoch(self):
        pass

    @Printer.function_name
    def on_user_set_grad(self):
        pass

    @Printer.function_name
    def on_user_calculate_loss(self):
        pass

    @Printer.function_name
    def on_user_calculate_back_grad(self):
        pass

    @Printer.function_name
    def on_user_update_parameter(self):
        pass

    @Printer.function_name
    def on_user_calculate_matric(self):
        pass

    def on_user_save_checkpoint(self):
        pass

    def register_hook(self):
        pass
