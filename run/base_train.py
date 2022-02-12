'''
Descripttion: 
version: 
Author: Gzhlaker
Date: 2022-01-22 22:07:17
LastEditors: Andy
LastEditTime: 2022-02-12 13:00:09
'''
import argparse
import random
import torch
import torch.cuda

import sys
sys.path.append(".")

from base_runner import base_runner
from core.util import Util
from core.printer import Printer
    
class base_trainer(base_runner):
    '''define basic train process'''
    def __init__(self):
        super().__init__()
        
    def run(self):
        self._init()
        self._train()
        self._valid()

    def _init(self):
        Printer.print_rule("Init...")
        self.hook["on_system_get_argument"]()
        self.hook["on_system_get_config"]()
        self.hook["on_system_init"]()

    def on_system_get_argument(self):
        '''save all console arguments to config'''
        Printer.print_log("Create Parser")
        self.parser = argparse.ArgumentParser()
        Printer.print_log("Create Argument")
        self.parser.add_argument('--config', '-cfg', default='')
        Printer.print_log("Parse Argument")
        self.args = self.parser.parse_args()
        for key in list(vars(self.args).keys()):
            Printer.print_log("-- {}: {}".format(key, vars(self.args)[key]))
            self.config[key] = vars(self.args)[key]
        return super().on_system_get_argument()
    
    def on_system_get_config(self):
        self.config = Util.get_yaml_data(self.config["config"])
        Printer.log.info(self.config)
        return super().on_system_get_config()

    def on_system_init(self):
        self._on_init_random()
        self._on_init_device()
        self._on_init_state()
        return super().on_system_init()

    def _on_init_random(self):
        seed = self.config["SEED"]
        Printer.print_log("seed: {}".format(seed))
        random.seed(seed)
        Printer.print_log("set random seed: {}".format(seed))
        torch.manual_seed(seed)
        Printer.print_log("set torch seed: {}".format(seed))

    def _on_init_device(self):
        if torch.cuda.is_available():
            Printer.print_log("Using GPU")
        else:
            Printer.print_log("Using CPU") 

    def _on_init_state(self):
        self.state["max_epoch"] = self.config["MAX_EPOCH"]


    def _train(self):
        Printer.print_rule("Training...")
        self.hook["on_system_start_train"]()
        self.hook["on_system_train"]()
        self.hook["on_system_end_train"]()

    def on_system_start_train(self):
        self.hook["on_user_get_dataset"]()
        self.hook["on_user_get_dataLoader"]()
        self.hook["on_user_get_model"]()
        self.hook["on_user_get_loss"]()
        self.hook["on_user_get_oprimizer"]()
        self.hook["on_user_get_checkpoint"]()
        return super().on_system_start_train()

    def on_system_train(self):
        for epoch in range(self.config["TRAIN_EPOCH"]):
            self.hook["on_start_epoch"]()
            self.hook["on_epoch"]()
            self.hook["on_end_epoch"]()
            self.state["epoch"] = epoch
            
    def on_system_end_train(self):
        return super().on_system_end_train()
    
    def on_system_start_epoch(self):
        self.hook["on_user_set_grad"]()
        self.hook["on_user_calculate_loss"]()
        self.hook["on_user_calculate_back_grad"]()
        self.hook["on_user_update_parameter"]()
        
    def on_system_end_epoch(self):
        self.hook["on_user_save_checkpoints"]()
        self.hook["on_user_calculate_matric"]()

    def _valid(self):
        Printer.print_rule("Validing...")
        self.hook["on_system_start_valid"]()
        self.hook["on_user_valid"]()
        self.hook["on_system_end_valid"]()

    def on_system_start_valid(self):
        return super().on_system_start_valid()
    
    def on_system_end_valid(self):
        return super().on_system_end_valid()

def main():
    trainer = base_trainer()
    trainer.run()
if __name__ == "__main__":
    main()