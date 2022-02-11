'''
Descripttion: 
version: 
Author: Gzhlaker
Date: 2022-01-22 22:07:17
LastEditors: Andy
LastEditTime: 2022-02-12 00:32:17
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
        self.hook["on_parse_argument"]()
        self.hook["on_get_config"]()
        self.hook["on_init"]()
        self.hook["on_get_dataset"]()
        self.hook["on_get_dataLoader"]()
        self.hook["on_get_model"]()
        self.hook["on_get_loss"]()
        self.hook["on_get_oprimizer"]()

    @Printer.function_name
    def on_init(self):
        self._on_init_random()
        self._on_init_device()

    def _on_init_random(self):
        seed = 100
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
    
    def _train(self):
        Printer.print_rule("Training...")
        self.hook["on_start_train"]()
        self.hook["on_train"]()
        self.hook["on_end_train"]()
    
    def _valid(self):
        Printer.print_rule("Validing...")
        self.hook["on_start_valid"]()
        self.hook["on_valid"]()
        self.hook["on_end_valid"]()

    @Printer.function_name
    def on_parse_argument(self):
        Printer.print_log("Create Parser")
        self.parser = argparse.ArgumentParser()
        Printer.print_log("Create Argument")
        self.parser.add_argument('--config', '-cfg', default='')
        Printer.print_log("Parse Argument")
        self.args = self.parser.parse_args()
        for key in list(vars(self.args).keys()):
            Printer.print_log("-- {}: {}".format(key, vars(self.args)[key]))
    
    def on_start_epoch(self):
        self.hook["on_set_grad"]()
        self.hook["on_calculate_loss"]()
        self.hook["on_calculate_back_grad"]()
        self.hook["on_update_parameter"]()
        
    def on_end_epoch(self):
        self.hook["on_calculate_matric"]()
    def on_end_train(self):
        self.on_save_checkpoints()
        return super().on_end_train()
    def on_save_checkpoints(self):
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
        },  "./result/{}/{}".format(Printer.timestr, Printer.timestr))
        return super().on_save_checkpoints()
def main():
    trainer = base_trainer()
    trainer.run()
if __name__ == "__main__":
    main()