'''
Descripttion: 
version: 
Author: Gzhlaker
Date: 2022-01-22 22:07:17
LastEditors: Andy
LastEditTime: 2022-02-11 13:34:54
'''
import time
import sys
import argparse
import random
import torch

from base_runner import base_runner
sys.path.append(".")
from core.util import Util
from core.printer import Printer
    
class base_trainer(base_runner):
    '''define basic process'''
    def __init__(self):
        super().__init__()
        
    def train(self):
        Printer.print_rule("Init...")
        self.hook["on_init"]()
        self.hook["on_parse_argument"]()
        self.hook["on_get_config"]()
        self.hook["on_get_dataset"]()
        self.hook["on_get_dataLoader"]()
        self.hook["on_get_model"]()
        self.hook["on_get_loss"]()
        self.hook["on_get_oprimizer"]()
        self.hook["on_start_train"]()
        Printer.print_rule("Training...")
        Printer.create_progressor(name="[red]Train...", total = 10)
        with Printer.get_progressor(name="[red]Train..."):
            while not Printer.is_progressor_finished(name="[red]Train..."):
                self.hook["on_start_epoch"]()
                Printer.update_progressor_without_progress(name="[red]Train...", advance=1)
                self.hook["on_end_epoch"]()
                time.sleep(1)
        self.hook["on_end_train"]()
    @Printer.function_name
    def on_init(self):
        self._on_init_random()

    def _on_init_random(self):
        seed = 100
        Printer.print_log("seed: {}".format(seed))
        random.seed(seed)
        Printer.print_log("set random seed: {}".format(seed))
        torch.manual_seed(seed)
        Printer.print_log("set torch seed: {}".format(seed))

    @Printer.function_name
    def on_parse_argument(self):
        Printer.print_log("Create Parser")
        self.parser = argparse.ArgumentParser()
        Printer.print_log("Create Argument")
        self.parser.add_argument('--config', '-cfg', default='')
        Printer.print_log("Parse Argument")
        self.args = self.parser.parse_args()
        
    
def main():
    trainer = base_trainer()
    trainer.train()
if __name__ == "__main__":
    main()