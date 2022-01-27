'''
Descripttion: 
version: 
Author: Gzhlaker
Date: 2022-01-22 22:07:17
LastEditors: Andy
LastEditTime: 2022-01-26 12:21:15
'''
import time
import sys
sys.path.append(".")

from core.printer import Printer
    
class Trainer:
    
    def test(self):
        Printer.print_rule("Config")
        Printer.create_progressor(name="[red]Hello", total = 1000)
        with Printer.get_progressor(name="[red]Hello"):
            while not Printer.is_progressor_finished(name="[red]Hello"):
                Printer.update_progressor_without_progress(name="[red]Hello")
                time.sleep(0.02)
    
    def define_data_lodaer(self):
        pass

    def define_optimizer(self):
        pass

    def define_model(self):
        pass

    def define_learn_rate_adjuster(self):
        pass

    def define_saver(self):
        pass

    def define_log_writer(self):
        pass
    
    def train():
        pass

    def on_start_train(self):
        pass

    def on_start_epoch(self):
        pass


def main():
    trainer = Trainer()
    trainer.test()
if __name__ == "__main__":
    main()