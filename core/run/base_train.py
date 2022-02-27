"""
Description:
version:
Author: Gzhlaker
Date: 2022-01-22 22:07:17
LastEditors: Andy
LastEditTime: 2022-02-12 13:14:41
"""

import argparse
import os
import random
import torch
import torch.cuda
import wandb

import sys

sys.path.append("./")

from core.run.base_runner import BaseRunner
from core.util.util import Util
from core.manager.printer import Printer


class BaseTrainer(BaseRunner):
    """define basic train process"""

    def __init__(self):
        super().__init__()

    def run_train(self):
        self._init()
        self._train()

    def run_valid(self):
        pass

    def _init(self):
        self.hook["on_system_get_argument"]()
        self.hook["on_system_get_config"]()
        self.hook["on_system_init"]()

    def on_system_get_argument(self):
        """save all console arguments to config"""
        Printer.print_log("Create Parser")
        self.state["parser"] = argparse.ArgumentParser()
        Printer.print_log("Create Argument")
        self.state["parser"].add_argument('--config', '-cfg', default='')
        self.state["parser"].add_argument('--log_time', default='')
        Printer.print_log("Parse Argument")
        self.state["args"] = self.state["parser"].parse_args()
        Printer.print_panle(vars(self.state["args"]), title="Args")
        return super().on_system_get_argument()

    def on_system_get_config(self):
        self.config = Util.get_yaml_data(self.state["args"].config)
        Printer.print_panle(self.config, title="Config")
        return super().on_system_get_config()

    def on_system_init(self):
        self._on_init_random()
        self._on_init_device()
        self._on_init_state()
        self._on_init_path()
        self._on_init_wandb()
        return super().on_system_init()

    def _on_init_random(self):
        """fixed random seed"""
        seed = self.config["seed"]
        Printer.print_log("seed: {}".format(seed))
        random.seed(seed)
        Printer.print_log("set random seed: {}".format(seed))
        torch.manual_seed(seed)
        Printer.print_log("set torch seed: {}".format(seed))

    def _on_init_device(self):
        """set device"""
        if torch.cuda.is_available():
            Printer.print_log("Using GPU")
            self.state["device"] = 'cuda'
        else:
            Printer.print_log("Using CPU")
            self.state["device"] = 'cpu'

    def _on_init_state(self):
        self.state["max_epoch"] = self.config["solver"]["epochs"]

    def _on_init_path(self):
        self.state["save_dir"] = os.path.join(os.getcwd(), "result", Printer.timestr)
        Printer.print_panle(self.state["save_dir"], title="Save Dir")

    def _on_init_wandb(self):
        if self.config["wandb"]:
            wandb.init(
                project='',
                name='',
            )

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
        self.hook["on_user_get_optimizer"]()
        self.hook["on_user_get_lr_scheduler"]()
        self.hook["on_user_get_checkpoint"]()
        return super().on_system_start_train()

    def on_system_train(self):
        for epoch in range(self.config["solver"]["start_epoch"], self.config["solver"]["epochs"]):
            self.state["epoch"] = epoch
            self.hook["on_system_start_epoch"]()
            self.hook["on_user_epoch"]()
            self.hook["on_system_end_epoch"]()
            self._valid()

    def on_system_end_train(self):
        return super().on_system_end_train()

    def on_system_start_epoch(self):
        self.hook["on_user_set_grad"]()
        self.hook["on_user_calculate_loss"]()
        self.hook["on_user_calculate_back_grad"]()
        self.hook["on_user_update_parameter"]()

    def on_system_end_epoch(self):

        self.hook["on_user_calculate_matric"]()

    def _valid(self):
        Printer.print_rule("Validing...")
        self.hook["on_system_start_valid"]()
        self.hook["on_user_valid"]()
        self.hook["on_system_end_valid"]()

    def on_system_start_valid(self):
        return super().on_system_start_valid()

    def on_system_end_valid(self):
        self.hook["on_user_save_checkpoint"]()
        return super().on_system_end_valid()
