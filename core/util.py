'''
Descripttion: 
version: 
Author: Gzhlaker
Date: 2022-01-22 22:07:32
LastEditors: Andy
LastEditTime: 2022-01-25 21:02:26
'''
# import torch.cuda
import json
from rich.console import Console
import yaml

class Util:
    # rich console
    console = None
    # @classmethod
    # def check_environment():
    #     return torch.cuda.is_avilable()
    @staticmethod
    def init_rich() -> Console:
        '''
        init rich
        '''
        rich_config = Util.get_yaml_data("./config/base_rich_config.yml")
        Util.console = Console(color_system = rich_config['COLOR_SYSTEM'])
        return Util.console
    @staticmethod
    def get_yaml_data(filename: str) -> dict:
        '''
        get yaml file data
        '''
        with open(filename) as file:
           return yaml.load(file.read(), Loader=yaml.SafeLoader)
