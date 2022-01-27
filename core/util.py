'''
Descripttion: 
version: 
Author: Gzhlaker
Date: 2022-01-22 22:07:32
LastEditors: Andy
LastEditTime: 2022-01-27 16:48:16
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
    def get_yaml_data(filename: str) -> dict:
        '''
        get yaml file data
        '''
        with open(filename) as file:
           return yaml.load(file.read(), Loader=yaml.SafeLoader)
