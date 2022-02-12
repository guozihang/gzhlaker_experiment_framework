'''
Descripttion: 
version: 
Author: Gzhlaker
Date: 2022-01-22 22:07:32
LastEditors: Andy
LastEditTime: 2022-02-12 11:59:23
'''
# import torch.cuda
import json
from rich.console import Console
import yaml

class Util:
    '''this is a static class, hold some useful method'''
    @staticmethod
    def get_yaml_data(filename: str) -> dict:
        '''
        get yaml file data
        '''
        with open(filename) as file:
           return yaml.load(file.read(), Loader=yaml.SafeLoader)
    
    def load_config(filename):
        return Util.get_yaml_data(filename)

    @staticmethod
    def save_config(filename):
        pass
