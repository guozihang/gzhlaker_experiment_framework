'''
Descripttion: 
version: 
Author: Gzhlaker
Date: 2022-01-22 22:10:18
LastEditors: Andy
LastEditTime: 2022-01-28 16:53:35
'''
import argparse

from run.base_runner import base_runner

class base_tester(base_runner):
    '''define basic process'''
    def __init__(self):
        super(base_tester).__init__()

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    return parser

def set_config_data() -> None:
    pass

def init() -> None:
    get_parser

def main():
    parser = get_parser

if __name__ == "__main__":
    pass