'''
Descripttion: 
version: 
Author: Gzhlaker
Date: 2022-01-31 17:34:15
LastEditors: Andy
LastEditTime: 2022-01-31 17:36:48
'''
from torch import nn
from torch import functional as F

class gzhlaker_sequential(nn.Module):
    def __init__(self, *args):
        super(gzhlaker_sequential).init()
        for block in args:
            self.modules[block] = block
        
    def forward(self, X):
        for block in self.modules.values():
            X = block(X)
        return X