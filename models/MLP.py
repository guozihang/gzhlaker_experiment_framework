'''
Descripttion: 
version: 
Author: Gzhlaker
Date: 2022-01-31 17:24:15
LastEditors: Andy
LastEditTime: 2022-02-11 14:32:44
'''
from torch import nn
from torch import functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
    
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))