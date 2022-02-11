'''
Descripttion: 
version: 
Author: Gzhlaker
Date: 2022-02-11 15:56:56
LastEditors: Andy
LastEditTime: 2022-02-11 19:21:16
'''
from turtle import forward
import torch
from torch import nn

class ReShape(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x):
        return x.view(-1, 1, 28, 28)

class LeNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ReShape(),
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )
    
    def forward(self, x):
        return self.layers(x)