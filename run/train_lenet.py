'''
Descripttion: 
version: 
Author: Gzhlaker
Date: 2022-02-11 16:03:06
LastEditors: Andy
LastEditTime: 2022-02-11 20:18:13
'''

import sys

import torch
import torchvision
sys.path.append(".")
from torchvision import transforms
from torch.utils.data import DataLoader
from base_train import base_trainer
from models.LeNET import LeNET
from core.printer import Printer
class train_lenet(base_trainer):
    def __init__(self):
        super().__init__()
        self.batch_size = 256
        self.lr = 0.1
        self.train_epoch = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def on_get_dataset(self):
        trans = transforms.ToTensor()
        self.mnist_test = torchvision.datasets.FashionMNIST(
            root="./data",
            train = False,
            transform = trans,
            download = True
        )
        trans = transforms.ToTensor()
        self.mnist_train = torchvision.datasets.FashionMNIST(
            root="./data",
            train = True,
            transform = trans,
            download = True
        )
        return super().on_get_dataset()
    
    def on_get_dataLoader(self):
        self.train_iter = DataLoader(
            self.mnist_train, 
            batch_size= self.batch_size,
            shuffle = True,
            num_workers = 12
        )
        self.test_iter = DataLoader(
            self.mnist_train, 
            batch_size= self.batch_size,
            shuffle = True,
            num_workers = 12
        )
        return super().on_get_dataLoader()
    
    def on_get_model(self):
        self.net = LeNET()
        return super().on_get_model()
    def on_get_loss(self):
        self.loss = torch.nn.CrossEntropyLoss()
        return super().on_get_loss()
    def on_get_oprimizer(self):
        self.oprimizer = torch.optim.SGD(self.net.parameters(), lr = self.lr)
        return super().on_get_oprimizer()
    
    def on_update_parameter(self):
        return super().on_update_parameter()

    def on_epoch(self):
        for i, (X, y) in enumerate(self.train_iter):
            self.oprimizer.zero_grad()
            self.net.to(self.device)
            X, y = X.to(self.device), y.to(self.device)
            y_hat = self.net(X)
            l = self.loss(y_hat, y)
            l.backward()
            self.oprimizer.step()
        

    def on_calculate_matric(self):
        return super().on_calculate_matric()
    
if __name__ == "__main__":
    trainer = train_lenet()
    trainer.run()