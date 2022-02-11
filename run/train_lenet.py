'''
Descripttion: 
version: 
Author: Gzhlaker
Date: 2022-02-11 16:03:06
LastEditors: Andy
LastEditTime: 2022-02-12 00:03:36
'''

import sys

import torch
import torchvision
sys.path.append(".")
from torchvision import transforms
from torch.utils.data import DataLoader
from core.printer import Printer
from models.LeNET import LeNET
from base_train import base_trainer
from core.util import Util

class train_lenet(base_trainer):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def on_get_config(self):
        self.config = Util.get_yaml_data("./config/LENET_TRAINER.yml")
        Printer.log.info(self.config)
        return super().on_get_config()
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
            batch_size= self.config["BATCH_SIZE"],
            shuffle = True,
            num_workers = 12
        )
        self.test_iter = DataLoader(
            self.mnist_train, 
            batch_size= self.config["BATCH_SIZE"],
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
        self.oprimizer = torch.optim.SGD(self.net.parameters(), lr = self.config["LR"])
        return super().on_get_oprimizer()
    
    def on_update_parameter(self):
        return super().on_update_parameter()
    
    def on_train(self):
        for epoch in range(self.config["TRAIN_EPOCH"]):
            self.hook["on_start_epoch"]()
            self.hook["on_epoch"](epoch)
            self.hook["on_end_epoch"]()

    def on_set_grad(self):
        self.oprimizer.zero_grad()
        return super().on_set_grad() 

    def on_epoch(self, epoch):
        j = 0
        for i, (X, Y) in enumerate(self.train_iter):
            j += 1
        Printer.create_progressor(name="[red]Epoch {}...".format(epoch), total = j)
        with Printer.get_progressor(name="[red]Epoch {}...".format(epoch)):
            for i, (X, y) in enumerate(self.train_iter):
                self.oprimizer.zero_grad()
                self.net.to(self.device)
                X, y = X.to(self.device), y.to(self.device)
                y_hat = self.net(X)
                l = self.loss(y_hat, y)
                l.backward()
                self.oprimizer.step()
                Printer.update_progressor_without_progress(name="[red]Epoch {}...".format(epoch), advance=1)
        
    
    def on_valid(self):
        self.net.eval()
        for X, y in self.train_iter:
            X = X.to(self.device)
            y = y.to(self.device)
            

    def on_calculate_matric(self):
        return super().on_calculate_matric()
    
if __name__ == "__main__":
    trainer = train_lenet()
    trainer.run()