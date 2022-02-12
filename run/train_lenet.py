'''
Descripttion: 
version: 
Author: Gzhlaker
Date: 2022-02-11 16:03:06
LastEditors: Andy
LastEditTime: 2022-02-12 13:11:30
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
    
    def on_user_get_dataset(self):
        trans = transforms.ToTensor()
        self.state["mnist_test"] = torchvision.datasets.FashionMNIST(
            root="./data",
            train = False,
            transform = trans,
            download = True
        )
        trans = transforms.ToTensor()
        self.state["mnist_train"] = torchvision.datasets.FashionMNIST(
            root="./data",
            train = True,
            transform = trans,
            download = True
        )
        return super().on_user_get_dataset()
    
    def on_user_get_dataLoader(self):
        self.state["train_iter"] = DataLoader(
            self.state["mnist_train"], 
            batch_size= self.config["BATCH_SIZE"],
            shuffle = True,
            num_workers = 12
        )
        self.state["test_iter"] = DataLoader(
            self.state["mnist_train"], 
            batch_size= self.config["BATCH_SIZE"],
            shuffle = True,
            num_workers = 12
        )
        return super().on_user_get_dataLoader()
    
    def on_user_get_model(self):
        self.state["net"] = LeNET()
        return super().on_user_get_model()

    def on_user_get_loss(self):
        self.state["loss"] = torch.nn.CrossEntropyLoss()
        return super().on_user_get_loss()
        
    def on_user_get_oprimizer(self):
        self.state["oprimizer"] = torch.optim.SGD(self.state["net"].parameters(), lr = self.config["LR"])
        return super().on_user_get_oprimizer()
    

    def on_user_set_grad(self):
        self.state["oprimizer"].zero_grad()
        return super().on_user_set_grad() 

    def on_user_update_parameter(self):
        return super().on_user_update_parameter()
    

    def on_user_epoch(self, epoch):
        j = 0
        for i, (X, Y) in enumerate(self.state["train_iter"]):
            j += 1
        Printer.create_progressor(name="[red]Epoch {}...".format(epoch), total = j)
        with Printer.get_progressor(name="[red]Epoch {}...".format(epoch)):
            for i, (X, y) in enumerate(self.state["train_iter"]):
                self.state["oprimizer"].zero_grad()
                self.state["net"].to(self.device)
                X, y = X.to(self.device), y.to(self.device)
                y_hat = self.self.state["net"](X)
                l = self.self.state["loss"](y_hat, y)
                l.backward()
                self.state["oprimizer"].step()
                Printer.update_progressor_without_progress(name="[red]Epoch {}...".format(epoch), advance=1)
        
    
    def on_user_valid(self):
        self.state["net"].eval()
        for X, y in self.self.state["train_iter"]:
            X = X.to(self.device)
            y = y.to(self.device)
            

    def on_user_calculate_matric(self):
        return super().on_user_calculate_matric()
    
    def on_user_save_checkpoint(self):
        dict = {}
        if(type(self.state["net"]) == torch.nn.Module): 
            dict["model_state_dict"] = self.state["net"].state_dict()
        if(type(self.state["oprimizer"]) == torch.optim.Optimizer):
            dict["optimizer_state_dict"] = self.state["oprimizer"].state_dict()
        torch.save(dict,  "./result/{}/{}".format(Printer.timestr, Printer.timestr))
        return super().on_user_save_checkpoint()
    
if __name__ == "__main__":
    trainer = train_lenet()
    trainer.run()