'''
Descripttion: 
version: 
Author: Gzhlaker
Date: 2022-02-11 22:46:53
LastEditors: Andy
LastEditTime: 2022-02-11 22:46:54
'''
'''
Descripttion: 
version: 
Author: Gzhlaker
Date: 2021-11-25 15:31:33
LastEditors: Andy
LastEditTime: 2022-02-11 22:45:15
'''
import itchat, time
import datetime
import threading
from itchat.content import *

class gzhlaker_wechat_talker:
    @staticmethod
    def init():
        itchat.auto_login(enableCmdQR=2, hotReload=False)
    
    @staticmethod
    def send(msg, nick_name):
        for user in itchat.get_friends():
            print(user["NickName"])
            if user["NickName"] == nick_name:
                print(user["UserName"])
                itchat.send_msg(msg, user["UserName"])

class gzhaler_message:
    def __init__(self):
        self._message = "";
    def add_message(self, message):
        self._message = self._message + message
    def change_line(self):
        self._message = self._message + "\n"
    def add_line(self, string):
        self.add_message(string)
        self.change_line()
    def __str__(self):
        return self._message

def main():
    gzhlaker_wechat_talker.init()
    msg = gzhaler_message()
    msg.add_line("本次程序运行结果已完成")
    msg.add_line("参数X的值为1")
    msg.add_line("参数Y的值为2")
    print(msg._message)
    gzhlaker_wechat_talker.send(str(msg), "不温")
    
if __name__ == "__main__":
    main()