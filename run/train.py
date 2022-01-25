'''
Descripttion: 
version: 
Author: Gzhlaker
Date: 2022-01-22 22:07:17
LastEditors: Andy
LastEditTime: 2022-01-25 20:54:35
'''
import sys
import time
sys.path.append(".")
from core.printer import Printer
def main():
    Printer.create_progressor(name="[red]Hello", total = 1000)
    while not Printer.is_progressor_finished(name="[red]Hello"):
        Printer.update_progressor(name="[red]Hello")
        time.sleep(0.02)
        

if __name__ == "__main__":
    main()