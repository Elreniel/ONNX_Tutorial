# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 10:43:16 2021

@author: bcosk
"""

from multiprocessing import Process
import multiprocessing
import math

def calc():
    print("asd")
    for i in range(0, 4000000):
        math.sqrt(i)

if __name__ ==  '__main__':
    processes = []
    for i in range(multiprocessing.cpu_count()):
        print('registering process %d' % i)
        processes.append(Process(target=calc))
    for process in processes:
        process.start()
    for process in processes:
        process.join()