# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 17:10:22 2018

@author: chen jin
"""
name = 1
list = []
while name:
    print("if end press enter")
    name = input("type your name:")
    if name:
        list.append([i for i in name.split()])
print(list)