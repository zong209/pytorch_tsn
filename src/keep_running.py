#! /usr/bin/env python
# -*- encoding:utf-8 -*-
'''
@File    : loop_func.py
@Time    : 2019/08/19 15:29:05
@Author  : Gaozong/260243
@Contact : 260243@gree.com.cn/zong209@163.com
'''
import time
"""
Usage: keep docker container running
"""
temp = 0
while True:
    print(temp)
    time.sleep(5)
    temp += 1
    if (temp > 100):
        temp = 0