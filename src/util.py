#! /usr/bin/env python
# -*- encoding:utf-8 -*-
'''
@File    : util.py
@Time    : 2019/09/29 10:02:41
@Author  : Gaozong/260243
@Contact : 260243@gree.com.cn/zong209@163.com
'''

import os


def my_mkdir(path):
    """Create Folder"""
    if not os.path.exists(path):
        os.mkdir(path)
    return
