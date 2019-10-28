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


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
