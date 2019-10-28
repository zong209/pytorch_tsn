#! .pyenv/bin/python
# -*- encoding:utf-8 -*-
'''
@File     : load_pytorch.py
@Time     : 2019/10/16 14:59:23
@Author   : Gaozong/260243
@Contact  : 260243@gree.com.cn/zong209@163.com
@Describe : load pre-training weights
'''

import torch

state = {}
pretrained_dict = torch.load('models/bn_inception-9f5701afb96c8044.pth')
myself_dict = torch.load('models/1571209390_zlfy_40_32_0.80800.pt')

for i, (k, v) in enumerate(pretrained_dict.items()):
    print(i, k, v.size())

for i, (k, v) in enumerate(myself_dict["model_state_dict"].items()):
    print(i, k, v.size())
