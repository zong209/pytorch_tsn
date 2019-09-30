#! /usr/bin/env python
# -*- encoding:utf-8 -*-
'''
@File    : action_datasets.py
@Time    : 2019/09/27 11:05:36
@Author  : Gaozong/260243
@Contact : 260243@gree.com.cn/zong209@163.com
'''
from __future__ import print_function, division

import os
import math
import numpy as np
from skimage import io
from torch.utils.data import Dataset


class ActionDataset(Dataset):
    """Tempral Action Dataset"""
    def __init__(self, file_txt, segement_nums, new_length, transform=None):
        self.file_txt = file_txt
        file = open(file_txt, 'r')
        self.data_lines = [
            line for line in file.readlines() if line.strip() != ""
        ]
        self.segement_nums = segement_nums
        self.new_length = new_length
        self.transform = transform

    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, idx):
        data_idx = self.data_lines[idx].split()
        sample_img = []
        frame_dir = data_idx[0]
        frame_cnts = int(data_idx[1])
        ava_duration = math.floor(frame_cnts / self.segement_nums)
        offset = math.floor(ava_duration / self.new_length)
        for i in range(self.segement_nums):
            start_frame = np.random.randint(offset) + 1
            for j in range(self.new_length):
                image_idx = start_frame + j * offset
                image_idx = ("{:0>6d}.jpg").format(image_idx)
                img = io.imread(os.path.join(frame_dir, image_idx))
                sample_img.append(img)
        sample = {"input": np.array(sample_img), "label": data_idx[2]}
        if self.transform:
            sample = self.transform(sample)
        return sample
