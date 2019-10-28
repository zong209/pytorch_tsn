#! ./.pyenv/bin/python
# -*- encoding:utf-8 -*-
'''
@File     : dataset.py
@Time     : 2019/09/30 14:38:04
@Author   : Gaozong/260243
@Contact  : 260243@gree.com.cn/zong209@163.com
@Describe : Define datasets class of action videos
'''
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def frame_cnts(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class ActionDataset(Dataset):
    def __init__(self,
                 root_path,
                 list_file,
                 num_segements,
                 new_length,
                 modality="RGB",
                 image_tmpl="{:0>6d}.jpg",
                 transform=None,
                 force_grayscale=False,
                 random_shift=True,
                 test_mode=False):
        """
        Args:
        root_path: data root dictory
        list_file: samples list, include [clip path, start frame, end frame, label]
        num_segements: split sample video to num_segements parts
        new_length: sample frame length in pre segement
        """
        self.root_path = root_path
        self.list_file = list_file
        self.num_segements = num_segements
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.force_grayscale = force_grayscale
        self.random_shift = random_shift
        self.test_mode = test_mode

        if self.modality == "RGBDiff":
            self.new_length += 1

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == "RGBDiff":
            img = cv2.imread(
                os.path.join(directory, self.image_tmpl.format(idx)))
            return [Image.fromarray(img).convert("RGB")]

    def _parse_list(self):
        self.video_list = [
            VideoRecord(x.strip().split()) for x in open(
                os.path.join(self.root_path, self.list_file)).readlines()
        ]

    def _sample_indices(self, record):

        average_duration = (record.frame_cnts - self.new_length +
                            1) // self.num_segements
        if average_duration > 0:
            offsets = np.multiply(list(range(
                self.num_segements)), average_duration) + np.random.randint(
                    average_duration, size=self.num_segements)
        elif record.frame_cnts > self.num_segements:
            offsets = np.sort(
                np.random.randint(record.frame_cnts - self.new_length + 1,
                                  size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segements, ))
        return offsets + 1

    def _get_val_indices(self, record):
        if record.frame_cnts > self.num_segements + self.new_length - 1:
            tick = (record.frame_cnts - self.new_length + 1) / float(
                self.num_segements)
            offsets = np.array([
                int(tick / 2.0 + tick * x) for x in range(self.num_segements)
            ])
        else:
            offsets = np.zeros((self.num_segements, ))
        return offsets + 1

    def _get_test_indices(self, record):

        tick = (record.frame_cnts - self.new_length + 1) / float(
            self.num_segments)

        offsets = np.array(
            [int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]
        if not self.test_mode:
            segement_indices = self._sample_indices(
                record) if self.random_shift else self._get_val_indices(record)
        else:
            segement_indices = self._get_test_indices(record)
        return self.get(record, segement_indices)

    def get(self, record, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.frame_cnts:
                    p += 1
        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)