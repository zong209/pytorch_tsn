#! /usr/bin/env python
# -*- encoding:utf-8 -*-
'''
@File    : excute.py
@Time    : 2019/09/29 11:36:50
@Author  : Gaozong/260243
@Contact : 260243@gree.com.cn/zong209@163.com
'''

from torch.utils.data import DataLoader
from torchvision import transforms
from image_util import Rescale, RandomCrop, ToTensor, Normalize
from TSN import Tsn_network
from action_datasets import ActionDataset
from util import my_mkdir
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

CUDA_AVALIABLE = torch.cuda.is_available()


class TSN_EXEC(object):
    def __init__(self,
                 data_dir,
                 train_txt,
                 valid_txt,
                 output_nums,
                 rescale_size=256,
                 crop_size=224,
                 segement_nums=3,
                 new_length=1):
        self.data_dir = data_dir
        self.train_txt = train_txt
        self.valid_txt = valid_txt
        self.segement_nums = segement_nums
        self.new_length = new_length
        self.output_nums = output_nums
        self.rescale_size = rescale_size
        self.crop_size = crop_size
        # 初始化网络
        if CUDA_AVALIABLE:
            self.net = Tsn_network(segement_nums, output_nums).cuda()
        # 数据初始化
        composed = transforms.Compose([
            Rescale(self.rescale_size),
            RandomCrop(self.crop_size),
            Normalize(),
            ToTensor()
        ])
        # 测试数据加载
        self.valid_transformed_dataset = ActionDataset(
            os.path.join(self.data_dir, self.valid_txt),
            segement_nums=self.segement_nums,
            new_length=self.new_length,
            transform=composed)
        # 训练数据加载
        self.train_transformed_dataset = ActionDataset(
            os.path.join(self.data_dir, self.train_txt),
            segement_nums=self.segement_nums,
            new_length=self.new_length,
            transform=composed)

    def valid(self):
        net = self.net
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.valid_transformed_dataset:
                images, labels = data["input"], data["label"]
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.long()).sum().item()
        print('Accuracy of the network on the test samples: %d %%' %
              (100 * correct / total))

    def train(self,
              epochs,
              lr,
              batch_size,
              num_works,
              model_save_path,
              model_prefix,
              load_model=None,
              valid_interval=None,
              print_param=False):

        net = self.net

        dataloader = DataLoader(self.train_transformed_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_works)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

        if print_param:
            # Print net's state_dict
            print("Model's state_dict:")
            for param_tensor in net.state_dict():
                print("\t", param_tensor, "\t",
                      net.state_dict()[param_tensor].size())

        # 初始化保存模型文件夹
        my_mkdir(model_save_path)

        if load_model:
            assert (os.path.exists(load_model))
            print("load params from pre-training model")

        for epoch in range(epochs):
            loss = 0
            for i_batch, sample_batched in enumerate(dataloader):
                inputs, label = sample_batched["input"], sample_batched[
                    "label"]
                if CUDA_AVALIABLE:
                    outputs = net.forward(inputs.cuda())
                else:
                    outputs = net.forward(inputs)
                outputs = F.softmax(outputs)

                outputs = outputs.reshape([-1, self.output_nums])
                loss = criterion(outputs, label.long().flatten())

                # 反向传播
                loss.backward()
                optimizer.step()

                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i_batch + 1, loss.item() / batch_size))
            if valid_interval:
                if epoch > 0 and epoch % valid_interval == 0:
                    self.valid()

            if epoch > 0 and epoch % 10 == 0:
                save_path = os.path.join(
                    model_save_path, model_prefix + "_" + str(epoch) + "_" +
                    str(batch_size) + ".pt")
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dir': optimizer.state_dict,
                        'loss': loss
                    }, save_path)
        # ###########################################
        # #　测试部分

        # sample0 = transformed_dataset[0]
        # sample1 = transformed_dataset[1]
        # inputs = torch.cat([sample0['input'], sample1["input"]],
        #                    0).reshape([2, 9, 224, 224])
        # label = torch.cat([sample0['label'], sample1["label"]],
        #                   0).reshape([2, 1])
        # outputs = net.forward(inputs)
        # outputs = F.softmax(outputs)
        # # lable_vector = torch.zeros(1, self.output_nums).scatter_(
        # #     1, label.long(), 1)

        # # 定义损失函数和优化器
        # criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        # outputs = outputs.reshape([-1, self.output_nums])
        # loss = criterion(outputs, label.long().flatten())
        # print(loss)
        # loss.backward()
        # optimizer.step()


action_reg = TSN_EXEC(data_dir='data/zlfy/train',
                      train_txt="train.txt",
                      valid_txt="valid.txt",
                      output_nums=4,
                      rescale_size=256,
                      crop_size=224,
                      segement_nums=3,
                      new_length=1)

action_reg.train(epochs=100,
                 lr=0.001,
                 batch_size=32,
                 num_works=4,
                 model_save_path='models',
                 model_prefix='zlfy',
                 valid_interval=2)
