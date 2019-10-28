#! /usr/bin/env python
# -*- encoding:utf-8 -*-
'''
@File    : main.py
@Time    : 2019/09/29 11:36:50
@Author  : Gaozong/260243
@Contact : 260243@gree.com.cn/zong209@163.com
'''
import os
import torch
import time
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from image_util import Stack, ToTorchFormatTensor, GroupNormalize,\
    GroupScale, GroupCenterCrop
from model import TSN
from dataset import ActionDataset
from util import my_mkdir, AverageMeter
from torch.nn.utils import clip_grad_norm


# CUDA_AVALIABLE = False
CUDA_AVALIABLE = torch.cuda.is_available()


class TSN_INIT(object):
    def __init__(self,
                 data_dir,
                 train_txt,
                 valid_txt,
                 output_nums,
                 modality="RGB",
                 dropout=0.8,
                 partialbn=True,
                 segement_nums=3,
                 resume_file=None):

        self.data_dir = data_dir
        self.train_txt = train_txt
        self.valid_txt = valid_txt
        self.segement_nums = segement_nums
        self.output_nums = output_nums
        self.resume_file = resume_file
        self.start_epoch = 0
        self.partialbn = partialbn
        # 初始化网络
        if CUDA_AVALIABLE:
            self.net = TSN(output_nums,
                           segement_nums,
                           modality,
                           consensus_type="avg",
                           dropout=dropout,
                           partial_bn=partialbn).cuda()
            cudnn.benchmark = True
        else:
            self.net = TSN(output_nums,
                           segement_nums,
                           modality,
                           consensus_type="avg",
                           dropout=dropout,
                           partial_bn=partialbn)

        self.rescale_size = self.net.scale_size
        self.crop_size = self.net.crop_size
        self.input_mean = self.net.input_mean
        self.input_std = self.net.input_std

        policies = self.net.get_optim_policies()
        train_augmentation = self.net.get_augmentation()

        # 加载预训练模型
        if resume_file:
            if os.path.isfile(resume_file):
                print(("=> loading checkpoint '{}'".format(resume_file)))
                checkpoint = torch.load(resume_file)
                self.start_epoch = checkpoint['epoch']
                self.net.load_state_dict(checkpoint['model_state_dict'])
                print(("=> loaded checkpoint '{}' (epoch {})".format(
                    resume_file, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(resume_file)))

        # 数据初始化/训练数据增强
        composed_train = transforms.Compose([
            train_augmentation,
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(mean=self.input_mean, std=self.input_std)
        ])
        composed_valid = transforms.Compose([
            GroupScale(256),
            GroupCenterCrop(224),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(mean=self.input_mean, std=self.input_std)
        ])

        if modality == 'RGB':
            self.new_length = 1
        elif modality in ['Flow', 'RGBDiff']:
            self.new_length = 5

        # 测试数据加载
        self.valid_transformed_dataset = ActionDataset(
            root_path=self.data_dir,
            list_file=self.valid_txt,
            num_segements=self.segement_nums,
            new_length=self.new_length,
            transform=composed_valid)

        # 训练数据加载
        self.train_transformed_dataset = ActionDataset(
            root_path=self.data_dir,
            list_file=self.train_txt,
            num_segements=self.segement_nums,
            new_length=self.new_length,
            transform=composed_train)

        self.criterion = nn.CrossEntropyLoss().cuda()

        for group in policies:
            print(
                ('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
                    group['name'], len(group['params']), group['lr_mult'],
                    group['decay_mult'])))

    def valid(self, print_freq=2, batch_size=32, shuffle=True, num_workers=4):
        batch_time = AverageMeter()
        losses = AverageMeter()
        correct = AverageMeter()
        net = self.net
        net.eval()

        valid_dataloader = DataLoader(self.valid_transformed_dataset,
                                      batch_size=batch_size,
                                      shuffle=shuffle,
                                      num_workers=num_workers)

        with torch.no_grad():
            end = time.time()
            for i, data in enumerate(valid_dataloader):
                inputs, labels = data
                if CUDA_AVALIABLE:
                    outputs = net.forward(inputs.cuda())
                    labels = labels.long().flatten().cuda()
                else:
                    outputs = net.forward(inputs)
                    labels = labels.long().flatten()
                outputs = outputs.reshape([-1, self.output_nums])
                loss = self.criterion(outputs, labels)
                # 更新统计数据
                losses.update(loss.item(), inputs.size(0))
                _, predicted = torch.max(outputs.data, 1)
                # 计算准确率
                correct.update(
                    (predicted == labels.long()).sum().item() / len(labels),
                    inputs.size(0))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                if i % print_freq == 0 or i * batch_size >= len(
                        self.valid_transformed_dataset):
                    print(('Test: [{0}/{1}]\t'
                           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                               i,
                               len(valid_dataloader),
                               batch_time=batch_time,
                               loss=losses,
                               top1=correct)))
        return correct.avg

    def test(self, test_file):
        net = self.net
        net.eval()
        composed_valid = transforms.Compose([
            GroupScale(256),
            GroupCenterCrop(224),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(mean=self.input_mean, std=self.input_std)
        ])
        # 测试数据加载
        test_transformed_dataset = ActionDataset(
            root_path=self.data_dir,
            list_file=test_file,
            num_segements=self.segement_nums,
            new_length=self.new_length,
            transform=composed_valid)
        test_dataloader = DataLoader(test_transformed_dataset, batch_size=1)
        for epoch in range(10):
            for i, (inputs, label) in enumerate(test_dataloader):
                if CUDA_AVALIABLE:
                    outputs = net.forward(inputs.cuda())
                else:
                    outputs = net.forward(inputs)
                outputs = outputs.reshape([-1, self.output_nums])
                _, predicted = torch.max(outputs.data, 1)
                print("[{}:{}] Test result:{} Truth:{} {}".format(
                    epoch + 1, i, predicted.item(), label,
                    predicted.item() == label))

    def train(self,
              epochs,
              lr,
              lr_steps,
              weight_decay,
              batch_size,
              num_workers,
              model_save_path,
              model_prefix,
              clip_gradient=4,
              valid_interval=None,
              print_param=False,
              print_freq=4):

        net = self.net
        policies = net.parameters()
        valid_precision = 0

        dataloader = DataLoader(self.train_transformed_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers)

        if self.partialbn:
            net.partialBN(True)
        else:
            net.partialBN(False)

        # 定义优化器
        optimizer = optim.SGD(policies,
                              lr=lr,
                              momentum=0.9,
                              weight_decay=weight_decay)

        # 打印网络结构
        if print_param:
            # Print net's state_dict
            print("Model's state_dict:")
            for param_tensor in net.state_dict():
                print("\t", param_tensor, "\t",
                      net.state_dict()[param_tensor].size())

        # 初始化保存模型文件夹
        my_mkdir(model_save_path)

        def adjust_learning_rate(optimizer, epoch, lr, lr_steps, weight_decay):
            """Sets the learning rate to the initial LR 
            decayed by 10 every 30 epochs"""
            decay = 0.1**(sum(epoch >= np.array(lr_steps)))
            lr = lr * decay
            decay = weight_decay
            for param_group in optimizer.param_groups:
                if hasattr(param_group, 'lr_mult'):
                    param_group['lr'] = lr * param_group['lr_mult']
                if hasattr(param_group, 'decay_mult'):
                    param_group[
                        'weight_decay'] = decay * param_group['decay_mult']

        for epoch in range(self.start_epoch + 1, epochs):
            if epoch > 10:
                net.partialBN(True)
                net.train()
                policies = net.get_optim_policies()
                optimizer = optim.SGD(policies,
                                      lr=lr,
                                      momentum=0.9,
                                      weight_decay=weight_decay)
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            correct = AverageMeter()
            end = time.time()

            adjust_learning_rate(optimizer, epoch, lr, lr_steps, weight_decay)

            for i_batch, sample_batched in enumerate(dataloader):

                # measure data loading time
                data_time.update(time.time() - end)

                inputs, labels = sample_batched

                if CUDA_AVALIABLE:
                    outputs = net.forward(inputs.cuda())
                    labels = labels.long().flatten().cuda()
                else:
                    outputs = net.forward(inputs)
                    labels = labels.long().flatten()

                outputs = outputs.reshape([-1, self.output_nums])
                loss = self.criterion(outputs, labels)

                # 更新统计数据
                losses.update(loss.item(), inputs.size(0))
                _, predicted = torch.max(outputs.data, 1)
                # 计算准确率
                correct.update(
                    (predicted == labels.long()).sum().item() / len(labels),
                    inputs.size(0))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # compute gradient and do SGD step
                optimizer.zero_grad()
                # 反向传播
                loss.backward()

                # 梯度截断
                if clip_gradient is not None:
                    total_norm = clip_grad_norm(net.parameters(),
                                                clip_gradient)
                    if total_norm > clip_gradient:
                        print("clipping gradient: {} with coef {}".format(
                            total_norm, clip_gradient / total_norm))

                optimizer.step()
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                if i_batch % print_freq == 0:
                    print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                               epoch,
                               i_batch,
                               len(dataloader),
                               batch_time=batch_time,
                               data_time=data_time,
                               loss=losses,
                               top1=correct,
                               lr=optimizer.param_groups[-1]['lr'])))

            if valid_interval:
                if (epoch + 1) % valid_interval == 0:
                    valid_precision = self.valid()

            # 保存网络
            if (epoch > 0 and epoch % 10 == 0) or epoch == epochs - 1:
                save_path = os.path.join(
                    model_save_path, "{:d}_{}_{:d}_{:d}_{:.5f}.pt".format(
                        int(time.time()), model_prefix, epoch, batch_size,
                        valid_precision))
                print("[INFO] Save weights to " + save_path)
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dir': optimizer.state_dict,
                        'loss': loss
                    }, save_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='/app/data/train')
    parser.add_argument('--output_nums', type=int, default=4, help='numbers of classes')
    parser.add_argument('--segement_nums', type=int, default=3,help="numbers of segements in per video's clip ")
    parser.add_argument('--resume_file', type=str,default=None, help="pre-training weights file")
    parser.add_argument('--epochs', type=int, default=200,help="epochs of training")
    parser.add_argument('--lr', type=float,default=0.001,help="base learning rate")
    parser.add_argument('--lr_steps', type=list, help="muti-steps learning rate", default=[50,100])
    parser.add_argument('--weight_decay', type=float, help="weight decay", default=0.005)
    parser.add_argument('--batch_size', type=int, help="batch size of samples", default=32)
    parser.add_argument('--num_workers', type=int, default=4,help="load videos")
    parser.add_argument('--model_save_path', type=str, default='models', help='dictory path for save model')
    parser.add_argument("--model_prefix", type=str, default='zlfy', help='prefix name of model')
    parser.add_argument("--valid_interval", type=int, default=4, help='frequency of valid')
    parser.add_argument("--print_param", type=bool, default=True, help='print network structure')
    args = parser.parse_args()


    action_reg = TSN_INIT(data_dir=args.data_dir,
                          train_txt="train.txt",
                          valid_txt="valid.txt",
                          output_nums=args.output_nums,
                          segement_nums=args.segement_nums,
                          resume_file=args.resume_file)

    action_reg.train(epochs=args.epochs,
                     lr=args.lr,
                     lr_steps=args.lr_steps,
                     weight_decay=args.weight_decay,
                     batch_size=args.batch_size,
                     num_workers=args.num_workers,
                     model_save_path=args.model_save_path,
                     model_prefix=args.model_prefix,
                     valid_interval=args.valid_interval,
                     print_param=args.print_param)

#     action_reg.test("test.txt")