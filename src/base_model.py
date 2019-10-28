#! /usr/bin/env python
# -*- encoding:utf-8 -*-
'''
@File    : base_model.py
@Time    : 2019/09/25 11:11:33
@Author  : Gaozong/260243
@Contact : 260243@gree.com.cn/zong209@163.com
'''
# import depend packages
import torch
import torch.nn as nn
import torch.nn.functional as F


class TSN_BASE(nn.Module):
    def __init__(self):
        super(TSN_BASE, self).__init__()
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2_reduce = BasicConv2d(64, 64, kernel_size=1)
        self.conv2_3x3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.Inception3A = Inception3a(192, [64, 64, 64, 64, 96, 32])
        self.Inception3B = Inception3a(256, [64, 64, 96, 96, 96, 64])
        self.Inception3C = Inception3c(320, [128, 160, 64, 96])
        self.Inception4A = Inception3a(576, [224, 64, 96, 96, 128, 128])
        self.Inception4B = Inception3a(576, [192, 96, 128, 96, 128, 128])
        self.Inception4C = Inception3a(576, [160, 128, 160, 128, 160, 128])
        self.Inception4D = Inception3a(608, [96, 128, 192, 160, 192, 128])
        self.Inception4E = Inception3c(608, [128, 192, 192, 256])
        self.Inception5A = Inception3a(1056, [352, 192, 320, 160, 224, 128])
        self.Inception5B = Inception3a(1024, [352, 192, 320, 192, 224, 128])
        self.Global_pool = Average_pool_module(1024)

    def forward(self, x):
        out = self.conv1(x)
        out = F.avg_pool2d(out, kernel_size=3, stride=2, padding=1)
        out = self.conv2_reduce(out)
        out = self.conv2_3x3(out)
        out = F.avg_pool2d(out, kernel_size=3, stride=2, padding=1)
        out = self.Inception3A(out)
        out = self.Inception3B(out)
        out = self.Inception3C(out)
        out = self.Inception4A(out)
        out = self.Inception4B(out)
        out = self.Inception4C(out)
        out = self.Inception4D(out)
        out = self.Inception4E(out)
        out = self.Inception5A(out)
        out = self.Inception5B(out)
        out = F.avg_pool2d(out, kernel_size=7, stride=1)
        out = torch.flatten(out, 1)
        out = self.Global_pool(out)
        return out

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(
                    m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError(
                        "New atomic module type: {}. Need to give it a learning policy"
                        .format(type(m)))

        return [
            {
                'params': first_conv_weight,
                'lr_mult': 1,
                'decay_mult': 1,
                'name': "first_conv_weight"
            },
            {
                'params': first_conv_bias,
                'lr_mult': 2,
                'decay_mult': 0,
                'name': "first_conv_bias"
            },
            {
                'params': normal_weight,
                'lr_mult': 1,
                'decay_mult': 1,
                'name': "normal_weight"
            },
            {
                'params': normal_bias,
                'lr_mult': 2,
                'decay_mult': 0,
                'name': "normal_bias"
            },
            {
                'params': bn,
                'lr_mult': 1,
                'decay_mult': 0,
                'name': "BN scale/shift"
            },
        ]


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inception3a(nn.Module):
    def __init__(self, in_channels, in_network_channels):
        """
    Inception Model for TSN network

    Args:
      in_channels: channels of input features, type: int
      in_network_channels: 
        output channels for every oprate, length:6, type: Array,
        example:[out_channels_1x1,out_channels_3x3_reduce,
        out_channels_3x3_conv,out_channels_doub3x3_reduce,
        out_channels_doub3x3_conv,out_channels_features]
    """
        super(Inception3a, self).__init__()
        [
            out_channels_1x1, out_channels_3x3_reduce, out_channels_3x3_conv,
            out_channels_doub3x3_reduce, out_channels_doub3x3_conv,
            out_channels_features
        ] = in_network_channels
        self.branch1x1 = BasicConv2d(in_channels,
                                     out_channels_1x1,
                                     kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels,
                                       out_channels_3x3_reduce,
                                       kernel_size=1)
        self.branch3x3_2 = BasicConv2d(out_channels_3x3_reduce,
                                       out_channels_3x3_conv,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        self.branchdb3x3_1 = BasicConv2d(in_channels,
                                         out_channels_doub3x3_reduce,
                                         kernel_size=1)
        self.branchdb3x3_2 = BasicConv2d(out_channels_doub3x3_reduce,
                                         out_channels_doub3x3_conv,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1)
        self.branchdb3x3_3 = BasicConv2d(out_channels_doub3x3_conv,
                                         out_channels_doub3x3_conv,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1)

        self.branch_pool = BasicConv2d(in_channels,
                                       out_channels_features,
                                       kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch3x3db = self.branchdb3x3_1(x)
        branch3x3db = self.branchdb3x3_2(branch3x3db)
        branch3x3db = self.branchdb3x3_3(branch3x3db)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3db, branch_pool]
        return torch.cat(outputs, 1)


class Inception3c(nn.Module):
    def __init__(self, in_channels, in_network_channels):
        """
    Inception Model for TSN network
    Args:
      in_channels: channels of input features, type: int
      in_network_channels: 
        output channels for every oprate, length:4, type: Array,
        example:[out_channels_3x3_reduce,out_channels_3x3_conv,
        out_channels_doub3x3_reduce,out_channels_doub3x3_conv]
    """
        super(Inception3c, self).__init__()
        [
            out_channels_3x3_reduce, out_channels_3x3_conv,
            out_channels_doub3x3_reduce, out_channels_doub3x3_conv
        ] = in_network_channels

        self.branch3x3_1 = BasicConv2d(in_channels,
                                       out_channels_3x3_reduce,
                                       kernel_size=1)
        self.branch3x3_2 = BasicConv2d(out_channels_3x3_reduce,
                                       out_channels_3x3_conv,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1)

        self.branchdb3x3_1 = BasicConv2d(in_channels,
                                         out_channels_doub3x3_reduce,
                                         kernel_size=1)
        self.branchdb3x3_2 = BasicConv2d(out_channels_doub3x3_reduce,
                                         out_channels_doub3x3_conv,
                                         kernel_size=3,
                                         padding=1,
                                         stride=1)
        self.branchdb3x3_3 = BasicConv2d(out_channels_doub3x3_conv,
                                         out_channels_doub3x3_conv,
                                         kernel_size=3,
                                         stride=2,
                                         padding=1)

    def forward(self, x):

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch3x3db = self.branchdb3x3_1(x)
        branch3x3db = self.branchdb3x3_2(branch3x3db)
        branch3x3db = self.branchdb3x3_3(branch3x3db)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)

        outputs = [branch3x3, branch3x3db, branch_pool]
        return torch.cat(outputs, 1)


class Average_pool_module(nn.Module):
    def __init__(self, in_features):
        super(Average_pool_module, self).__init__()
        self.in_features = in_features

    def forward(self, x):
        return x
