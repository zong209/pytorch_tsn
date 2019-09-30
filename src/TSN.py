#! /usr/bin/env python
# -*- encoding:utf-8 -*-
'''
@File    : InceptionV3.py
@Time    : 2019/09/25 11:11:33
@Author  : Gaozong/260243
@Contact : 260243@gree.com.cn/zong209@163.com
'''
# import depend packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class Tsn_network(nn.Module):
    def __init__(self, num_segements, num_classes=4):
        super(Tsn_network, self).__init__()
        self.conv1 = BasicConv2d(num_segements,
                                 64,
                                 kernel_size=7,
                                 stride=2,
                                 padding=3)
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
        self.pool_fc = InceptionAux(1024, num_classes)
        self.Consensus = AttentionLayer(num_segements, num_classes)

    def forward(self, x):
        x = x.reshape([-1, 3, 224, 224]).float()
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
        out = self.pool_fc(out)
        out = self.Consensus(out)
        return out


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


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.fc = nn.Linear(in_channels, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=7, stride=1)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


class PoolConsensus(nn.Module):
    def __init__(self):
        super(PoolConsensus, self).__init__()

    def forward(self, x):
        x = F.avg_pool2d(x, kernel=0, stride=1)

        return x


def sum_exp_el(exp_el_sum, extend_array, size):
    batch_zize, segement_num, output_num = size
    sum_array = []
    for i in range(batch_zize):
        sum_array.append(exp_el_sum[i] * extend_array[i])
    sum_array = torch.cat(sum_array).reshape(size)
    return sum_array


class AttConsensus(Function):
    @staticmethod
    def forward(ctx, input, weight):
        el = input * weight
        el_exp = el.exp()
        sum_1 = el_exp.sum(1)
        size = el_exp.size()
        sum_array = sum_exp_el(sum_1, torch.ones(size), size)
        attention = el_exp.div(sum_array)
        ctx.save_for_backward(input, weight, el_exp, sum_array, attention)
        output = (input * (attention)).sum(1)
        return output

    @staticmethod
    def backward(ctx, grad_output):

        input, weight, el_exp, sum_array, attention = ctx.saved_variables
        grad_input = grad_weight = None

        if ctx.needs_input_grad[0]:
            grad_input = sum_exp_el(grad_output, attention, attention.size())
        if ctx.needs_input_grad[1]:
            grad_att = attention * (sum_array - el_exp)

            grad_weight = sum_exp_el(grad_output, (input) * (grad_att),
                                     grad_att.size())

        return grad_input, grad_weight


class AttentionLayer(nn.Module):
    def __init__(self, segememt_num, outputs_num):
        super(AttentionLayer, self).__init__()
        self.segememt_num = segememt_num
        self.outputs_num = outputs_num
        w = torch.empty(segememt_num, outputs_num)
        self.weight = nn.Parameter(torch.nn.init.xavier_uniform_(w, gain=1))

    def forward(self, input):

        return AttConsensus.apply(
            input.reshape([-1, self.segememt_num, self.outputs_num]),
            self.weight)
