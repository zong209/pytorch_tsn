#! .pyenv/bin/python
# -*- encoding:utf-8 -*-
'''
@File     : consensus.py
@Time     : 2019/10/09 09:04:37
@Author   : Gaozong/260243
@Contact  : 260243@gree.com.cn/zong209@163.com
@Describe : consensus layer
'''

import torch


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


class SegmentConsensus(torch.autograd.Function):
    # def __init__(self, consensus_type, dim=1):
    #     self.consensus_type = consensus_type
    #     self.dim = dim
    #     self.shape = None

    @staticmethod
    def forward(ctx, input_tensor, consensus_type, dim=1):

        ctx.shape = input_tensor.size()
        ctx.consensus_type = consensus_type
        ctx.dim = dim
        if consensus_type == 'avg':
            output = input_tensor.mean(dim=dim, keepdim=True)
        elif consensus_type == 'identity':
            output = input_tensor
        else:
            output = None
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.consensus_type == 'avg':
            grad_in = grad_output.expand(ctx.shape) / float(ctx.shape[ctx.dim])
        elif ctx.consensus_type == 'identity':
            grad_in = grad_output
        else:
            grad_in = None

        return grad_in, None, None


class ConsensusModule(torch.nn.Module):
    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus.apply(input, self.consensus_type, self.dim)
