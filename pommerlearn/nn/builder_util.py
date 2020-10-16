"""
Utility methods for building the neural network architectures.
"""
from typing import Any

import torch
from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, LeakyReLU, Sigmoid, Tanh, Linear, Hardsigmoid, Hardswish


def get_act(act_type):
    """Wrapper method for different non linear activation functions"""
    if act_type == "relu":
        return ReLU()
    if act_type == "sigmoid":
        return Sigmoid()
    if act_type == "tanh":
        return Tanh()
    if act_type == "lrelu":
        return LeakyReLU(negative_slope=0.2)
    if act_type == "hard_sigmoid":
        return Hardsigmoid()
    if act_type == "hard_swish":
        return Hardswish()
    raise NotImplementedError


class MixConv(torch.nn.Module):
    def __init__(self, data, name, nb_input_channels, channels, bn_mom, kernels):
        """
        Mix depth-wise convolution layers, Mingxing Tan, Quoc V. Le, https://arxiv.org/abs/1907.09595
        :param data: Input data
        :param name: Name of the block
        :param nb_input_channels: Number of input channels
        :param channels: Number of convolutional channels
        :param bn_mom: Batch normalization momentum
        :param kernels: List of kernel sizes to use
        :return: symbol
        """
        super(MixConv, self).__init__()

        self.branches = []
        self.num_splits = len(kernels)
        conv_layers = []

        for xi, kernel in zip(torch.split(data, dim=1, split_size_or_sections=self.num_splits), kernels):
            branch = Sequential()

            branch.append(Conv2d(in_channels=nb_input_channels//self.num_splits,
                                      out_channels=channels//self.num_splits, kernel_size=(kernel, kernel),
                                      padding=(kernel//2, kernel//2), bias=False,
                                      groups=channels//self.num_splits))
            self.branches.append(branch)

    def forward(self, x):
        """
        Compute forward pass
        :param x: Input data to the block
        :return: Activation maps of the block
        """
        if self.num_splits == 1:
            return self.branches[0](x)

        conv_layers = []
        for xi, branch in zip(torch.split(x, dim=1, split_size_or_sections=self.num_splits), self.branches):
            conv_layers.append(branch(xi))

        return torch.cat(conv_layers, 0)
