"""Utility for neural network layers"""

import torch


def conv_bn_relu(
    in_filter: int, out_filter: int, kernel: int, stride: int, pad: int
) -> torch.nn.Sequential:
    """Helper function for creating down convolution layer"""
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_filter, out_filter, kernel_size=kernel, stride=stride, padding=pad),
        torch.nn.BatchNorm2d(out_filter),
        torch.nn.ReLU(),
    )


def trans_conv_bn_relu(
    in_filter: int, out_filter: int, kernel: int, stride: int, pad: int
) -> torch.nn.Sequential:
    """Helper function for creating up convolution layer"""
    return torch.nn.Sequential(
        torch.nn.ConvTranspose2d(
            in_filter,
            out_filter,
            kernel_size=kernel,
            stride=stride,
            padding=pad,
            output_padding=1,
        ),
        torch.nn.BatchNorm2d(out_filter),
        torch.nn.ReLU(),
    )


def trans_conv_relu(
    in_filter: int, out_filter: int, kernel: int, stride: int, pad: int
) -> torch.nn.Sequential:
    """Helper function for creating up convolution layer"""
    return torch.nn.Sequential(
        torch.nn.ConvTranspose2d(
            in_filter,
            out_filter,
            kernel_size=kernel,
            stride=stride,
            padding=pad,
            output_padding=1,
        ),
        torch.nn.ReLU(),
    )


def trans_conv_sigmoid(
    in_filter: int, out_filter: int, kernel: int, stride: int, pad: int
) -> torch.nn.Sequential:
    """Helper function for creating up convolution layer"""
    return torch.nn.Sequential(
        torch.nn.ConvTranspose2d(
            in_filter,
            out_filter,
            kernel_size=kernel,
            stride=stride,
            padding=pad,
            output_padding=1,
        ),
        torch.nn.Sigmoid(),
    )


def linear_bn_relu(in_: int, out_: int) -> torch.nn.Sequential:
    """Linear layers helper function"""
    return torch.nn.Sequential(
        torch.nn.Linear(in_, out_),
        torch.nn.BatchNorm1d(out_),
        torch.nn.ReLU(),
    )
