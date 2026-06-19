import torch.nn as nn


def _conv1d_3(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
        nn.BatchNorm1d(out_channels),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
    )