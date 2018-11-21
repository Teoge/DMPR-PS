"""Universal network struture unit definition."""
from torch import nn


def define_squeeze_unit(basic_channel_size):
    """Define a 1x1 squeeze convolution with norm and activation."""
    conv = nn.Conv2d(2 * basic_channel_size, basic_channel_size, kernel_size=1,
                     stride=1, padding=0, bias=False)
    norm = nn.BatchNorm2d(basic_channel_size)
    relu = nn.LeakyReLU(0.1)
    layers = [conv, norm, relu]
    return layers


def define_expand_unit(basic_channel_size):
    """Define a 3x3 expand convolution with norm and activation."""
    conv = nn.Conv2d(basic_channel_size, 2 * basic_channel_size, kernel_size=3,
                     stride=1, padding=1, bias=False)
    norm = nn.BatchNorm2d(2 * basic_channel_size)
    relu = nn.LeakyReLU(0.1)
    layers = [conv, norm, relu]
    return layers


def define_halve_unit(basic_channel_size):
    """Define a 4x4 stride 2 expand convolution with norm and activation."""
    conv = nn.Conv2d(basic_channel_size, 2 * basic_channel_size, kernel_size=4,
                     stride=2, padding=1, bias=False)
    norm = nn.BatchNorm2d(2 * basic_channel_size)
    relu = nn.LeakyReLU(0.1)
    layers = [conv, norm, relu]
    return layers


def define_depthwise_expand_unit(basic_channel_size):
    """Define a 3x3 expand convolution with norm and activation."""
    conv1 = nn.Conv2d(basic_channel_size, 2 * basic_channel_size,
                      kernel_size=1, stride=1, padding=0, bias=False)
    norm1 = nn.BatchNorm2d(2 * basic_channel_size)
    relu1 = nn.LeakyReLU(0.1)
    conv2 = nn.Conv2d(2 * basic_channel_size, 2 * basic_channel_size, kernel_size=3,
                      stride=1, padding=1, bias=False, groups=2 * basic_channel_size)
    norm2 = nn.BatchNorm2d(2 * basic_channel_size)
    relu2 = nn.LeakyReLU(0.1)
    layers = [conv1, norm1, relu1, conv2, norm2, relu2]
    return layers


def define_detector_block(basic_channel_size):
    """Define a unit composite of a squeeze and expand unit."""
    layers = []
    layers += define_squeeze_unit(basic_channel_size)
    layers += define_expand_unit(basic_channel_size)
    return layers
