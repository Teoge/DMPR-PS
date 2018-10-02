# -*- coding: utf-8 -*-
import math
import time
import cv2 as cv
import torch
import numpy as np
from PIL import Image


class Timer(object):
    """Timer."""

    def __init__(self):
        self.start_ticking = False
        self.start = 0.

    def tic(self):
        """Start timer."""
        self.start = time.time()
        self.start_ticking = True

    def toc(self):
        """End timer."""
        duration = time.time() - self.start
        self.start_ticking = False
        print("Time elapsed:", duration, "s.")


def tensor2array(image_tensor, imtype=np.uint8):
    """
    Convert float CxHxW image tensor between [0, 1] to HxWxC numpy ndarray
    between [0, 255]
    """
    assert isinstance(image_tensor, torch.Tensor)
    image_numpy = (image_tensor.detach().cpu().numpy()) * 255.0
    image_numpy = np.transpose(image_numpy, (1, 2, 0)).astype(imtype)
    return image_numpy


def tensor2im(image_tensor, imtype=np.uint8):
    """Convert float CxHxW BGR image tensor to RGB PIL Image"""
    image_numpy = tensor2array(image_tensor, imtype)
    image_numpy = cv.cvtColor(image_numpy, cv.COLOR_BGR2RGB)
    return Image.fromarray(image_numpy)
