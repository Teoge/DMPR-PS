# -*- coding: utf-8 -*-
import math
import time
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


def non_maximum_suppression(marking_points):
    """Perform non-maxmum suppression on marking points."""
    suppressed = [False] * len(marking_points)
    for i in range(len(marking_points) - 1):
        for j in range(i + 1, len(marking_points)):
            distx = marking_points[i][0] - marking_points[j][0]
            disty = marking_points[i][1] - marking_points[j][1]
            dist_square = distx ** 2 + disty ** 2
            # minimum distance in training set: 40.309
            # (40.309 / 600)^2 = 0.004513376
            if dist_square < 0.0045:
                idx = i if marking_points[i][3] < marking_points[j][3] else j
                suppressed[idx] = True
    if any(suppressed):
        new_marking_points = []
        for i, supres in enumerate(suppressed):
            if not supres:
                new_marking_points.append(marking_points[i])
        return new_marking_points
    return marking_points


def get_marking_points(prediction, thresh):
    """Get marking point from predicted feature map."""
    assert isinstance(prediction, torch.Tensor)
    marking_points = []
    prediction = prediction.detach().cpu().numpy()
    for i in range(prediction.shape[1]):
        for j in range(prediction.shape[2]):
            if prediction[0, i, j] > thresh:
                xval = (j + prediction[1, i, j]) / prediction.shape[2]
                yval = (i + prediction[2, i, j]) / prediction.shape[1]
                cos_value = prediction[3, i, j]
                sin_value = prediction[4, i, j]
                angle = math.atan2(sin_value, cos_value)
                marking_points.append([xval, yval, angle, prediction[0, i, j]])
    return non_maximum_suppression(marking_points)


def tensor2array(image_tensor, imtype=np.uint8):
    """Convert float image tensor to numpy ndarray"""
    assert isinstance(image_tensor, torch.Tensor)
    image_numpy = (image_tensor.detach().cpu().numpy()) * 255.0
    return image_numpy.astype(imtype)


def tensor2im(image_tensor, imtype=np.uint8):
    """Convert float image tensor to PIL Image"""
    image_numpy = np.transpose(tensor2array(image_tensor, imtype), (1, 2, 0))
    return Image.fromarray(image_numpy)
