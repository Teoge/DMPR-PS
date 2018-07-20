# -*- coding: utf-8 -*-
import math
import numpy as np
from visdom import Visdom
from PIL import ImageDraw


class Logger():
    """Logger for training."""
    def __init__(self, curve_names=None):
        self.curve_names = curve_names
        if curve_names:
            self.vis = Visdom()
            assert self.vis.check_connection()
            self.curve_y = None
            self.curve_x_start = 0
            self.curve_x_end = 0

    def log(self, **kwargs):
        """Log and print the information."""
        print("##############################################################")
        for key, value in kwargs.items():
            print(key, value, sep='\t')
        if not self.curve_names:
            return
        curve_step = np.array([kwargs[cn] for cn in self.curve_names])
        if self.curve_y is None:
            self.curve_y = curve_step
        else:
            self.curve_y = np.row_stack((self.curve_y, curve_step))
        self.curve_x_end = self.curve_x_end + 1

    def plot_curve(self):
        """Plot curve on visdom."""
        if (self.curve_x_end - self.curve_x_start < 2 or not self.curve_names):
            return
        if self.curve_x_start == 0:
            update_opt = None
        else:
            update_opt = 'append'
        curve_x = np.arange(self.curve_x_start, self.curve_x_end)
        curve_x = np.transpose(np.tile(curve_x, (len(self.curve_names), 1)))
        self.vis.line(Y=self.curve_y, X=curve_x, win='loss', update=update_opt,
                      opts=dict(showlegend=True, legend=self.curve_names))
        self.curve_x_start = self.curve_x_end
        self.curve_y = None

    def plot_marking_points(self, image, marking_points, win_name='mk_points'):
        """Plot marking points on visdom."""
        width, height = image.size
        draw = ImageDraw.Draw(image)
        for point in marking_points:
            p0_x = width * point[0]
            p0_y = height * point[1]
            p1_x = p0_x + 50*math.cos(point[2])
            p1_y = p0_y + 50*math.sin(point[2])
            draw.line((p0_x, p0_y, p1_x, p1_y), fill=(255, 0, 0))
        image = np.asarray(image, dtype="uint8")
        image = np.transpose(image, (2, 0, 1))
        self.vis.image(image, win=win_name)
