"""Class for logging."""
import math
import numpy as np
from visdom import Visdom
from PIL import ImageDraw


class Logger():
    """Logger for training."""

    def __init__(self, enable_visdom=False, curve_names=None):
        self.curve_names = curve_names
        if enable_visdom:
            self.vis = Visdom()
            assert self.vis.check_connection()
            self.curve_x = np.array([0])
        else:
            self.curve_names = None

    def log(self, xval=None, win_name='loss', **kwargs):
        """Log and print the information."""
        print("##############################################################")
        for key, value in kwargs.items():
            print(key, value, sep='\t')

        if self.curve_names:
            if not xval:
                xval = self.curve_x
            for i in range(len(self.curve_names)):
                name = self.curve_names[i]
                if name not in kwargs:
                    continue
                yval = np.array([kwargs[name]])
                self.vis.line(Y=yval, X=xval, win=win_name, update='append',
                              name=name, opts=dict(showlegend=True))
                self.curve_x += 1

    def plot_curve(self, yvals, xvals, win_name='pr_curves'):
        """Plot curve."""
        self.vis.line(Y=np.array(yvals), X=np.array(xvals), win=win_name)

    def plot_marking_points(self, image, marking_points, win_name='mk_points'):
        """Plot marking points on visdom."""
        width, height = image.size
        draw = ImageDraw.Draw(image)
        for point in marking_points:
            p0_x = width * point.x
            p0_y = height * point.y
            p1_x = p0_x + 50*math.cos(point.direction)
            p1_y = p0_y + 50*math.sin(point.direction)
            draw.line((p0_x, p0_y, p1_x, p1_y), fill=(255, 0, 0))
            p2_x = p0_x - 50*math.sin(point.direction)
            p2_y = p0_y + 50*math.cos(point.direction)
            if point.shape > 0.5:
                draw.line((p2_x, p2_y, p0_x, p0_y), fill=(255, 0, 0))
            else:
                p3_x = p0_x + 50*math.sin(point.direction)
                p3_y = p0_y - 50*math.cos(point.direction)
                draw.line((p2_x, p2_y, p3_x, p3_y), fill=(255, 0, 0))
        image = np.asarray(image, dtype="uint8")
        image = np.transpose(image, (2, 0, 1))
        self.vis.image(image, win=win_name)
