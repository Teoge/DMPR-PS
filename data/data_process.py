"""Defines data structure and related function to process these data."""
import math
import torch
import config
from data.struct import MarkingPoint


def non_maximum_suppression(pred_points):
    """Perform non-maxmum suppression on marking points."""
    suppressed = [False] * len(pred_points)
    for i in range(len(pred_points) - 1):
        for j in range(i + 1, len(pred_points)):
            dist_square = cal_squre_dist(pred_points[i][1], pred_points[j][1])
            # TODO: recalculate following parameter
            # minimum distance in training set: 40.309
            # (40.309 / 600)^2 = 0.004513376
            if dist_square < 0.0045:
                idx = i if pred_points[i][0] < pred_points[j][0] else j
                suppressed[idx] = True
    if any(suppressed):
        unsupres_pred_points = []
        for i, supres in enumerate(suppressed):
            if not supres:
                unsupres_pred_points.append(pred_points[i])
        return unsupres_pred_points
    return pred_points


def get_predicted_points(prediction, thresh):
    """Get marking points from one predicted feature map."""
    assert isinstance(prediction, torch.Tensor)
    predicted_points = []
    prediction = prediction.detach().cpu().numpy()
    for i in range(prediction.shape[1]):
        for j in range(prediction.shape[2]):
            if prediction[0, i, j] >= thresh:
                xval = (j + prediction[2, i, j]) / prediction.shape[2]
                yval = (i + prediction[3, i, j]) / prediction.shape[1]
                cos_value = prediction[4, i, j]
                sin_value = prediction[5, i, j]
                direction = math.atan2(sin_value, cos_value)
                marking_point = MarkingPoint(
                    xval, yval, direction, prediction[1, i, j])
                predicted_points.append((prediction[0, i, j], marking_point))
    return non_maximum_suppression(predicted_points)


def cal_squre_dist(point_a, point_b):
    """Calculate distance between two marking points."""
    distx = point_a.x - point_b.x
    disty = point_a.y - point_b.y
    return distx ** 2 + disty ** 2


def cal_direction_angle(point_a, point_b):
    """Calculate angle between direction in rad."""
    angle = abs(point_a.direction - point_b.direction)
    if angle > math.pi:
        angle = 2*math.pi - angle
    return angle


def match_marking_points(point_a, point_b):
    """Determine whether a detected point match ground truth."""
    dist_square = cal_squre_dist(point_a, point_b)
    angle = cal_direction_angle(point_a, point_b)
    return (dist_square < config.SQUARED_DISTANCE_THRESH
            and angle < config.DIRECTION_ANGLE_THRESH)
