"""Defines data structure."""
import math
from collections import namedtuple
from enum import Enum
import config


MarkingPoint = namedtuple('MarkingPoint', ['x', 'y', 'direction', 'shape'])
Slot = namedtuple('Slot', ['x1', 'y1', 'x2', 'y2'])


class PointShape(Enum):
    """The point shape types used to pair two marking points into slot."""
    none = 0
    l_down = 1
    t_down = 2
    t_middle = 3
    t_up = 4
    l_up = 5


def direction_diff(direction_a, direction_b):
    """Calculate the angle between two direction."""
    diff = abs(direction_a - direction_b)
    return diff if diff < math.pi else 2*math.pi - diff


def detemine_point_shape(point, vector):
    """Determine which category the point is in."""
    vec_direct = math.atan2(vector[1], vector[0])
    vec_direct_up = math.atan2(-vector[0], vector[1])
    vec_direct_down = math.atan2(vector[0], -vector[1])
    if point.shape < 0.5:
        if direction_diff(vec_direct, point.direction) < config.BRIDGE_ANGLE_DIFF:
            return PointShape.t_middle
        if direction_diff(vec_direct_up, point.direction) < config.SEPARATOR_ANGLE_DIFF:
            return PointShape.t_up
        if direction_diff(vec_direct_down, point.direction) < config.SEPARATOR_ANGLE_DIFF:
            return PointShape.t_down
    else:
        if direction_diff(vec_direct, point.direction) < config.BRIDGE_ANGLE_DIFF:
            return PointShape.l_down
        if direction_diff(vec_direct_up, point.direction) < config.SEPARATOR_ANGLE_DIFF:
            return PointShape.l_up
    return PointShape.none


def calc_point_squre_dist(point_a, point_b):
    """Calculate distance between two marking points."""
    distx = point_a.x - point_b.x
    disty = point_a.y - point_b.y
    return distx ** 2 + disty ** 2


def calc_point_direction_angle(point_a, point_b):
    """Calculate angle between direction in rad."""
    return direction_diff(point_a.direction, point_b.direction)


def match_marking_points(point_a, point_b):
    """Determine whether a detected point match ground truth."""
    dist_square = calc_point_squre_dist(point_a, point_b)
    angle = calc_point_direction_angle(point_a, point_b)
    if point_a.shape > 0.5 and point_b.shape < 0.5:
        return False
    if point_a.shape < 0.5 and point_b.shape > 0.5:
        return False
    return (dist_square < config.SQUARED_DISTANCE_THRESH
            and angle < config.DIRECTION_ANGLE_THRESH)


def match_slots(slot_a, slot_b):
    """Determine whether a detected slot match ground truth."""
    dist_x1 = slot_b.x1 - slot_a.x1
    dist_y1 = slot_b.y1 - slot_a.y1
    squared_dist1 = dist_x1**2 + dist_y1**2
    dist_x2 = slot_b.x2 - slot_a.x2
    dist_y2 = slot_b.y2 - slot_a.y2
    squared_dist2 = dist_x2 ** 2 + dist_y2 ** 2
    return (squared_dist1 < config.SQUARED_DISTANCE_THRESH
            and squared_dist2 < config.SQUARED_DISTANCE_THRESH)
