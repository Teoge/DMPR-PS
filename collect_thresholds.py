"""Collect the value range of different propertity of ps dataset."""
import argparse
import json
import math
import os
import matplotlib.pyplot as plt
import numpy as np
from data import MarkingPoint
from data.struct import calc_point_squre_dist, direction_diff
from prepare_dataset import generalize_marks


def get_parser():
    """Return argument parser for collecting thresholds."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_directory', required=True,
                        help="The location of label directory.")
    return parser


def collect_thresholds(args):
    """Collect range of value from ground truth to determine threshold."""
    distances = []
    separator_angles = []
    bridge_angles = []

    for label_file in os.listdir(args.label_directory):
        print(label_file)
        with open(os.path.join(args.label_directory, label_file), 'r') as file:
            label = json.load(file)
        marks = np.array(label['marks'])
        slots = np.array(label['slots'])
        if slots.size == 0:
            continue
        if len(marks.shape) < 2:
            marks = np.expand_dims(marks, axis=0)
        if len(slots.shape) < 2:
            slots = np.expand_dims(slots, axis=0)
        marks[:, 0:4] -= 300.5
        marks = [MarkingPoint(*mark) for mark in generalize_marks(marks)]
        for slot in slots:
            mark_a = marks[slot[0] - 1]
            mark_b = marks[slot[1] - 1]
            distances.append(calc_point_squre_dist(mark_a, mark_b))

            vector_ab = np.array([mark_b.x - mark_a.x, mark_b.y - mark_a.y])
            vector_ab = vector_ab / np.linalg.norm(vector_ab)
            ab_bridge_direction = math.atan2(vector_ab[1], vector_ab[0])
            ba_bridge_direction = math.atan2(-vector_ab[1], -vector_ab[0])
            separator_direction = math.atan2(-vector_ab[0], vector_ab[1])

            sangle = direction_diff(separator_direction, mark_a.direction)
            if mark_a.shape > 0.5:
                separator_angles.append(sangle)
            else:
                bangle = direction_diff(ab_bridge_direction, mark_a.direction)
                if sangle < bangle:
                    separator_angles.append(sangle)
                else:
                    bridge_angles.append(bangle)

            bangle = direction_diff(ba_bridge_direction, mark_b.direction)
            if mark_b.shape > 0.5:
                bridge_angles.append(bangle)
            else:
                sangle = direction_diff(separator_direction, mark_b.direction)
                if sangle < bangle:
                    separator_angles.append(sangle)
                else:
                    bridge_angles.append(bangle)

    distances = sorted(distances)
    separator_angles = sorted(separator_angles)
    bridge_angles = sorted(bridge_angles)
    plt.figure()
    plt.hist(distances, len(distances) // 10)
    plt.figure()
    plt.hist(separator_angles, len(separator_angles) // 10)
    plt.figure()
    plt.hist(bridge_angles, len(bridge_angles) // 3)


if __name__ == '__main__':
    collect_thresholds(get_parser().parse_args())
