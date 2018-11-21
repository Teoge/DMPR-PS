"""Perform data augmentation and preprocessing."""
import argparse
import json
import math
import os
import random
import cv2 as cv
import numpy as np


def get_parser():
    """Return argument parser for generating dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,
                        choices=['trainval', 'test'],
                        help="Generate trainval or test dataset.")
    parser.add_argument('--val_prop', type=float, default=0.1,
                        help="The proportion of val sample in trainval.")
    parser.add_argument('--label_directory', required=True,
                        help="The location of label directory.")
    parser.add_argument('--image_directory', required=True,
                        help="The location of image directory.")
    parser.add_argument('--output_directory', required=True,
                        help="The location of output directory.")
    return parser


def boundary_check(centralied_marks):
    """Check situation that marking point appears too near to border."""
    for mark in centralied_marks:
        if mark[0] < -260 or mark[0] > 260 or mark[1] < -260 or mark[1] > 260:
            return False
    return True


def overlap_check(centralied_marks):
    """Check situation that multiple marking points appear in same cell."""
    for i in range(len(centralied_marks) - 1):
        i_x = centralied_marks[i, 0]
        i_y = centralied_marks[i, 1]
        for j in range(i + 1, len(centralied_marks)):
            j_x = centralied_marks[j, 0]
            j_y = centralied_marks[j, 1]
            if abs(j_x - i_x) < 600 / 16 and abs(j_y - i_y) < 600 / 16:
                return False
    return True


def generalize_marks(centralied_marks):
    """Convert coordinate to [0, 1] and calculate direction label."""
    generalized_marks = []
    for mark in centralied_marks:
        xval = (mark[0] + 300) / 600
        yval = (mark[1] + 300) / 600
        direction = math.atan2(mark[3] - mark[1], mark[2] - mark[0])
        generalized_marks.append([xval, yval, direction, mark[4]])
    return generalized_marks


def write_image_and_label(name, image, centralied_marks, name_list):
    """Write image and label with given name."""
    name_list.append(os.path.basename(name))
    print("Processing NO.%d samples: %s..." % (len(name_list), name_list[-1]))
    image = cv.resize(image, (512, 512))
    cv.imwrite(name + '.jpg', image, [int(cv.IMWRITE_JPEG_QUALITY), 100])
    with open(name + '.json', 'w') as file:
        json.dump(generalize_marks(centralied_marks), file)


def rotate_vector(vector, angle_degree):
    """Rotate a vector with given angle in degree."""
    angle_rad = math.pi * angle_degree / 180
    xval = vector[0]*math.cos(angle_rad) + vector[1]*math.sin(angle_rad)
    yval = -vector[0]*math.sin(angle_rad) + vector[1]*math.cos(angle_rad)
    return xval, yval


def rotate_centralized_marks(centralied_marks, angle_degree):
    """Rotate centralized marks with given angle in degree."""
    rotated_marks = centralied_marks.copy()
    for i in range(centralied_marks.shape[0]):
        mark = centralied_marks[i]
        rotated_marks[i, 0:2] = rotate_vector(mark[0:2], angle_degree)
        rotated_marks[i, 2:4] = rotate_vector(mark[2:4], angle_degree)
    return rotated_marks


def rotate_image(image, angle_degree):
    """Rotate image with given angle in degree."""
    rows, cols, _ = image.shape
    rotation_matrix = cv.getRotationMatrix2D((rows/2, cols/2), angle_degree, 1)
    return cv.warpAffine(image, rotation_matrix, (rows, cols))


def generate_dataset(args):
    """Generate dataset according to arguments."""
    if args.dataset == 'trainval':
        val_directory = os.path.join(args.output_directory, 'val')
        args.output_directory = os.path.join(args.output_directory, 'train')
    elif args.dataset == 'test':
        args.output_directory = os.path.join(args.output_directory, 'test')
    os.makedirs(args.output_directory, exist_ok=True)
    name_list = []
    for label_file in os.listdir(args.label_directory):
        name = os.path.splitext(label_file)[0]
        image = cv.imread(os.path.join(args.image_directory, name + '.jpg'))
        with open(os.path.join(args.label_directory, label_file), 'r') as file:
            label = json.load(file)
        centralied_marks = np.array(label['marks'])
        if len(centralied_marks.shape) < 2:
            centralied_marks = np.expand_dims(centralied_marks, axis=0)
        centralied_marks[:, 0:4] -= 300.5
        if boundary_check(centralied_marks) or args.dataset == 'test':
            output_name = os.path.join(args.output_directory, name)
            write_image_and_label(output_name, image,
                                  centralied_marks, name_list)

        if args.dataset == 'test':
            continue
        for angle in range(5, 360, 5):
            rotated_marks = rotate_centralized_marks(centralied_marks, angle)
            if boundary_check(rotated_marks) and overlap_check(rotated_marks):
                rotated_image = rotate_image(image, angle)
                output_name = os.path.join(
                    args.output_directory, name + '_' + str(angle))
                write_image_and_label(
                    output_name, rotated_image, rotated_marks, name_list)

    if args.dataset == 'trainval':
        print("Dividing training set and validation set...")
        val_idx = random.sample(list(range(len(name_list))),
                                int(round(len(name_list)*args.val_prop)))
        val_samples = [name_list[idx] for idx in val_idx]
        os.makedirs(val_directory, exist_ok=True)
        for val_sample in val_samples:
            train_directory = args.output_directory
            image_src = os.path.join(train_directory, val_sample + '.jpg')
            label_src = os.path.join(train_directory, val_sample + '.json')
            image_dst = os.path.join(val_directory, val_sample + '.jpg')
            label_dst = os.path.join(val_directory, val_sample + '.json')
            os.rename(image_src, image_dst)
            os.rename(label_src, label_dst)
    print("Done.")


if __name__ == '__main__':
    generate_dataset(get_parser().parse_args())
