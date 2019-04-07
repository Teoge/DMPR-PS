"""Evaluate directional marking point detector."""
import json
import os
import cv2 as cv
import numpy as np
import torch
import config
import util
from data import match_slots, Slot
from model import DirectionalPointDetector
from inference import detect_marking_points, inference_slots


def get_ground_truths(label):
    """Read label to get ground truth slot."""
    slots = np.array(label['slots'])
    if slots.size == 0:
        return []
    if len(slots.shape) < 2:
        slots = np.expand_dims(slots, axis=0)
    marks = np.array(label['marks'])
    if len(marks.shape) < 2:
        marks = np.expand_dims(marks, axis=0)
    ground_truths = []
    for slot in slots:
        mark_a = marks[slot[0] - 1]
        mark_b = marks[slot[1] - 1]
        coords = np.array([mark_a[0], mark_a[1], mark_b[0], mark_b[1]])
        coords = (coords - 0.5) / 600
        ground_truths.append(Slot(*coords))
    return ground_truths


def psevaluate_detector(args):
    """Evaluate directional point detector."""
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device('cuda:' + str(args.gpu_id) if args.cuda else 'cpu')
    torch.set_grad_enabled(False)

    dp_detector = DirectionalPointDetector(
        3, args.depth_factor, config.NUM_FEATURE_MAP_CHANNEL).to(device)
    if args.detector_weights:
        dp_detector.load_state_dict(torch.load(args.detector_weights))
    dp_detector.eval()

    logger = util.Logger(enable_visdom=args.enable_visdom)

    ground_truths_list = []
    predictions_list = []
    for idx, label_file in enumerate(os.listdir(args.label_directory)):
        name = os.path.splitext(label_file)[0]
        print(idx, name)
        image = cv.imread(os.path.join(args.image_directory, name + '.jpg'))
        pred_points = detect_marking_points(
            dp_detector, image, config.CONFID_THRESH_FOR_POINT, device)
        if pred_points:
            marking_points = list(list(zip(*pred_points))[1])
            slots = inference_slots(marking_points)
        pred_slots = []
        for slot in slots:
            point_a = marking_points[slot[0]]
            point_b = marking_points[slot[1]]
            prob = min((pred_points[slot[0]][0], pred_points[slot[1]][0]))
            pred_slots.append(
                (prob, Slot(point_a.x, point_a.y, point_b.x, point_b.y)))
        predictions_list.append(pred_slots)

        with open(os.path.join(args.label_directory, label_file), 'r') as file:
            ground_truths_list.append(get_ground_truths(json.load(file)))

    precisions, recalls = util.calc_precision_recall(
        ground_truths_list, predictions_list, match_slots)
    average_precision = util.calc_average_precision(precisions, recalls)
    if args.enable_visdom:
        logger.plot_curve(precisions, recalls)
    logger.log(average_precision=average_precision)


if __name__ == '__main__':
    psevaluate_detector(config.get_parser_for_ps_evaluation().parse_args())
