"""Inference demo of directional point detector."""
import math
import cv2 as cv
import numpy as np
import torch
from torchvision.transforms import ToTensor
import config
from data import get_predicted_points, pair_marking_points, calc_point_squre_dist, pass_through_third_point
from model import DirectionalPointDetector
from util import Timer


def plot_points(image, pred_points):
    """Plot marking points on the image."""
    if not pred_points:
        return
    height = image.shape[0]
    width = image.shape[1]
    for confidence, marking_point in pred_points:
        p0_x = width * marking_point.x - 0.5
        p0_y = height * marking_point.y - 0.5
        cos_val = math.cos(marking_point.direction)
        sin_val = math.sin(marking_point.direction)
        p1_x = p0_x + 50*cos_val
        p1_y = p0_y + 50*sin_val
        p2_x = p0_x - 50*sin_val
        p2_y = p0_y + 50*cos_val
        p3_x = p0_x + 50*sin_val
        p3_y = p0_y - 50*cos_val
        p0_x = int(round(p0_x))
        p0_y = int(round(p0_y))
        p1_x = int(round(p1_x))
        p1_y = int(round(p1_y))
        p2_x = int(round(p2_x))
        p2_y = int(round(p2_y))
        cv.line(image, (p0_x, p0_y), (p1_x, p1_y), (0, 0, 255), 2)
        cv.putText(image, str(confidence), (p0_x, p0_y),
                   cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        if marking_point.shape > 0.5:
            cv.line(image, (p0_x, p0_y), (p2_x, p2_y), (0, 0, 255), 2)
        else:
            p3_x = int(round(p3_x))
            p3_y = int(round(p3_y))
            cv.line(image, (p2_x, p2_y), (p3_x, p3_y), (0, 0, 255), 2)


def plot_slots(image, pred_points, slots):
    """Plot parking slots on the image."""
    if not pred_points or not slots:
        return
    marking_points = list(list(zip(*pred_points))[1])
    height = image.shape[0]
    width = image.shape[1]
    for slot in slots:
        point_a = marking_points[slot[0]]
        point_b = marking_points[slot[1]]
        p0_x = width * point_a.x - 0.5
        p0_y = height * point_a.y - 0.5
        p1_x = width * point_b.x - 0.5
        p1_y = height * point_b.y - 0.5
        vec = np.array([p1_x - p0_x, p1_y - p0_y])
        vec = vec / np.linalg.norm(vec)
        distance = calc_point_squre_dist(point_a, point_b)
        if config.VSLOT_MIN_DIST <= distance <= config.VSLOT_MAX_DIST:
            separating_length = config.LONG_SEPARATOR_LENGTH
        elif config.HSLOT_MIN_DIST <= distance <= config.HSLOT_MAX_DIST:
            separating_length = config.SHORT_SEPARATOR_LENGTH
        p2_x = p0_x + height * separating_length * vec[1]
        p2_y = p0_y - width * separating_length * vec[0]
        p3_x = p1_x + height * separating_length * vec[1]
        p3_y = p1_y - width * separating_length * vec[0]
        p0_x = int(round(p0_x))
        p0_y = int(round(p0_y))
        p1_x = int(round(p1_x))
        p1_y = int(round(p1_y))
        p2_x = int(round(p2_x))
        p2_y = int(round(p2_y))
        p3_x = int(round(p3_x))
        p3_y = int(round(p3_y))
        cv.line(image, (p0_x, p0_y), (p1_x, p1_y), (255, 0, 0), 2)
        cv.line(image, (p0_x, p0_y), (p2_x, p2_y), (255, 0, 0), 2)
        cv.line(image, (p1_x, p1_y), (p3_x, p3_y), (255, 0, 0), 2)


def preprocess_image(image):
    """Preprocess numpy image to torch tensor."""
    if image.shape[0] != 512 or image.shape[1] != 512:
        image = cv.resize(image, (512, 512))
    return torch.unsqueeze(ToTensor()(image), 0)


def detect_marking_points(detector, image, thresh, device):
    """Given image read from opencv, return detected marking points."""
    prediction = detector(preprocess_image(image).to(device))
    return get_predicted_points(prediction[0], thresh)


def inference_slots(marking_points):
    """Inference slots based on marking points."""
    num_detected = len(marking_points)
    slots = []
    for i in range(num_detected - 1):
        for j in range(i + 1, num_detected):
            point_i = marking_points[i]
            point_j = marking_points[j]
            # Step 1: length filtration.
            distance = calc_point_squre_dist(point_i, point_j)
            if not (config.VSLOT_MIN_DIST <= distance <= config.VSLOT_MAX_DIST
                    or config.HSLOT_MIN_DIST <= distance <= config.HSLOT_MAX_DIST):
                continue
            # Step 2: pass through filtration.
            if pass_through_third_point(marking_points, i, j):
                continue
            result = pair_marking_points(point_i, point_j)
            if result == 1:
                slots.append((i, j))
            elif result == -1:
                slots.append((j, i))
    return slots


def detect_video(detector, device, args):
    """Demo for detecting video."""
    timer = Timer()
    input_video = cv.VideoCapture(args.video)
    frame_width = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))
    output_video = cv.VideoWriter()
    if args.save:
        output_video.open('record.avi', cv.VideoWriter_fourcc(*'XVID'),
                          input_video.get(cv.CAP_PROP_FPS),
                          (frame_width, frame_height), True)
    frame = np.empty([frame_height, frame_width, 3], dtype=np.uint8)
    while input_video.read(frame)[0]:
        timer.tic()
        pred_points = detect_marking_points(
            detector, frame, args.thresh, device)
        slots = None
        if pred_points and args.inference_slot:
            marking_points = list(list(zip(*pred_points))[1])
            slots = inference_slots(marking_points)
        timer.toc()
        plot_points(frame, pred_points)
        plot_slots(frame, pred_points, slots)
        cv.imshow('demo', frame)
        cv.waitKey(1)
        if args.save:
            output_video.write(frame)
    print("Average time: ", timer.calc_average_time(), "s.")
    input_video.release()
    output_video.release()


def detect_image(detector, device, args):
    """Demo for detecting images."""
    timer = Timer()
    while True:
        image_file = input('Enter image file path: ')
        image = cv.imread(image_file)
        timer.tic()
        pred_points = detect_marking_points(
            detector, image, args.thresh, device)
        slots = None
        if pred_points and args.inference_slot:
            marking_points = list(list(zip(*pred_points))[1])
            slots = inference_slots(marking_points)
        timer.toc()
        plot_points(image, pred_points)
        plot_slots(image, pred_points, slots)
        cv.imshow('demo', image)
        cv.waitKey(1)
        if args.save:
            cv.imwrite('save.jpg', image, [int(cv.IMWRITE_JPEG_QUALITY), 100])


def inference_detector(args):
    """Inference demo of directional point detector."""
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device('cuda:' + str(args.gpu_id) if args.cuda else 'cpu')
    torch.set_grad_enabled(False)
    dp_detector = DirectionalPointDetector(
        3, args.depth_factor, config.NUM_FEATURE_MAP_CHANNEL).to(device)
    dp_detector.load_state_dict(torch.load(args.detector_weights))
    dp_detector.eval()
    if args.mode == "image":
        detect_image(dp_detector, device, args)
    elif args.mode == "video":
        detect_video(dp_detector, device, args)


if __name__ == '__main__':
    inference_detector(config.get_parser_for_inference().parse_args())
