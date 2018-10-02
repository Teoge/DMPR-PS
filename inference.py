"""Inference demo of directional point detector."""
import math
import cv2 as cv
import numpy as np
import torch
from torchvision.transforms import ToTensor
import config
from data import get_predicted_points
from detector import DirectionalPointDetector
from utils import Timer


def plot_marking_points(image, marking_points):
    """Plot marking points on the image and show."""
    height = image.shape[0]
    width = image.shape[1]
    for marking_point in marking_points:
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
        cv.line(image, (p0_x, p0_y), (p1_x, p1_y), (0, 0, 255))
        if marking_point.shape > 0.5:
            cv.line(image, (p0_x, p0_y), (p2_x, p2_y), (0, 0, 255))
        else:
            p3_x = int(round(p3_x))
            p3_y = int(round(p3_y))
            cv.line(image, (p2_x, p2_y), (p3_x, p3_y), (0, 0, 255))


def preprocess_image(image):
    """Preprocess numpy image to torch tensor."""
    if image.shape[0] != 512 or image.shape[1] != 512:
        image = cv.resize(image, (512, 512))
    return torch.unsqueeze(ToTensor()(image), 0)


def detect_video(detector, device, args):
    """Demo for detecting video."""
    timer = Timer()
    input_video = cv.VideoCapture(args.video)
    frame_width = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))
    output_video = cv.VideoWriter()
    if args.save:
        output_video.open('record.avi', cv.VideoWriter_fourcc(* 'MJPG'),
                          input_video.get(cv.CAP_PROP_FPS),
                          (frame_width, frame_height))
    frame = np.empty([frame_height, frame_width, 3], dtype=np.uint8)
    while input_video.read(frame)[0]:
        if args.timing:
            timer.tic()
        prediction = detector(preprocess_image(frame).to(device))
        if args.timing:
            timer.toc()
        pred_points = get_predicted_points(prediction[0], args.thresh)
        if pred_points:
            plot_marking_points(frame, list(list(zip(*pred_points))[1]))
            cv.imshow('demo', frame)
            cv.waitKey(1)
        if args.save:
            output_video.write(frame)
    input_video.release()
    output_video.release()


def detect_image(detector, device, args):
    """Demo for detecting images."""
    image_file = input('Enter image file path: ')
    image = cv.imread(image_file)
    prediction = detector(preprocess_image(image).to(device))
    pred_points = get_predicted_points(prediction[0], args.thresh)
    if pred_points:
        plot_marking_points(image, list(list(zip(*pred_points))[1]))
        cv.imshow('demo', image)
        cv.waitKey(1)


def inference_detector(args):
    """Inference demo of directional point detector."""
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device('cuda:' + str(args.gpu_id) if args.cuda else 'cpu')
    dp_detector = DirectionalPointDetector(
        3, args.depth_factor, config.NUM_FEATURE_MAP_CHANNEL).to(device)
    dp_detector.load_state_dict(torch.load(args.detector_weights))
    if args.mode == "image":
        detect_image(dp_detector, device, args)
    elif args.mode == "video":
        detect_video(dp_detector, device, args)


if __name__ == '__main__':
    inference_detector(config.get_parser_for_inference().parse_args())
