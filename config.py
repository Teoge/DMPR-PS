"""Configurate arguments."""
import argparse


INPUT_IMAGE_SIZE = 512
# 0: confidence, 1: offset_x, 2: offset_y, 3: cos(direction), 4: sin(direction)
NUM_FEATURE_MAP_CHANNEL = 5
# image_size / 2^5 = 512 / 32 = 16
FEATURE_MAP_SIZE = 16


def add_common_arguments(parser):
    """Add common arguments for training and inference."""
    parser.add_argument('--detector_weights',
                        help="The weights of pretrained detector.")
    parser.add_argument('--depth_factor', type=int, default=32,
                        help="Depth factor.")
    parser.add_argument('--disable_cuda', action='store_true',
                        help="Disable CUDA.")
    parser.add_argument('--gpu_id', type=int, default=1,
                        help="Select which gpu to use.")


def get_parser_for_training():
    """Return argument parser for training."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_directory', required=True,
                        help="The location of dataset.")
    parser.add_argument('--optimizer_weights',
                        help="The weights of optimizer.")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size.")
    parser.add_argument('--num_epochs', type=int, default=100,
                        help="Number of epochs to train for.")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="The learning rate of back propagation.")
    parser.add_argument('--enable_visdom', action='store_true',
                        help="Enable Visdom to visualize training progress")
    add_common_arguments(parser)
    return parser


def get_parser_for_inference():
    """Return argument parser for inference."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['image', 'video'],
                        help="Inference image or video.")
    parser.add_argument('--video',
                        help="Video path if you choose to inference video.")
    parser.add_argument('--thresh', type=float, default=0.5,
                        help="Detection threshold.")
    parser.add_argument('--timing', action='store_true',
                        help="Perform timing during reference.")
    parser.add_argument('--save', action='store_true',
                        help="Save detection result to file.")
    add_common_arguments(parser)
    return parser
