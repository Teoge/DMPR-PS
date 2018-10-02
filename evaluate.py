"""Evaluate directional marking point detector."""
import torch
from torch.utils.data import DataLoader
from precision_recall import calc_average_precision
from precision_recall import calc_precision_recall
import config
from data import generate_objective
from data import get_predicted_points
from data import match_marking_points
from dataset import ParkingSlotDataset
from detector import DirectionalPointDetector
from log import Logger


def evaluate_detector(args):
    """Evaluate directional point detector."""
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device('cuda:'+str(args.gpu_id) if args.cuda else 'cpu')

    dp_detector = DirectionalPointDetector(
        3, args.depth_factor, config.NUM_FEATURE_MAP_CHANNEL).to(device)
    if args.detector_weights:
        dp_detector.load_state_dict(torch.load(args.detector_weights))

    data_loader = DataLoader(ParkingSlotDataset(args.dataset_directory),
                             batch_size=args.batch_size, shuffle=True,
                             num_workers=args.data_loading_workers,
                             collate_fn=lambda x: list(zip(*x)))
    logger = Logger()

    total_loss = 0
    num_evaluation = 0
    ground_truths_list = []
    predictions_list = []
    for image, marking_points in data_loader:
        image = torch.stack(image)
        image = image.to(device)
        ground_truths_list += list(marking_points)

        prediction = dp_detector(image)
        objective, gradient = generate_objective(marking_points, device)
        loss = (prediction - objective) ** 2
        total_loss += torch.sum(loss*gradient).item()
        num_evaluation += loss.size(0)

        pred_points = [get_predicted_points(pred, 0.01) for pred in prediction]
        predictions_list += pred_points

    precisions, recalls = calc_precision_recall(
        ground_truths_list, predictions_list, match_marking_points)
    average_precision = calc_average_precision(precisions, recalls)
    if args.enable_visdom:
        logger.plot_curve(precisions, recalls)
    logger.log(average_loss=total_loss / num_evaluation,
               average_precision=average_precision)


if __name__ == '__main__':
    evaluate_detector(config.get_parser_for_evaluation().parse_args())
