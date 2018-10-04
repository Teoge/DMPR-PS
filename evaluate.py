"""Evaluate directional marking point detector."""
import torch
from torch.utils.data import DataLoader
import config
import util
from data import get_predicted_points, match_marking_points
from data import ParkingSlotDataset
from model import DirectionalPointDetector
from train import generate_objective


def evaluate_detector(args):
    """Evaluate directional point detector."""
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device('cuda:' + str(args.gpu_id) if args.cuda else 'cpu')
    torch.set_grad_enabled(False)

    dp_detector = DirectionalPointDetector(
        3, args.depth_factor, config.NUM_FEATURE_MAP_CHANNEL).to(device)
    if args.detector_weights:
        dp_detector.load_state_dict(torch.load(args.detector_weights))
    dp_detector.eval()

    torch.multiprocessing.set_sharing_strategy('file_system')
    data_loader = DataLoader(ParkingSlotDataset(args.dataset_directory),
                             batch_size=args.batch_size, shuffle=True,
                             num_workers=args.data_loading_workers,
                             collate_fn=lambda x: list(zip(*x)))
    logger = util.Logger(enable_visdom=args.enable_visdom)

    total_loss = 0
    num_evaluation = 0
    ground_truths_list = []
    predictions_list = []
    for iter_idx, (image, marking_points) in enumerate(data_loader):
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
        logger.log(iter=iter_idx, total_loss=total_loss)

    precisions, recalls = util.calc_precision_recall(
        ground_truths_list, predictions_list, match_marking_points)
    average_precision = util.calc_average_precision(precisions, recalls)
    if args.enable_visdom:
        logger.plot_curve(precisions, recalls)
    logger.log(average_loss=total_loss / num_evaluation,
               average_precision=average_precision)


if __name__ == '__main__':
    evaluate_detector(config.get_parser_for_evaluation().parse_args())
