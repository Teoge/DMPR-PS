"""Evaluate directional marking point detector."""
import torch
import config
import util
from data import get_predicted_points, match_marking_points, calc_point_squre_dist, calc_point_direction_angle
from data import ParkingSlotDataset
from model import DirectionalPointDetector
from train import generate_objective


def is_gt_and_pred_matched(ground_truths, predictions, thresh):
    """Check if there is any false positive or false negative."""
    predictions = [pred for pred in predictions if pred[0] >= thresh]
    prediction_matched = [False] * len(predictions)
    for ground_truth in ground_truths:
        idx = util.match_gt_with_preds(ground_truth, predictions,
                                       match_marking_points)
        if idx < 0:
            return False
        prediction_matched[idx] = True
    if not all(prediction_matched):
        return False
    return True


def collect_error(ground_truths, predictions, thresh):
    """Collect errors for those correctly detected points."""
    dists = []
    angles = []
    predictions = [pred for pred in predictions if pred[0] >= thresh]
    for ground_truth in ground_truths:
        idx = util.match_gt_with_preds(ground_truth, predictions,
                                       match_marking_points)
        if idx >= 0:
            detected_point = predictions[idx][1]
            dists.append(calc_point_squre_dist(detected_point, ground_truth))
            angles.append(calc_point_direction_angle(
                detected_point, ground_truth))
        else:
            continue
    return dists, angles


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

    psdataset = ParkingSlotDataset(args.dataset_directory)
    logger = util.Logger(enable_visdom=args.enable_visdom)

    total_loss = 0
    position_errors = []
    direction_errors = []
    ground_truths_list = []
    predictions_list = []
    for iter_idx, (image, marking_points) in enumerate(psdataset):
        ground_truths_list.append(marking_points)

        image = torch.unsqueeze(image, 0).to(device)
        prediction = dp_detector(image)
        objective, gradient = generate_objective([marking_points], device)
        loss = (prediction - objective) ** 2
        total_loss += torch.sum(loss*gradient).item()

        pred_points = get_predicted_points(prediction[0], 0.01)
        predictions_list.append(pred_points)

        dists, angles = collect_error(marking_points, pred_points,
                                      config.CONFID_THRESH_FOR_POINT)
        position_errors += dists
        direction_errors += angles

        logger.log(iter=iter_idx, total_loss=total_loss)

    precisions, recalls = util.calc_precision_recall(
        ground_truths_list, predictions_list, match_marking_points)
    average_precision = util.calc_average_precision(precisions, recalls)
    if args.enable_visdom:
        logger.plot_curve(precisions, recalls)
    logger.log(average_loss=total_loss / len(psdataset),
               average_precision=average_precision)


if __name__ == '__main__':
    evaluate_detector(config.get_parser_for_evaluation().parse_args())
