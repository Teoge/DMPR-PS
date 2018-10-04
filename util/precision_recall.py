"""Universal procedure of calculating precision and recall."""
import bisect


def match_gt_with_preds(ground_truth, predictions, match_labels):
    """Match a ground truth with every predictions and return matched index."""
    max_confidence = 0.
    matched_idx = -1
    for i, pred in enumerate(predictions):
        if match_labels(ground_truth, pred[1]) and max_confidence < pred[0]:
            max_confidence = pred[0]
            matched_idx = i
    return matched_idx


def get_confidence_list(ground_truths_list, predictions_list, match_labels):
    """Generate a list of confidence of true positives and false positives."""
    assert len(ground_truths_list) == len(predictions_list)
    true_positive_list = []
    false_positive_list = []
    num_samples = len(ground_truths_list)
    for i in range(num_samples):
        ground_truths = ground_truths_list[i]
        predictions = predictions_list[i]
        prediction_matched = [False] * len(predictions)
        for ground_truth in ground_truths:
            idx = match_gt_with_preds(ground_truth, predictions, match_labels)
            if idx >= 0:
                prediction_matched[idx] = True
                true_positive_list.append(predictions[idx][0])
            else:
                true_positive_list.append(.0)
        for idx, pred_matched in enumerate(prediction_matched):
            if not pred_matched:
                false_positive_list.append(predictions[idx][0])
    return true_positive_list, false_positive_list


def calc_precision_recall(ground_truths_list, predictions_list, match_labels):
    """Adjust threshold to get mutiple precision recall sample."""
    true_positive_list, false_positive_list = get_confidence_list(
        ground_truths_list, predictions_list, match_labels)
    true_positive_list = sorted(true_positive_list)
    false_positive_list = sorted(false_positive_list)
    thresholds = sorted(list(set(true_positive_list)))
    recalls = [0.]
    precisions = [0.]
    for thresh in reversed(thresholds):
        if thresh == 0.:
            recalls.append(1.)
            precisions.append(0.)
            break
        false_negatives = bisect.bisect_left(true_positive_list, thresh)
        true_positives = len(true_positive_list) - false_negatives
        true_negatives = bisect.bisect_left(false_positive_list, thresh)
        false_positives = len(false_positive_list) - true_negatives
        recalls.append(true_positives / (true_positives+false_negatives))
        precisions.append(true_positives / (true_positives + false_positives))
    return precisions, recalls


def calc_average_precision(precisions, recalls):
    """Calculate average precision defined in VOC contest."""
    total_precision = 0.
    for i in range(11):
        index = next(conf[0] for conf in enumerate(recalls) if conf[1] >= i/10)
        total_precision += max(precisions[index:])
    return total_precision / 11
