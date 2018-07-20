"""Train directional point detector."""
import math
import random
import torch
from torch.utils.data import DataLoader
import config
from data import ParkingSlotDataset
from detector import DirectionalPointDetector
from log import Logger
from utils import tensor2im, get_marking_points


def get_objective_from_labels(marking_points_batch, device):
    """Get regression objective and gradient for directional point detector."""
    batch_size = len(marking_points_batch)
    objective = torch.zeros(batch_size, config.NUM_FEATURE_MAP_CHANNEL,
                            config.FEATURE_MAP_SIZE, config.FEATURE_MAP_SIZE,
                            device=device)
    gradient = torch.zeros_like(objective)
    gradient[:, 0].fill_(1.)
    for batch_idx, marking_points in enumerate(marking_points_batch):
        for marking_point in marking_points:
            col = math.floor(marking_point[0] * 16)
            row = math.floor(marking_point[1] * 16)
            # Confidence Regression
            objective[batch_idx, 0, row, col] = 1.
            # Offset Regression
            offset_x = marking_point[0]*16 - col
            offset_y = marking_point[1]*16 - row
            objective[batch_idx, 1, row, col] = offset_x
            objective[batch_idx, 2, row, col] = offset_y
            # Direction Regression
            direction = marking_point[2]
            objective[batch_idx, 3, row, col] = math.cos(direction)
            objective[batch_idx, 4, row, col] = math.sin(direction)
            # Assign Gradient
            gradient[batch_idx, 1:5, row, col].fill_(1.)
    return objective, gradient


def plot_random_prediction(logger, image, marking_points, prediction):
    """Plot the ground truth and prediction of a random sample in a batch."""
    rand_sample = random.randint(0, image.size(0)-1)
    sampled_image = tensor2im(image[rand_sample])
    logger.plot_marking_points(sampled_image, marking_points[rand_sample],
                               win_name='gt_marking_points')
    sampled_image = tensor2im(image[rand_sample])
    pred_points = get_marking_points(prediction[rand_sample], 0.01)
    logger.plot_marking_points(sampled_image, pred_points,
                               win_name='pred_marking_points')


def train_detector(args):
    """Train directional point detector."""
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device("cuda:"+str(args.gpu_id) if args.cuda else "cpu")

    dp_detector = DirectionalPointDetector(3, args.depth_factor, 5).to(device)
    if args.detector_weights is not None:
        dp_detector.load_state_dict(torch.load(args.detector_weights))

    optimizer = torch.optim.Adam(dp_detector.parameters(), lr=args.lr)
    if args.optimizer_weights is not None:
        optimizer.load_state_dict(torch.load(args.optimizer_weights))

    if args.enable_visdom:
        logger = Logger(['loss'])
    else:
        logger = Logger()

    data_loader = DataLoader(ParkingSlotDataset(args.dataset_directory),
                             batch_size=args.batch_size, shuffle=True,
                             collate_fn=lambda x: list(zip(*x)))
    for epoch_idx in range(args.num_epochs):
        for iter_idx, (image, marking_points) in enumerate(data_loader):
            image = torch.stack(image)
            image = image.to(device)

            optimizer.zero_grad()
            prediction = dp_detector(image)
            objective, gradient = get_objective_from_labels(marking_points,
                                                            device)
            loss = (prediction - objective) ** 2
            loss.backward(gradient)
            optimizer.step()

            logger.log(epoch=epoch_idx, iter=iter_idx,
                       loss=torch.sum(loss * gradient).item())
            if args.enable_visdom:
                logger.plot_curve()
                plot_random_prediction(logger, image, marking_points,
                                       prediction)
        torch.save(dp_detector.state_dict(),
                   'weights/dp_detector_%d.pth' % epoch_idx)
    torch.save(optimizer.state_dict(), 'weights/optimizer.pth')


if __name__ == '__main__':
    train_detector(config.get_parser_for_training().parse_args())
