"""Train directional marking point detector."""
import random
import torch
from torch.utils.data import DataLoader
import config
from data.data_process import get_predicted_points, generate_objective
from data.dataset import ParkingSlotDataset
from model.detector import DirectionalPointDetector
from util.log import Logger
from util import tensor2im


def plot_prediction(logger, image, marking_points, prediction):
    """Plot the ground truth and prediction of a random sample in a batch."""
    rand_sample = random.randint(0, image.size(0)-1)
    sampled_image = tensor2im(image[rand_sample])
    logger.plot_marking_points(sampled_image, marking_points[rand_sample],
                               win_name='gt_marking_points')
    sampled_image = tensor2im(image[rand_sample])
    pred_points = get_predicted_points(prediction[rand_sample], 0.01)
    if pred_points:
        logger.plot_marking_points(sampled_image,
                                   list(list(zip(*pred_points))[1]),
                                   win_name='pred_marking_points')


def train_detector(args):
    """Train directional point detector."""
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device('cuda:'+str(args.gpu_id) if args.cuda else 'cpu')

    dp_detector = DirectionalPointDetector(
        3, args.depth_factor, config.NUM_FEATURE_MAP_CHANNEL).to(device)
    if args.detector_weights:
        dp_detector.load_state_dict(torch.load(args.detector_weights))

    optimizer = torch.optim.Adam(dp_detector.parameters(), lr=args.lr)
    if args.optimizer_weights:
        optimizer.load_state_dict(torch.load(args.optimizer_weights))

    logger = Logger(['train_loss'] if args.enable_visdom else None)
    data_loader = DataLoader(ParkingSlotDataset(args.dataset_directory),
                             batch_size=args.batch_size, shuffle=True,
                             num_workers=args.data_loading_workers,
                             collate_fn=lambda x: list(zip(*x)))

    for epoch_idx in range(args.num_epochs):
        for iter_idx, (image, marking_points) in enumerate(data_loader):
            image = torch.stack(image)
            image = image.to(device)

            optimizer.zero_grad()
            prediction = dp_detector(image)
            objective, gradient = generate_objective(marking_points, device)
            loss = (prediction - objective) ** 2
            loss.backward(gradient)
            optimizer.step()

            train_loss = torch.sum(loss*gradient).item() / loss.size(0)
            logger.log(epoch=epoch_idx, iter=iter_idx, train_loss=train_loss)
            if args.enable_visdom:
                plot_prediction(logger, image, marking_points, prediction)
        torch.save(dp_detector.state_dict(),
                   'weights/dp_detector_%d.pth' % epoch_idx)
        torch.save(optimizer.state_dict(), 'weights/optimizer.pth')


if __name__ == '__main__':
    train_detector(config.get_parser_for_training().parse_args())
