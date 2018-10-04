"""Defines the parking slot dataset for directional marking point detection."""
import json
import os
import os.path
import cv2 as cv
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from data.struct import MarkingPoint


class ParkingSlotDataset(Dataset):
    """Parking slot dataset."""
    def __init__(self, root):
        super(ParkingSlotDataset, self).__init__()
        self.root = root
        self.sample_names = []
        self.image_transform = ToTensor()
        for file in os.listdir(root):
            if file.endswith(".json"):
                self.sample_names.append(os.path.splitext(file)[0])

    def __getitem__(self, index):
        name = self.sample_names[index]
        image = cv.imread(os.path.join(self.root, name+'.jpg'))
        image = self.image_transform(image)
        marking_points = []
        with open(os.path.join(self.root, name + '.json'), 'r') as file:
            for label in json.load(file):
                marking_points.append(MarkingPoint(*label))
        return image, marking_points

    def __len__(self):
        return len(self.sample_names)
