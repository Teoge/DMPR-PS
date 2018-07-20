# -*- coding: utf-8 -*-
import os
import os.path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ParkingSlotDataset(Dataset):
    """Parking slot dataset."""
    def __init__(self, root):
        super(ParkingSlotDataset, self).__init__()
        self.root = root
        self.sample_names = []
        self.image_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        for file in os.listdir(root):
            if file.endswith(".txt"):
                self.sample_names.append(os.path.splitext(file)[0])

    def __getitem__(self, index):
        name = self.sample_names[index]
        image = Image.open(os.path.join(self.root, name+'.bmp'))
        image = self.image_transform(image)
        marking_points = []
        with open(os.path.join(self.root, name+'.txt'), 'r') as file:
            for line in file:
                marking_point = tuple([float(n) for n in line.split()])
                marking_points.append(marking_point)
        return image, marking_points

    def __len__(self):
        return len(self.sample_names)
