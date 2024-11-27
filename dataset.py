import os
from PIL import Image

from natsort import natsorted
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision

from config import settings
from utils import Mode


class DIBCO2017Dataset(Dataset):
    def __init__(self, mode: Mode, images_path: str, transform=None):
        self.mode = mode
        self.images_path = images_path
        self.filenames = natsorted(os.listdir(self.images_path))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int):
        filename = self.filenames[index]

        image_path = os.path.join(self.images_path, filename)
        canvas = np.array(Image.open(image_path).convert("L"))
        assert len(canvas.shape) == 2, "Image shape dim != 2"

        if self.mode != Mode.TEST:
            image = canvas[:settings.height]
            mask = canvas[settings.height:]

            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']

            return image, (mask.unsqueeze(0) != 0).to(dtype=torch.float32)
        else:
            image = canvas
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']

            return image



