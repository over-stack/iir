import torch
from torch import nn


class BaseConvNet(nn.Module):
    def __init__(self, in_channels: int, num_filters: int, kernel_size: int, num_classes: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size, padding="same"),
            nn.Conv2d(num_filters, num_filters, kernel_size, padding="same"),
            nn.Conv2d(num_filters, num_filters, kernel_size, padding="same"),
            nn.Conv2d(num_filters, num_filters, kernel_size, padding="same"),
            nn.Conv2d(num_filters, num_classes, kernel_size, padding="same"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
