import torch
from torch import nn

from .OriginalIIR import IIRConvBlock


class IIRConvNet(nn.Module):
    def __init__(self, in_channels: int, num_filters: int, kernel_size: int, num_classes: int):
        super().__init__()

        self.layers = nn.Sequential(
            IIRConvBlock(in_channels, num_filters, kernel_size, kernel_size),
            IIRConvBlock(num_filters, num_filters, kernel_size, kernel_size),
            IIRConvBlock(num_filters, num_filters, kernel_size, kernel_size),
            IIRConvBlock(num_filters, num_filters, kernel_size, kernel_size),
            nn.Conv2d(num_filters, num_classes, kernel_size, padding="same"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
