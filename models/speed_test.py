from time import time

import torch
from torch import nn
from tqdm import tqdm

from config import settings
import OriginalIIR
import IIRConvBlock


class IIRConvNet(nn.Module):
    def __init__(self, iir_block, in_channels: int, num_filters: int, kernel_size: int, num_classes: int):
        super().__init__()

        self.layers = nn.Sequential(
            iir_block(in_channels, num_filters, kernel_size, kernel_size),
            iir_block(num_filters, num_filters, kernel_size, kernel_size),
            iir_block(num_filters, num_filters, kernel_size, kernel_size),
            iir_block(num_filters, num_filters, kernel_size, kernel_size),
            nn.Conv2d(num_filters, num_classes, kernel_size, padding="same"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def main():
    measure_speed(IIRConvBlock.IIRConvBlock)
    measure_speed(OriginalIIR.IIRConvBlock)


def measure_speed(iir_block, iterations=3):
    net = IIRConvNet(iir_block, 1, 24, 3, 1)
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)

    stats: list[tuple[float, float, float]] = list()
    print(f"inference\t grad\t opt")
    for _ in range(iterations):
        x = torch.rand((32, 1, 128, 128))
        gt = torch.randint_like(x, 2)

        start_time = time()
        y = net(x)
        inference_time = time() - start_time

        loss = loss_func(y, gt)
        start_time = time()
        loss.backward()
        grad_time = time() - start_time

        start_time = time()
        optimizer.step()
        optimizer.zero_grad()
        opt_time = time() - start_time

        stats.append((inference_time, grad_time, opt_time))
        print(f"{inference_time:.2f}\t {grad_time:.2f}\t {opt_time:.2f}")

    print("MEAN STATS")
    print(f"inference:\t {sum([row[0] for row in stats]) / len(stats):.2f}")
    print(f"grad:\t {sum([row[1] for row in stats]) / len(stats):.2f}")
    print(f"opt:\t {sum([row[2] for row in stats]) / len(stats):.2f}")


if __name__ == "__main__":
    main()
