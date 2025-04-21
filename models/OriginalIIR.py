import time

import torch
import torch.nn as nn
import torch.nn.functional as F

LEFT2RIGHT = 0
RIGHT2LEFT = 1
TOP2BOTTOM = 2
BOTTOM2TOP = 3


class IIRConv2d(nn.Module):
    def __init__(
        self,
        num_channels: int,
        conv1_window: int,
        conv2_window: int,
        direction: int,
    ):
        super(IIRConv2d, self).__init__()

        assert conv1_window % 2 == 1
        assert conv2_window % 2 == 1

        self.conv1_window = conv1_window
        self.conv2_window = conv2_window

        self.conv1_weight = nn.Parameter(
            torch.empty(num_channels, 1, self.conv1_window)
        )
        self.conv2_weight = nn.Parameter(
            torch.empty(num_channels, self.conv2_window)
        )

        # self.conv2_padding = nn.ZeroPad2d((conv2_window, 0, 0, 0))
        self.direction = direction

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.uniform_(self.conv1_weight, 0.1, 0.2)
        nn.init.uniform_(self.conv2_weight, 0.1, 0.2)

    def forward(self, x: torch.Tensor):
        assert x.dim() == 4, "Input must be a 4D tensor"
        match self.direction:
            case 0:
                pass
            case 1:
                x = x.flip(3)
            case 2:
                x = x.transpose(2, 3)
            case 3:
                x = x.transpose(2, 3).flip(3)

        batch_size, channels, height, width = x.size()
        x = x.transpose(1, 2).reshape(batch_size * height, channels, -1).contiguous()
        x = F.conv1d(
            x, self.conv1_weight, padding=self.conv1_window // 2, groups=channels
        )
        x = F.pad(x, (self.conv2_window - 1, 0), "constant", 0)  # BxCxHxW -> BxCxHx(W+2)

        x = x.transpose(0, 2).contiguous()
        for j in range(self.conv2_window, width):
            x[j] -= torch.einsum(
                "wch,cw->ch", x[j - self.conv2_window: j].clone(), self.conv2_weight
            )

        x = (
            x.narrow(0, self.conv2_window - 1, width)
            .transpose(0, 2)
            .reshape(batch_size, height, channels, width)
            .transpose(1, 2)
        )

        match self.direction:
            case 0:
                pass
            case 1:
                x = x.flip(3)
            case 2:
                x = x.transpose(2, 3)
            case 3:
                x = x.flip(3).transpose(2, 3)
        return x


class IIRConvBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        conv1_window: int,
        conv2_window: int,
    ) -> None:
        super(IIRConvBlock, self).__init__()
        self.r2l_iir_conv = IIRConv2d(
            num_channels=input_channels,
            conv1_window=conv1_window,
            conv2_window=conv2_window,
            direction=RIGHT2LEFT,
        )
        self.l2r_iir_conv = IIRConv2d(
            num_channels=input_channels,
            conv1_window=conv1_window,
            conv2_window=conv2_window,
            direction=LEFT2RIGHT,
        )
        self.t2b_iir_conv = IIRConv2d(
            num_channels=input_channels,
            conv1_window=conv1_window,
            conv2_window=conv2_window,
            direction=TOP2BOTTOM,
        )
        self.b2t_iir_conv = IIRConv2d(
            num_channels=input_channels,
            conv1_window=conv1_window,
            conv2_window=conv2_window,
            direction=BOTTOM2TOP,
        )
        self.conv_1x1 = nn.Conv2d(input_channels * 4, output_channels, kernel_size=1)

    def forward(self, x):
        branch_r2l = self.r2l_iir_conv(x)
        branch_l2r = self.l2r_iir_conv(x)
        branch_t2b = self.t2b_iir_conv(x)
        branch_b2t = self.b2t_iir_conv(x)

        return self.conv_1x1(
            torch.cat([branch_r2l, branch_l2r, branch_t2b, branch_b2t], 1)
        )


if __name__ == "__main__":
    n = 20
    clocks = [0.] * n
    x = torch.rand((64, 3, 128, 128))
    # iirc2d = IIRConv2d(3, 3, 3, 0)
    iirc2d = IIRConvBlock(3, 5, 3, 3)
    for i in range(n):
        t1 = time.time()
        y = iirc2d(x)
        t2 = time.time()
        clocks[i] = t2 - t1
    print(sum(clocks) / n)
