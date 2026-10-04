"""RDANet model definition."""

from typing import Sequence

import torch
import torch.nn as nn

from .modules.msad import MSAD
from .modules.pgsm import PGSM


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_atten=True):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = None
        self.use_atten = use_atten
        if use_atten:
            self.ca = ChannelAttention(out_channels)
            self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_atten:
            out = self.ca(out) * out
            out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out


class RDANet(nn.Module):
    def __init__(
        self,
        input_channels,
        in_size,
        layers=4,
        factor=2,
        block_factor=1,
        block=ResidualBlock,
        mem_slots: Sequence[int] = (256, 128, 64, 64),
        patch: Sequence[int] = (16, 16, 8, 8),
        topk: Sequence[int] = (16, 16, 8, 8),
        drop_out: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
        use_overlap: Sequence[bool] = (True, True, False, False),
    ):
        super().__init__()
        assert block is not None, "Please provide a residual block class"

        base_channels = [8, 16, 32, 64, 128, 256]
        assert layers <= len(base_channels), "layers exceeds the channel table"
        self.block_factor = block_factor
        chs = [c * factor for c in base_channels[:layers]]

        self.skip_layers = nn.ModuleList(
            PGSM(
                in_channels=chs[i],
                in_size=in_size // (2**i),
                patch=patch[i],
                mem_slots=mem_slots[i],
                topk=topk[i],
                mid_channels=max(16, chs[i] // 2),
                use_overlap=use_overlap[i],
            )
            for i in range(layers - 1)
        )

        self.down_layers = nn.ModuleList(
            MSAD(in_ch=chs[i]) for i in range(layers - 1)
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_init = nn.Conv2d(input_channels, chs[0], kernel_size=1, stride=1)

        self.encoder = nn.ModuleList()
        self.encoder.append(self._make_layer(chs[0], chs[0], block))
        for i in range(layers - 2):
            self.encoder.append(
                self._make_layer(chs[i], chs[i + 1], block, block_num=2)
            )

        self.bottleneck = self._make_layer(
            chs[layers - 2], chs[layers - 1], block, block_num=2
        )

        self.decoder = nn.ModuleList()
        for i in range(layers - 2, 0, -1):
            in_ch = chs[i] + chs[i + 1]
            out_ch = chs[i]
            self.decoder.append(
                self._make_layer(in_ch, out_ch, block, block_num=2)
            )
        self.decoder.append(
            self._make_layer(chs[0] + chs[1], chs[0], block, block_num=1)
        )
        self.head = nn.Conv2d(chs[0], 1, 3, 1, 1)

    def _make_layer(self, in_channels, out_channels, block, block_num=1):
        block_num = block_num * self.block_factor
        layers = [block(in_channels, out_channels)]
        for _ in range(block_num - 1):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        skips = []
        x = self.conv_init(x)
        for idx, layer in enumerate(self.encoder):
            x = layer(x)
            skip = x
            if idx < len(self.skip_layers):
                skips.append(self.skip_layers[idx](skip))
            else:
                skips.append(skip)
            x = self.down_layers[idx](x)

        x = self.bottleneck(x)

        for layer in self.decoder:
            skip = skips.pop()
            x = self.up(x)
            x = torch.cat([skip, x], dim=1)
            x = layer(x)

        return self.head(x)


__all__ = ["RDANet", "ResidualBlock", "ChannelAttention", "SpatialAttention"]
