"""Small RGB-to-ink U-Net with full-resolution skip connections."""

import torch
from torch import nn
from torch.nn import functional as F


class _ConvBlock(nn.Sequential):
    def __init__(self, inputs, outputs):
        super().__init__(
            nn.Conv2d(inputs, outputs, 3, padding=1), nn.ReLU(),
            nn.Conv2d(outputs, outputs, 3, padding=1), nn.ReLU(),
        )


class PhotoInkRestorer(nn.Module):
    def __init__(self, base_channels=24):
        super().__init__()
        if base_channels < 1:
            raise ValueError('base_channels must be positive')
        c = base_channels
        self.encoder = nn.ModuleList([
            _ConvBlock(3, c), _ConvBlock(c, 2 * c), _ConvBlock(2 * c, 4 * c),
        ])
        self.bridge = _ConvBlock(4 * c, 8 * c)
        self.decoder = nn.ModuleList([
            _ConvBlock(12 * c, 4 * c), _ConvBlock(6 * c, 2 * c), _ConvBlock(3 * c, c),
        ])
        self.output = nn.Conv2d(c, 1, 1)

    def forward(self, photo):
        skips = []
        for block in self.encoder:
            photo = block(photo)
            skips.append(photo)
            photo = F.max_pool2d(photo, 2)
        photo = self.bridge(photo)
        for block, skip in zip(self.decoder, reversed(skips)):
            photo = F.interpolate(photo, size=skip.shape[-2:], mode='nearest')
            photo = block(torch.cat((photo, skip), dim=1))
        return self.output(photo)


def restoration_loss(logits, target):
    # Soft labels preserve antialiasing; positive weighting protects thin strokes.
    weight = target.new_tensor(3.0)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight)
    prediction = logits.sigmoid()
    axes = (1, 2, 3)
    dice = (2 * (prediction * target).sum(axes) + 1) / (
        prediction.sum(axes) + target.sum(axes) + 1
    )
    return bce + (1 - dice).mean()
