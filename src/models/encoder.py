"""Convolutional encoder of the baseline autoencoder (random init, from scratch).

The encoder maps an image ``x`` of shape ``(B, in_channels, H, W)`` to a latent
code ``z`` of shape ``(B, latent_channels, H / 2**S, W / 2**S)`` where ``S`` is
the number of pooling stages. Every stage is an ordinary
``Conv2d -> BatchNorm2d -> ReLU`` stack (twice per stage by default) followed
by a ``MaxPool2d(2)`` downsample; nothing here uses pretrained weights or a
copied architecture - it is written for this project.

Design notes:

* **No skip connections.** A U-Net style autoencoder could copy the input
  almost losslessly and would defeat the anomaly-detection hypothesis
  (defects would be reconstructed as well as normal structure). The decoder
  only ever sees the compressed latent code.
* **BatchNorm2d** stabilises the activation scales of the randomly
  initialised stack, which makes a from-scratch run far more trainable.
* **MaxPool2d** for downsampling (instead of strided convolutions) keeps the
  architecture simple and the parameter count small; the decoder mirrors it
  with nearest-neighbour upsampling.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
from torch import nn


class ConvBlock(nn.Module):
    """``num_convs`` x (Conv2d 3x3 -> BatchNorm2d -> ReLU), spatial size unchanged.

    The first convolution maps ``in_channels -> out_channels``; the remaining
    ones keep ``out_channels``. Padding=1 with a 3x3 kernel preserves H and W.
    """

    def __init__(self, in_channels: int, out_channels: int, num_convs: int = 2) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1 or num_convs < 1:
            raise ValueError(
                f"ConvBlock needs positive channels/num_convs, got "
                f"in={in_channels}, out={out_channels}, num_convs={num_convs}"
            )
        layers = []
        for index in range(num_convs):
            layers.append(
                nn.Conv2d(
                    in_channels if index == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,  # BatchNorm2d provides the bias term
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    """Stack of ConvBlocks with 2x max-pooling after each stage.

    Channel/spatial progression for the default
    ``base_channels=32, channel_mults=(1, 2, 4, 8, 8)`` and a 256x256 input::

        stage 0: ConvBlock(  3 ->  32)  256x256  -> pool ->  32ch 128x128
        stage 1: ConvBlock( 32 ->  64)  128x128  -> pool ->  64ch  64x64
        stage 2: ConvBlock( 64 -> 128)   64x64   -> pool -> 128ch  32x32
        stage 3: ConvBlock(128 -> 256)   32x32   -> pool -> 256ch  16x16
        stage 4: ConvBlock(256 -> 256)   16x16   -> pool -> 256ch   8x8
        bottleneck: Conv2d 1x1 (256 -> latent_channels)            8x8

    The 1x1 bottleneck convolution only mixes channels; it does not change the
    spatial size, so the latent keeps a compact ``8 x 8`` spatial layout that
    the decoder can upsample symmetrically.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        channel_mults: Sequence[int] = (1, 2, 4, 8, 8),
        latent_channels: int = 256,
        num_convs_per_stage: int = 2,
    ) -> None:
        super().__init__()
        mults = tuple(int(m) for m in channel_mults)
        if not mults:
            raise ValueError("channel_mults must contain at least one stage")
        self.channel_mults = mults
        self.base_channels = base_channels
        self.num_downsamples = len(mults)

        stages = []
        channels_in = in_channels
        for mult in mults:
            channels_out = base_channels * mult
            stages.append(ConvBlock(channels_in, channels_out, num_convs_per_stage))
            channels_in = channels_out
        self.stages = nn.ModuleList(stages)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = nn.Conv2d(channels_in, latent_channels, kernel_size=1)
        self.out_channels = latent_channels

    @property
    def stage_channels(self) -> List[int]:
        return [stage.out_channels for stage in self.stages]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = self.pool(stage(x))
        return self.bottleneck(x)
