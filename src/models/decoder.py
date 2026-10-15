"""Convolutional decoder of the baseline autoencoder.

The decoder mirrors :class:`src.models.encoder.Encoder`: it upsamples the
latent code ``z`` of shape ``(B, latent_channels, H / 2**S, W / 2**S)`` back to
an image ``x_hat`` of shape ``(B, out_channels, H, W)``.

Design notes:

* **Nearest-neighbour upsampling + ConvBlock** instead of
  ``ConvTranspose2d``: transposed convolutions can learn uneven overlap and
  produce a checkerboard pattern in the output; ``Upsample(mode="nearest")``
  followed by a normal convolution is a simple, well-understood way to avoid
  it while keeping the parameter count low.
* **Sigmoid output activation** matches the input normalisation produced by
  :func:`src.data.dataset.build_transform`: images are scaled to ``[0, 1]``,
  so the reconstruction must live in the same range for the MSE loss and the
  pixel-wise error maps to be meaningful.
* The output may still differ from the input: the latent code is a lossy
  compression (e.g. 8x8x256 floats versus 256x256x3), and MSE training only
  rewards "plausible normal-looking" reconstructions, not exact copies.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
from torch import nn

from .encoder import ConvBlock


class Decoder(nn.Module):
    """Mirror of :class:`Encoder`: upsample + ConvBlock per stage, then 1x1/3x3 head.

    Channel/spatial progression for the default settings (latent 256ch at
    8x8, ``base_channels=32``, ``channel_mults=(1, 2, 4, 8, 8)``)::

        stage 0: Upsample x2 + ConvBlock(256 -> 256)   8x8  -> 16x16
        stage 1: Upsample x2 + ConvBlock(256 -> 128)  16x16 -> 32x32
        stage 2: Upsample x2 + ConvBlock(128 ->  64)  32x32 -> 64x64
        stage 3: Upsample x2 + ConvBlock( 64 ->  32)  64x64 -> 128x128
        stage 4: Upsample x2 + ConvBlock( 32 ->  32) 128x128 -> 256x256
        head:    Conv2d 3x3 (32 -> out_channels) + Sigmoid          256x256
    """

    def __init__(
        self,
        out_channels: int = 3,
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

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        stages = []
        channels_in = latent_channels
        for mult in reversed(mults):  # widest stage first, mirroring the encoder
            channels_out = base_channels * mult
            stages.append(ConvBlock(channels_in, channels_out, num_convs_per_stage))
            channels_in = channels_out
        self.stages = nn.ModuleList(stages)
        self.head = nn.Conv2d(channels_in, out_channels, kernel_size=3, padding=1)
        self.activation = nn.Sigmoid()  # outputs in [0, 1], matching ToTensor inputs
        self.out_channels = out_channels

    @property
    def stage_channels(self) -> List[int]:
        return [stage.out_channels for stage in self.stages]

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            z = stage(self.upsample(z))
        return self.activation(self.head(z))
