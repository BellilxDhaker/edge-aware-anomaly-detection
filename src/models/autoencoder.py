"""Baseline convolutional autoencoder: encoder + decoder, trained from scratch.

This is the Phase-2 baseline of the research project. It reconstructs
defect-free industrial images; later notebooks will score reconstruction
errors to localise unseen defects. No multi-scale scoring, edge-aware loss or
pretrained feature extractor is included here on purpose - the baseline must
stay simple so later improvements can be attributed to them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from torch import nn

from .decoder import Decoder
from .encoder import Encoder

if TYPE_CHECKING:  # avoids importing config at runtime (keeps models standalone)
    from src.config import ModelConfig


class Autoencoder(nn.Module):
    """Encoder-decoder pair with a single latent tensor between the two parts.

    ``forward(x)`` returns the reconstruction ``x_hat`` with the same shape as
    ``x``. ``encode``/``decode`` are exposed separately so later anomaly
    scoring can inspect the latent code without rebuilding the model.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8, 8),
        latent_channels: int = 256,
        num_convs_per_stage: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            latent_channels=latent_channels,
            num_convs_per_stage=num_convs_per_stage,
        )
        self.decoder = Decoder(
            out_channels=in_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            latent_channels=latent_channels,
            num_convs_per_stage=num_convs_per_stage,
        )
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.downsample_factor = 2 ** len(tuple(channel_mults))

    @classmethod
    def from_config(cls, model_config: "ModelConfig") -> "Autoencoder":
        """Build an autoencoder from a validated :class:`src.config.ModelConfig`."""
        return cls(
            in_channels=model_config.in_channels,
            base_channels=model_config.base_channels,
            channel_mults=tuple(model_config.channel_mults),
            latent_channels=model_config.latent_channels,
            num_convs_per_stage=model_config.num_convs_per_stage,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Image ``(B, C, H, W)`` -> latent ``(B, latent, H/f, W/f)``."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Latent ``(B, latent, h, w)`` -> reconstruction ``(B, C, h*2**S, w*2**S)``."""
        return self.decoder(z)

    def _check_spatial(self, x: torch.Tensor) -> None:
        height, width = x.shape[-2:]
        factor = self.downsample_factor
        if height % factor != 0 or width % factor != 0:
            raise ValueError(
                f"Input spatial size {height}x{width} is not divisible by the "
                f"encoder's total downsample factor {factor}. Resize images to a "
                f"multiple of {factor} (e.g. 256)."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected a BCHW batch, got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Model expects {self.in_channels} input channels, got {x.shape[1]}"
            )
        self._check_spatial(x)
        return self.decode(self.encode(x))


def count_trainable_parameters(model: nn.Module) -> int:
    """Total number of trainable scalars (weights + BatchNorm affine terms)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_by_part(model: Autoencoder) -> dict:
    """Trainable parameter counts split into encoder and decoder (for reporting)."""
    def _count(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    return {"encoder": _count(model.encoder), "decoder": _count(model.decoder)}
