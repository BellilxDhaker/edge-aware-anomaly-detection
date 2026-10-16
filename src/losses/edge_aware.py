"""Edge-aware reconstruction loss (Phase 4 research improvement).

Motivation
----------
Plain MSE (:mod:`src.losses.reconstruction`) penalises intensity error
uniformly. Structure that defines "normal appearance" - object contours,
texture boundaries - is carried by spatial *gradients*; an autoencoder that
reproduces gradient structure faithfully tends to blur defects less
ambiguously at their boundaries. The edge-aware loss adds a gradient-matching
term next to the pixel term:

.. math::

    \\mathcal{L} = \\underbrace{\\frac{1}{N}\\sum_i (x_i - \\hat{x}_i)^2}_{\\text{MSE}}
                   + \\lambda \\underbrace{\\frac{1}{N}\\sum_i
                     (G(x)_i - G(\\hat{x})_i)^2}_{\\text{gradient MSE}}

where ``G`` is the channel-averaged Sobel gradient *magnitude* (``spatial_gradient_magnitude``)
and ``lambda = edge_weight`` is a fixed a-priori constant (default 0.5), never
tuned on test data. ``edge_weight = 0`` recovers plain MSE exactly, so the
module is a strict generalisation of :class:`src.losses.ReconstructionLoss`.

Sobel kernels are normalised by 8 (the kernel's L1 norm), keeping gradient
magnitudes on a scale comparable to pixel values in ``[0, 1]`` - which is
what makes ``lambda = O(1)`` a meaningful prior. ``sqrt(x^2 + 1e-12)`` inside
the magnitude keeps the backward pass finite at zero-gradient pixels (a
plain ``sqrt`` would produce Inf gradients on flat regions).

Contract: ``forward(reconstruction, target)`` returns a scalar tensor, matching
the criterion interface :class:`src.training.Trainer` already expects.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .reconstruction import _check_shapes, mse_reconstruction_loss

__all__ = [
    "EdgeAwareReconstructionLoss",
    "gradient_magnitude_mse",
    "spatial_gradient_magnitude",
]

_SOBEL_X = torch.tensor(
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
) / 8.0
_SOBEL_Y = _SOBEL_X.t()


def _sobel_kernels(channels: int, dtype, device):
    kernel_x = _SOBEL_X.to(device=device, dtype=dtype).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    kernel_y = _SOBEL_Y.to(device=device, dtype=dtype).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    return kernel_x, kernel_y


def spatial_gradient_magnitude(images: torch.Tensor) -> torch.Tensor:
    """Channel-averaged Sobel gradient magnitude ``G``, shape ``(B, H, W)``.

    Each channel's gradients are computed with 3x3 Sobel kernels (normalised
    by 8, replicate padding to avoid artificial border edges), combined as
    ``sqrt(Gx^2 + Gy^2)``, then averaged over channels - one non-negative
    scalar per pixel, matching the shape of an anomaly score map.
    """
    if not torch.is_tensor(images):
        raise TypeError(f"images must be a torch.Tensor, got {type(images).__name__}")
    if images.ndim != 4:
        raise ValueError(f"images must be a (B, C, H, W) batch, got shape {tuple(images.shape)}")
    if images.numel() == 0:
        raise ValueError("images is empty")
    if not torch.is_floating_point(images):
        raise TypeError(f"images must be floating-point, got {images.dtype}")
    if not bool(torch.isfinite(images).all()):
        raise ValueError("images contain NaN or Inf")

    channels = images.shape[1]
    kernel_x, kernel_y = _sobel_kernels(channels, images.dtype, images.device)
    padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
    grad_x = F.conv2d(padded, kernel_x, groups=channels)
    grad_y = F.conv2d(padded, kernel_y, groups=channels)
    magnitude = (grad_x.pow(2) + grad_y.pow(2) + 1e-12).sqrt()
    return magnitude.mean(dim=1)


def gradient_magnitude_mse(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE between gradient magnitudes ``G(x_hat)`` and ``G(x)`` (scalar)."""
    _check_shapes(reconstruction, target)
    return F.mse_loss(
        spatial_gradient_magnitude(reconstruction),
        spatial_gradient_magnitude(target),
    )


class EdgeAwareReconstructionLoss(nn.Module):
    """``MSE + edge_weight * gradient_magnitude_MSE``.

    Drop-in replacement for :class:`src.losses.ReconstructionLoss` inside
    :class:`src.training.Trainer`: same ``forward(reconstruction, target)``
    signature returning a scalar. ``edge_weight=0.0`` is exactly plain MSE
    (verified in ``notebooks/05``).
    """

    def __init__(self, edge_weight: float = 0.5, reduction: str = "mean") -> None:
        super().__init__()
        if not isinstance(edge_weight, (int, float)) or isinstance(edge_weight, bool):
            raise TypeError(f"edge_weight must be a number, got {edge_weight!r}")
        edge_weight = float(edge_weight)
        if edge_weight < 0.0:
            raise ValueError(f"edge_weight must be >= 0, got {edge_weight}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unsupported reduction {reduction!r}")
        self.edge_weight = edge_weight
        self.reduction = reduction

    def forward(self, reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pixel_loss = mse_reconstruction_loss(reconstruction, target, self.reduction)
        edge_loss = gradient_magnitude_mse(reconstruction, target)
        return pixel_loss + self.edge_weight * edge_loss

    def extra_repr(self) -> str:
        return f"edge_weight={self.edge_weight}, reduction={self.reduction!r}"
