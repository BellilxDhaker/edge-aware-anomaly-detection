"""Reconstruction losses for the baseline autoencoder.

The Phase-2 baseline uses plain mean squared error (MSE) between the input
image and its reconstruction:

.. math::

    \\mathcal{L}_{MSE} = \\frac{1}{N} \\sum_{i=1}^{N} (x_i - \\hat{x}_i)^2

where ``x`` is the input image tensor (values in ``[0, 1]`` after
``build_transform``), ``x_hat`` is the decoder output (values in ``[0, 1]``
after the sigmoid activation) and the sum runs over every element of the
batch. A single scalar is returned per call; the trainer averages those
scalars weighted by batch size to obtain the epoch loss.

Edge-aware or multi-scale terms are intentionally NOT implemented here yet -
they belong to the later research phases and must be compared against this
baseline, not mixed into it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _check_shapes(reconstruction: torch.Tensor, target: torch.Tensor) -> None:
    if reconstruction.shape != target.shape:
        raise ValueError(
            "Reconstruction and target must have identical shapes for MSE, got "
            f"{tuple(reconstruction.shape)} vs {tuple(target.shape)}. "
            "Check that the decoder output size matches the input resolution."
        )
    if reconstruction.numel() == 0:
        raise ValueError("Empty tensors passed to the reconstruction loss")


def mse_reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Mean squared error between reconstruction and target.

    ``reduction='mean'`` returns the scalar mean over all batch elements;
    ``'sum'`` and ``'none'`` are available for diagnostics.
    """
    _check_shapes(reconstruction, target)
    return F.mse_loss(reconstruction, target, reduction=reduction)


def per_image_mse(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample MSE of shape ``(B,)`` - useful for per-image statistics."""
    _check_shapes(reconstruction, target)
    return (reconstruction - target).flatten(1).pow(2).mean(dim=1)


def absolute_error_map(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-pixel ``|x_hat - x|`` averaged over channels, shape ``(B, H, W)``.

    This is the map later anomaly scoring will build on; here it is only used
    to visualise where a normal image is reconstructed well or poorly.
    """
    _check_shapes(reconstruction, target)
    return (reconstruction - target).abs().mean(dim=1)


class ReconstructionLoss(nn.Module):
    """Thin module wrapper around :func:`mse_reconstruction_loss`.

    Being an ``nn.Module`` lets the trainer treat the criterion uniformly
    (e.g. a future edge-aware loss can be swapped in with the same interface).
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unsupported reduction {reduction!r}")
        self.reduction = reduction

    def forward(self, reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return mse_reconstruction_loss(reconstruction, target, self.reduction)

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}"
