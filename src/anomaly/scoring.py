"""Pixel-wise anomaly scoring from reconstruction error (Phase 3, baseline).

Three different objects exist in this phase and must never share a name:

* **reconstruction image** ``x_hat``: the raw model output - still an RGB
  image that can be displayed next to the input.
* **anomaly score map** ``S``: one non-negative scalar per pixel derived from
  the input and its reconstruction. Evaluation (thresholding, AUROC, AP,
  Dice) consumes *only* this.
* **anomaly heatmap** (``src/anomaly/heatmap.py``, Step 3): a colored
  *rendering* of ``S`` for humans. Its colors are display-normalised, not
  calibrated defect probabilities.

Baseline scoring - the two formulas this module implements::

    A(x, y, c) = |I(x, y, c) - x_hat(x, y, c)|    # per-channel absolute error
    S(x, y)    = (1 / C) * sum_c A(x, y, c)        # channel-averaged score

Because inputs are float32 in [0, 1] (``build_transform``) and the decoder
ends in a sigmoid, both ``A`` and ``S`` are in [0, 1]: ``S(x, y)`` reads as
the mean absolute intensity error of one pixel.

Contract kept by every function here (verified in ``notebooks/04``):

* accepts whole batches ``(B, C, H, W)`` - no per-image Python loop,
* returns finite, non-negative floating-point tensors,
* raises ``ValueError`` / ``TypeError`` with a precise message on bad input
  (wrong rank, shape mismatch, NaN/Inf, integer dtype) instead of silently
  producing garbage scores.

Multi-scale and edge-aware scoring are intentionally NOT implemented here -
they are later research phases that must be compared against this baseline.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch

from ..losses import absolute_error_map

ArrayLike = Union[np.ndarray, torch.Tensor]

__all__ = [
    "anomaly_score_map",
    "image_score",
    "pixelwise_absolute_error",
    "reconstruct",
    "select_percentile_threshold",
]


def _require_image_batch(tensor: torch.Tensor, name: str) -> None:
    """Validate one ``(B, C, H, W)`` floating-point batch."""
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim != 4:
        raise ValueError(
            f"{name} must be a (B, C, H, W) batch, got shape {tuple(tensor.shape)}"
        )
    if tensor.numel() == 0:
        raise ValueError(f"{name} is empty")
    if not torch.is_floating_point(tensor):
        raise TypeError(
            f"{name} must be a floating-point tensor, got {tensor.dtype}. "
            "Convert with build_transform (ToTensor -> float32 in [0, 1])."
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(
            f"{name} contains NaN or Inf - the model output or input is corrupt; "
            "refusing to score it."
        )


def _check_pair(reconstruction: torch.Tensor, target: torch.Tensor) -> None:
    _require_image_batch(reconstruction, "reconstruction")
    _require_image_batch(target, "target")
    if reconstruction.shape != target.shape:
        raise ValueError(
            "reconstruction and target must have identical shapes, got "
            f"{tuple(reconstruction.shape)} vs {tuple(target.shape)}. "
            "Check that the decoder output matches the input resolution."
        )


def _check_scores(scores: torch.Tensor, name: str = "anomaly scores") -> None:
    """Validate a ``(B, H, W)`` anomaly score map."""
    if not torch.is_tensor(scores):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(scores).__name__}")
    if scores.ndim != 3:
        raise ValueError(
            f"{name} must be a (B, H, W) map, got shape {tuple(scores.shape)}"
        )
    if scores.numel() == 0:
        raise ValueError(f"{name} is empty")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError(
            f"{name} contains NaN or Inf - scoring must always produce finite values"
        )
    if bool((scores < 0).any()):
        raise ValueError(
            f"{name} contains negative values, which |I - x_hat| can never produce "
            "- implementation bug or corrupted input."
        )


def pixelwise_absolute_error(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-channel absolute error ``A(x, y, c) = |x_hat - I|``, shape ``(B, C, H, W)``.

    The full three-dimensional error field. ``anomaly_score_map`` collapses it
    across channels; keep this function for diagnostics that need one channel
    at a time.
    """
    _check_pair(reconstruction, target)
    return (reconstruction - target).abs()


def anomaly_score_map(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Channel-averaged anomaly score ``S(x, y)``, shape ``(B, H, W)``.

    ``S(x, y) = (1 / C) * sum_c |x_hat(x, y, c) - I(x, y, c)|`` - the baseline
    pixel-wise score of this project.

    Delegates the reduction to :func:`src.losses.absolute_error_map` (the
    Phase-2 primitive, ``src/losses/reconstruction.py:59``) after validating
    the inputs, so training-time visualisations and evaluation-time scores
    can never diverge. The notebook checks both paths against a hand-computed
    example.
    """
    _check_pair(reconstruction, target)
    scores = absolute_error_map(reconstruction, target)
    _check_scores(scores, name="anomaly_score_map output")
    return scores


def image_score(scores: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Aggregate a ``(B, H, W)`` score map to one score per image, shape ``(B,)``.

    ``reduction='mean'``
        Average anomaly evidence over all pixels. Robust default; used for
        image-level AUROC (Step 6).
    ``reduction='max'``
        The single hottest pixel. Sensitive to small, intense hotspots but
        equally sensitive to one-pixel artifacts.

    Thresholding is deliberately not done here - binary masks belong to
    ``src/anomaly/heatmap.py`` (Step 3) with a threshold chosen in Step 5.
    """
    _check_scores(scores)
    if reduction not in ("mean", "max"):
        raise ValueError(f"reduction must be 'mean' or 'max', got {reduction!r}")
    flat = scores.flatten(1)
    if reduction == "mean":
        return flat.mean(dim=1)
    return flat.max(dim=1).values


@torch.no_grad()
def reconstruct(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Batched inference forward pass: ``images -> reconstructions``.

    Switches *model* to ``eval()`` for the call and restores its previous
    mode afterwards, so a stray training-mode model cannot leak batch
    statistics into evaluation. The output must be a finite float batch with
    the input's shape; anything else raises instead of poisoning every score
    computed downstream.
    """
    _require_image_batch(images, "images")
    was_training = model.training
    model.eval()
    try:
        output = model(images)
    finally:
        model.train(was_training)
    if not torch.is_tensor(output) or output.shape != images.shape:
        got = tuple(output.shape) if torch.is_tensor(output) else type(output).__name__
        raise ValueError(
            f"model must return a tensor shaped like its input "
            f"{tuple(images.shape)}, got {got}"
        )
    _require_image_batch(output, "model output")
    return output


def select_percentile_threshold(scores: ArrayLike, percentile: float) -> float:
    """Derive a decision threshold as a high percentile of *scores*.

    Parameters
    ----------
    scores:
        Anomaly scores of a **normal-only** population - typically the pixel
        scores ``S`` of the held-out normal validation split (see Step 5).
        Accepts a torch tensor or numpy array of any shape.
    percentile:
        0 < percentile < 100. ``99.5`` keeps the top 0.5% of normal pixel
        scores as the "anomalous" region budget.

    Returns
    -------
    float
        The threshold ``t`` such that ``scores > t`` marks pixels as anomalous
        (see :func:`src.anomaly.heatmap.apply_threshold`).

    Leakage rule (why this function exists in this form)
    ----------------------------------------------------
    Feeding **test-set** scores - or anything derived from test ground-truth
    masks - into this function and then reporting Dice/IoU on the same test
    set would make those numbers optimistic, not unbiased. The threshold must
    come from normal training/validation data only, *chosen before* looking at
    any defective image. AUROC and average precision are threshold-free and
    are unaffected by this choice.
    """
    if not isinstance(percentile, (int, float)) or isinstance(percentile, bool):
        raise TypeError(f"percentile must be a number, got {percentile!r}")
    percentile = float(percentile)
    if not 0.0 < percentile < 100.0:
        raise ValueError(f"percentile must be in (0, 100), got {percentile}")
    if torch.is_tensor(scores):
        array = scores.detach().cpu().numpy()
    else:
        array = np.asarray(scores)
    array = np.asarray(array, dtype=np.float64).ravel()
    if array.size == 0:
        raise ValueError("scores is empty - nothing to derive a threshold from")
    if not np.isfinite(array).all():
        raise ValueError("scores contain NaN/Inf - threshold refused")
    if (array < 0).any():
        raise ValueError("scores contain negative values - threshold refused")
    return float(np.percentile(array, percentile))
