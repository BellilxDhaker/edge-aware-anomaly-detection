"""Multi-scale anomaly scoring (Phase 4 research improvement).

Baseline scoring (:mod:`src.anomaly.scoring`) computes the channel-averaged
absolute error ``S`` at the model's full output resolution. Small defects that
live mostly in high-frequency detail can be diluted by that single-scale view;
averaging scores computed at several resolutions is a standard way to give
coarser, region-level error evidence a vote as well.

Method
------
For each integer downsample factor ``s`` in *scales*:

1. pool input and reconstruction to roughly ``1/s`` resolution
   (``torch.nn.functional.adaptive_avg_pool2d``, so any H x W works),
2. compute the baseline score ``S_s`` there with
   :func:`src.anomaly.scoring.anomaly_score_map`,
3. bilinearly upsample ``S_s`` back to the full resolution.

The final map is the weighted average

    ``S_ms(x, y) = sum_s w_s * upsample(S_s)(x, y)``

with ``w`` normalised to sum to 1 (equal weights by default). Weights and
scales are display-independent constants fixed *before* evaluation - never
tuned on test masks (same leakage rule as the Phase-3 threshold).

Contract (verified in ``notebooks/05``)
---------------------------------------
Same as the baseline scorer: whole batches ``(B, C, H, W)`` in, one finite,
non-negative ``(B, H, W)`` float map out; bad input raises ``ValueError`` /
``TypeError`` with a precise message.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from .scoring import _check_pair, _check_scores, anomaly_score_map

__all__ = ["multiscale_anomaly_score_map", "normalize_weights"]


def normalize_weights(weights: Sequence[float], name: str = "weights") -> torch.Tensor:
    """Validate positive weights and return them normalised to sum to 1."""
    if not isinstance(weights, (list, tuple)) or len(weights) == 0:
        raise ValueError(f"{name} must be a non-empty sequence of numbers, got {weights!r}")
    values = torch.tensor([float(w) for w in weights], dtype=torch.float64)
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} must all be finite, got {weights!r}")
    if bool((values <= 0).any()):
        raise ValueError(f"{name} must all be > 0, got {weights!r}")
    return (values / values.sum()).to(torch.float32)


def _validate_scales(scales: Sequence[int]) -> tuple:
    if not isinstance(scales, (list, tuple)) or len(scales) == 0:
        raise ValueError(f"scales must be a non-empty sequence of ints, got {scales!r}")
    out = []
    for s in scales:
        if not isinstance(s, int) or isinstance(s, bool) or s < 1:
            raise ValueError(f"scale factors must be ints >= 1, got {scales!r}")
        out.append(s)
    if len(set(out)) != len(out):
        raise ValueError(f"scales must be distinct, got {scales!r}")
    return tuple(out)


def multiscale_anomaly_score_map(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    scales: Sequence[int] = (1, 2, 4),
    weights: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Weighted multi-scale anomaly score ``S_ms``, shape ``(B, H, W)``.

    Parameters
    ----------
    reconstruction, target:
        Matching ``(B, C, H, W)`` float batches - the model output and the
        input image (values in ``[0, 1]`` after ``build_transform``).
    scales:
        Distinct integer downsample factors; ``1`` is the baseline full
        resolution score. The default ``(1, 2, 4)`` covers fine, medium and
        coarse error evidence at 256x256.
    weights:
        Positive per-scale weights, normalised internally. ``None`` -> equal
        weights. Fixed a priori; must not be tuned on test masks.

    Returns
    -------
    torch.Tensor
        ``(B, H, W)``, finite and non-negative - drop-in replacement for
        :func:`src.anomaly.scoring.anomaly_score_map` (same contract, so
        thresholds, AUROC and Dice pipelines need no changes).
    """
    _check_pair(reconstruction, target)
    scale_tuple = _validate_scales(scales)
    if weights is None:
        weights = [1.0] * len(scale_tuple)
    if len(weights) != len(scale_tuple):
        raise ValueError(
            f"weights length {len(weights)} != scales length {len(scale_tuple)}"
        )
    weight_tensor = normalize_weights(weights)

    height, width = reconstruction.shape[-2:]
    output = torch.zeros(
        reconstruction.shape[0], height, width, dtype=reconstruction.dtype,
        device=reconstruction.device,
    )
    for weight, scale in zip(weight_tensor, scale_tuple):
        if scale == 1:
            scores_s = anomaly_score_map(reconstruction, target)
        else:
            out_size = (max(1, height // scale), max(1, width // scale))
            pooled_recon = F.adaptive_avg_pool2d(reconstruction, out_size)
            pooled_target = F.adaptive_avg_pool2d(target, out_size)
            scores_s = anomaly_score_map(pooled_recon, pooled_target)
            if scores_s.shape[-2:] != (height, width):
                scores_s = F.interpolate(
                    scores_s.unsqueeze(1),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
        output = output + float(weight) * scores_s

    _check_scores(output, name="multiscale_anomaly_score_map output")
    return output
