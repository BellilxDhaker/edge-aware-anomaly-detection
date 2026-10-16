"""Display heatmaps, overlays and thresholded binary masks (Phase 3, Step 3).

Separation of concerns, enforced by this module's API:

* **Evaluation data** = raw anomaly scores ``S`` (from
  ``src.anomaly.scoring``) and binary masks (``apply_threshold``). Metrics in
  ``src.evaluation.metrics`` consume only these.
* **Display data** = RGB renderings produced by ``score_to_heatmap`` /
  ``overlay_heatmap``. These are *never* fed back into scoring or metrics.

Why heatmap colors are not calibrated defect probabilities:

1. ``minmax_normalize`` maps the panel's score range onto [0, 1] so the image
   is visible; the mapping changes per map (or per batch) unless vmin/vmax are
   pinned. The same score can therefore be red in one panel and blue in
   another.
2. The colormap (``inferno`` by default) is an aesthetic choice: equal steps
   in color are not equal steps in "defect likelihood", and no training ever
   fitted these colors to outcomes.
3. Nothing here estimates ``P(defect | pixel)``: there are no labels in the
   scoring path at all. A pixel at color ``c`` means "score sits at fraction
   ``c`` of this panel's display range", nothing more.

Evaluation always uses the unmodified ``S`` and the thresholded mask, and the
notebook pins ``vmin``/``vmax`` when panels must be visually comparable.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]

__all__ = [
    "apply_threshold",
    "mask_to_rgb",
    "minmax_normalize",
    "overlay_heatmap",
    "score_to_heatmap",
    "to_numpy",
]


def to_numpy(data: ArrayLike, *, name: str = "array") -> np.ndarray:
    """Convert a torch tensor (detached, on CPU) or array to a float numpy array."""
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    array = np.asarray(data, dtype=np.float64)
    if array.ndim not in (2, 3):
        raise ValueError(f"{name} must be 2D (H, W) or 3D (B, H, W), got {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf; render finite scores only")
    return array


def minmax_normalize(
    scores: ArrayLike,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Map scores onto [0, 1] for *display only*; returns a new float64 array.

    ``vmin``/``vmax`` default to the min/max of the whole input array (so a
    batch of maps shares one scale). A constant map (``vmax == vmin``)
    normalises to zeros instead of dividing by zero. Input is never modified:
    the scores used by metrics stay byte-identical.
    """
    array = to_numpy(scores, name="scores")
    low = float(array.min()) if vmin is None else float(vmin)
    high = float(array.max()) if vmax is None else float(vmax)
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError(f"vmin/vmax must be finite, got vmin={vmin}, vmax={vmax}")
    if high < low:
        raise ValueError(f"vmax ({high}) must be >= vmin ({low})")
    if high == low:
        return np.zeros_like(array, dtype=np.float64)
    scaled = (np.clip(array, low, high) - low) / (high - low)
    return scaled.astype(np.float64)


def _get_cmap(name: str):
    from matplotlib import colormaps  # lazy: metrics workflows need no matplotlib

    try:
        return colormaps[name]
    except KeyError as exc:
        raise ValueError(f"Unknown colormap {name!r}") from exc


def score_to_heatmap(
    scores: ArrayLike,
    *,
    cmap: str = "inferno",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Render anomaly scores as an RGB image in ``[0, 1]``.

    ``(H, W) -> (H, W, 3)`` and ``(B, H, W) -> (B, H, W, 3)``. Normalization
    (min-max, optional pinned range) happens first, *then* the colormap is
    applied - colors are a display layer on top of the scores, never a
    replacement for them. Pass explicit ``vmin``/``vmax`` when several panels
    must share one color scale.
    """
    normalized = minmax_normalize(scores, vmin=vmin, vmax=vmax)
    cmap = _get_cmap(cmap)
    mapped = cmap(normalized)  # (..., 4) RGBA in [0, 1]
    return np.ascontiguousarray(mapped[..., :3], dtype=np.float64)


def _image_to_hwc(image: ArrayLike) -> np.ndarray:
    """Accept (3, H, W) CHW or (H, W, 3) HWC, float [0, 1] or uint8."""
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"image must be 3D (3,H,W) or (H,W,3), got {array.shape}")
    if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.moveaxis(array, 0, -1)          # CHW -> HWC
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] != 3:
        raise ValueError(f"image must have 1 or 3 channels, got shape {array.shape}")
    if np.issubdtype(array.dtype, np.integer):
        array = array.astype(np.float64) / 255.0
    else:
        array = array.astype(np.float64)
        if array.min() < -1e-3 or array.max() > 1.0 + 1e-3:
            raise ValueError(
                f"float images must be in [0, 1], got range "
                f"[{array.min():.3f}, {array.max():.3f}]"
            )
    return np.clip(array, 0.0, 1.0)


def overlay_heatmap(
    image: ArrayLike,
    scores: ArrayLike,
    *,
    alpha: float = 0.45,
    cmap: str = "inferno",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Blend ``score_to_heatmap(image's scores)`` over the original image.

    ``result = (1 - alpha) * image + alpha * heatmap`` in ``[0, 1]``, shape
    ``(H, W, 3)``. The blend is global (all pixels): regions without a defect
    still receive the low-score colors of the colormap, which is why the
    overlay is a *reading aid* while the binary mask (``apply_threshold``)
    remains the evaluated artifact.
    """
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    hwc = _image_to_hwc(image)
    heat = score_to_heatmap(scores, cmap=cmap, vmin=vmin, vmax=vmax)
    if heat.shape[:2] != hwc.shape[:2]:
        raise ValueError(
            f"scores shape {heat.shape[:2]} does not match image {hwc.shape[:2]}"
        )
    return (1.0 - float(alpha)) * hwc + float(alpha) * heat


def apply_threshold(scores: ArrayLike, threshold: float) -> np.ndarray:
    """Binary anomaly mask: ``mask = scores > threshold`` (strict).

    Returns a **bool** array with the input's shape (``(H, W)`` or
    ``(B, H, W)``). The threshold must be a finite, non-negative scalar - the
    same contract as the scores - and is expected to come from
    ``src.anomaly.scoring.select_percentile_threshold`` on *normal* scores
    (Step 5), never from test ground-truth masks.
    """
    array = to_numpy(scores, name="scores")
    try:
        value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"threshold must be a scalar number, got {threshold!r}") from exc
    if not np.isfinite(value):
        raise ValueError(f"threshold must be finite, got {value}")
    if value < 0.0:
        raise ValueError(
            f"threshold must be >= 0 (scores are non-negative), got {value}"
        )
    return array > value


def mask_to_rgb(mask: ArrayLike) -> np.ndarray:
    """Boolean/0-1 mask -> display RGB (white foreground, black background)."""
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    array = np.asarray(mask)
    if array.ndim not in (2, 3):
        raise ValueError(f"mask must be 2D or 3D, got {array.shape}")
    if array.size == 0:
        raise ValueError("mask is empty")
    binary = array.astype(bool)
    rgb = np.zeros((*binary.shape, 3), dtype=np.float64)
    rgb[binary] = 1.0
    return rgb
