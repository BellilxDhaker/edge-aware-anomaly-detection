"""Anomaly scoring, heatmaps and thresholding for reconstruction-based detection.

Phase 3: :mod:`src.anomaly.scoring` (baseline pixel-wise scores) and
:mod:`src.anomaly.heatmap` (display renderings, overlays, thresholded masks).
Phase 4: :mod:`src.anomaly.multiscale` - weighted multi-scale anomaly scoring.
"""

from .heatmap import (  # noqa: F401
    apply_threshold,
    mask_to_rgb,
    minmax_normalize,
    overlay_heatmap,
    score_to_heatmap,
    to_numpy,
)
from .multiscale import multiscale_anomaly_score_map, normalize_weights  # noqa: F401
from .scoring import (  # noqa: F401
    anomaly_score_map,
    image_score,
    pixelwise_absolute_error,
    reconstruct,
    select_percentile_threshold,
)

__all__ = [
    "anomaly_score_map",
    "apply_threshold",
    "image_score",
    "mask_to_rgb",
    "minmax_normalize",
    "multiscale_anomaly_score_map",
    "normalize_weights",
    "overlay_heatmap",
    "pixelwise_absolute_error",
    "reconstruct",
    "score_to_heatmap",
    "select_percentile_threshold",
    "to_numpy",
]
