"""Evaluation metrics for reconstruction-based anomaly detection (Phase 3, Step 6).

Protocols and conventions used by this project (also stated in the notebook
whenever a number is shown):

**Image AUROC** (``image_auroc``)
    One score per image (``src.anomaly.scoring.image_score``) vs. the image
    label (defective / normal). Threshold-free ranking metric. Requires **both
    classes** - with only normal or only defective images it is undefined and
    raises :class:`ValueError`.

**Pixel AUROC** (``pixel_auroc``)
    All pixels of all test images pooled into a single ranking problem
    (scores vs. binary ground truth). Also threshold-free, also requires both
    classes (at least one defect pixel *and* one non-defect pixel).

**Pixel Average Precision** (``pixel_average_precision``)
    Same pooling, but sensitive to defect rarity: precision is integrated over
    recall increases (step-wise, ties grouped by score - the same semantics as
    ``sklearn.metrics.average_precision_score``). Requires at least one
    positive pixel; images with no defect contribute their (negative) pixels
    to the ranking, which is what makes the metric reflect rarity.

**Dice score** (``dice_score``) and **IoU** (``iou_score``)
    Overlap of a *thresholded* prediction mask with the ground truth - they are
    only interpretable together with the thresholding method (Step 5: a
    percentile of *normal validation* scores). Computed per image; the
    notebook reports the mean over **defective images only** as the headline
    number (mean over all images would be inflated by easy normal/normal
    agreements), plus the raw arrays.

**Empty-mask convention** (normal images have empty ground truth):
    both prediction and ground truth empty -> ``empty_value`` (default
    ``1.0``: both agree there is no defect); any other combination with an
    empty side -> ``0.0`` when the intersection is empty. The per-image arrays
    keep every case visible so nothing is averaged away silently.

Validation: every function is checked against hand-worked examples and,
when scikit-learn is installed, against its trusted implementations
(``roc_auc_score`` / ``average_precision_score``) on random data.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]

__all__ = [
    "binary_auroc",
    "dice_score",
    "image_auroc",
    "iou_score",
    "pixel_average_precision",
    "pixel_auroc",
]


def _to_numpy(data: ArrayLike, name: str) -> np.ndarray:
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    return np.asarray(data)


def _scores_and_labels(scores: ArrayLike, labels: ArrayLike):
    score_array = _to_numpy(scores, "scores").astype(np.float64).ravel()
    label_array = _to_numpy(labels, "labels").ravel()
    if score_array.size != label_array.size:
        raise ValueError(
            f"scores and labels must have the same number of elements, got "
            f"{score_array.size} vs {label_array.size}"
        )
    if score_array.size == 0:
        raise ValueError("empty inputs")
    if not np.isfinite(score_array).all():
        raise ValueError("scores contain NaN/Inf")
    label_array = label_array.astype(np.float64)
    if not np.isin(label_array, (0.0, 1.0)).all():
        raise ValueError("labels must be binary (0/1)")
    return score_array, label_array.astype(np.int64)


def binary_auroc(scores: ArrayLike, labels: ArrayLike) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic.

    ``AUROC = (sum of ranks of positives - n_pos*(n_pos+1)/2) / (n_pos*n_neg)``
    with average ranks assigned to tied scores, which is exactly the area
    under the step-wise ROC curve. Returns a value in ``[0, 1]``.

    Raises :class:`ValueError` when only one class is present: AUROC needs
    both positive and negative examples to be defined.
    """
    score_array, label_array = _scores_and_labels(scores, labels)
    n_pos = int(label_array.sum())
    n_neg = int(label_array.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"AUROC requires both positive and negative examples; got "
            f"n_pos={n_pos}, n_neg={n_neg}. Collect both normal and defective "
            "samples before computing AUROC."
        )

    order = np.argsort(score_array, kind="mergesort")
    sorted_scores = score_array[order]
    _, inverse, counts = np.unique(
        sorted_scores, return_inverse=True, return_counts=True
    )
    cumulative = np.cumsum(counts)
    starts = cumulative - counts                          # 0-based first index of group
    group_avg_rank = starts + (counts - 1) / 2.0 + 1.0    # 1-based, ties averaged
    ranks = np.empty(score_array.size, dtype=np.float64)  # one rank per element
    ranks[order] = group_avg_rank[inverse]

    rank_sum_pos = float(ranks[label_array == 1].sum())
    auroc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(np.clip(auroc, 0.0, 1.0))


def image_auroc(image_scores: ArrayLike, labels: ArrayLike) -> float:
    """Image-level AUROC: normal vs. defective images (alias of `binary_auroc`).

    ``image_scores[i]`` is typically ``src.anomaly.scoring.image_score(...,
    'mean')`` for test image *i*; ``labels[i]`` is 1 for defective, 0 for
    normal. Requires both classes (see :func:`binary_auroc`).
    """
    return binary_auroc(image_scores, labels)


def _pixel_inputs(score_maps: ArrayLike, gt_masks: ArrayLike):
    scores = _to_numpy(score_maps, "score_maps").astype(np.float64)
    masks = _to_numpy(gt_masks, "gt_masks")
    if scores.shape != masks.shape:
        raise ValueError(
            f"score_maps and gt_masks must have identical shapes, got "
            f"{scores.shape} vs {masks.shape}"
        )
    if scores.ndim not in (2, 3):
        raise ValueError(
            f"expected (H, W) or (N, H, W) arrays, got shape {scores.shape}"
        )
    if not np.isfinite(scores).all():
        raise ValueError("score_maps contain NaN/Inf")
    masks = _as_binary(masks, "gt_masks")
    return scores.ravel(), masks.ravel()


def _as_binary(data: ArrayLike, name: str) -> np.ndarray:
    """Boolean (or strict 0/1) mask -> bool array; floats that look like scores raise."""
    array = _to_numpy(data, name)
    if array.dtype == bool:
        return array
    as_float = array.astype(np.float64)
    if not np.isfinite(as_float).all():
        raise ValueError(f"{name} contains NaN/Inf")
    if not np.isin(as_float, (0.0, 1.0)).all():
        raise ValueError(
            f"{name} must be boolean or contain only 0/1 values, got a range of "
            f"[{as_float.min()}, {as_float.max()}] - threshold the anomaly scores "
            "first (src.anomaly.heatmap.apply_threshold)."
        )
    return as_float.astype(bool)


def _mask_pair(pred: ArrayLike, gt: ArrayLike, *, names: str):
    pred_array = _to_numpy(pred, names.split(" vs ")[0])
    gt_array = _to_numpy(gt, names.split(" vs ")[-1])
    if pred_array.shape != gt_array.shape:
        raise ValueError(
            f"{names} must have identical shapes, got "
            f"{pred_array.shape} vs {gt_array.shape}"
        )
    if pred_array.ndim not in (2, 3):
        raise ValueError(f"{names} must be (H, W) or (N, H, W)")
    if pred_array.size == 0:
        raise ValueError(f"{names} is empty")
    return _as_binary(pred_array, names.split(" vs ")[0]), _as_binary(
        gt_array, names.split(" vs ")[-1]
    )


def _check_empty_value(empty_value: float) -> float:
    try:
        value = float(empty_value)
    except (TypeError, ValueError) as exc:
        raise TypeError("empty_value must be a number") from exc
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"empty_value must be in [0, 1], got {value}")
    return value


def _overlap_counts(pred: np.ndarray, gt: np.ndarray):
    """Shared per-image intersection/union accounting with the empty convention."""
    batched_pred = pred[np.newaxis] if pred.ndim == 2 else pred
    batched_gt = gt[np.newaxis] if gt.ndim == 2 else gt
    axes = (1, 2)
    pred_area = batched_pred.sum(axis=axes).astype(np.float64)
    gt_area = batched_gt.sum(axis=axes).astype(np.float64)
    intersection = (batched_pred & batched_gt).sum(axis=axes).astype(np.float64)
    both_empty = (pred_area == 0) & (gt_area == 0)
    union = pred_area + gt_area - intersection
    return pred_area, gt_area, intersection, union, both_empty, pred.ndim == 2


def dice_score(pred: ArrayLike, gt: ArrayLike, *, empty_value: float = 1.0):
    """Per-image Dice: ``2|pred & gt| / (|pred| + |gt|)``.

    ``(H, W)`` input -> ``float``; ``(N, H, W)`` -> ``(N,)`` array. When both
    masks are empty the result is ``empty_value`` (default 1.0 - agreement on
    "no defect"); every other empty-side case scores 0 via the intersection.
    Dice uses the thresholded prediction, so always report it with the
    thresholding method (Step 5).
    """
    empty = _check_empty_value(empty_value)
    pred_array, gt_array = _mask_pair(pred, gt, names="pred vs gt")
    pred_area, gt_area, intersection, _, both_empty, scalar = _overlap_counts(
        pred_array, gt_array
    )
    denominator = pred_area + gt_area
    values = np.where(
        both_empty, empty, np.divide(2.0 * intersection, denominator, where=denominator > 0, out=np.zeros_like(denominator))
    )
    return float(values[0]) if scalar else values


def iou_score(pred: ArrayLike, gt: ArrayLike, *, empty_value: float = 1.0):
    """Per-image IoU: ``|pred & gt| / |pred ∪ gt|``.

    Same shape rules and empty-mask convention as :func:`dice_score`.
    """
    empty = _check_empty_value(empty_value)
    pred_array, gt_array = _mask_pair(pred, gt, names="pred vs gt")
    _, _, intersection, union, both_empty, scalar = _overlap_counts(
        pred_array, gt_array
    )
    values = np.where(
        both_empty, empty, np.divide(intersection, union, where=union > 0, out=np.zeros_like(union))
    )
    return float(values[0]) if scalar else values


def pixel_auroc(score_maps: ArrayLike, gt_masks: ArrayLike) -> float:
    """Pixel-level AUROC over all pixels of the given images (pooled ranking).

    Requires at least one defect pixel and one non-defect pixel; raises
    otherwise (same both-classes rule as image AUROC).
    """
    flat_scores, flat_masks = _pixel_inputs(score_maps, gt_masks)
    return binary_auroc(flat_scores, flat_masks.astype(np.int64))


def pixel_average_precision(score_maps: ArrayLike, gt_masks: ArrayLike) -> float:
    """Pixel-level average precision (pooled, tie-aware step integration).

    Scores are grouped by distinct value (each group = one operating point),
    precision/recall are computed from the cumulative counts, and AP is
    ``sum_k (recall_k - recall_{k-1}) * precision_k`` - identical semantics to
    ``sklearn.metrics.average_precision_score``. Requires at least one defect
    pixel (no positives -> AP undefined -> :class:`ValueError`).
    """
    flat_scores, flat_masks = _pixel_inputs(score_maps, gt_masks)
    labels = flat_masks.astype(np.int64)
    n_pos = int(labels.sum())
    if n_pos == 0:
        raise ValueError(
            "pixel_average_precision requires at least one defect pixel "
            "(no positive examples in the ground truth)"
        )
    if n_pos == labels.size:
        raise ValueError(
            "pixel_average_precision requires at least one non-defect pixel"
        )

    order = np.argsort(-flat_scores, kind="mergesort")   # descending
    sorted_scores = flat_scores[order]
    sorted_labels = labels[order]
    # distinct score groups -> one operating point each (ties handled together)
    new_group = np.r_[np.diff(sorted_scores) != 0, True]
    true_positives = np.cumsum(sorted_labels)[new_group].astype(np.float64)
    false_positives = np.cumsum(1 - sorted_labels)[new_group].astype(np.float64)

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / true_positives[-1]
    recall_steps = np.diff(np.r_[0.0, recall])
    ap = float(np.sum(recall_steps * precision))
    return float(np.clip(ap, 0.0, 1.0))
