"""Shared evaluation protocol for comparing anomaly-detection configurations.

One function, :func:`evaluate_configuration`, scores a whole test split with a
pluggable scoring function and a threshold derived **only** from held-out
normal validation images. Notebook 04 established this protocol for the
baseline; Phase 4 reuses it unchanged for every configuration in the
comparison so that differences between configurations come from the models /
scoring functions under test, not from evaluation settings.

Protocol (fixed; identical for all configurations)
--------------------------------------------------
1. **Inference**: batched ``model(images)`` with ``reconstruct`` (eval mode,
   no labels involved).
2. **Scoring**: ``score_fn(reconstruction, images) -> (B, H, W)`` - either the
   baseline :func:`src.anomaly.scoring.anomaly_score_map` or
   :func:`src.anomaly.multiscale.multiscale_anomaly_score_map`. The same
   ``score_fn`` is used for the threshold source and for the test split.
3. **Thresholds**: pixel and image thresholds are percentiles of the score
   distribution of the **normal validation split only** (``val_paths``, all
   under ``train/good``). Test images, defective images and ground-truth
   masks never enter this step.
4. **Ground truth**: masks are loaded for evaluation only, nearest-neighbour
   resized to the score-map resolution.
5. **Metrics**: image AUROC, pooled pixel AUROC, pooled pixel AP
   (``src.evaluation.metrics``), plus per-image Dice/IoU at the pixel
   threshold (headline = defective-image mean, empty-mask convention 1.0).

The returned dictionary contains the raw score maps, thresholds and metrics -
everything a notebook needs for tables, figures and saved artifacts.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image as PILImage
from torch.utils.data import DataLoader

from ..anomaly.heatmap import apply_threshold
from ..anomaly.scoring import image_score, reconstruct, select_percentile_threshold
from ..data.dataset import NormalImageDataset
from .metrics import (
    dice_score,
    image_auroc,
    iou_score,
    pixel_average_precision,
    pixel_auroc,
)

__all__ = ["evaluate_configuration", "load_model_from_checkpoint"]

ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

_RESAMPLE_NEAREST = getattr(PILImage, "Resampling", PILImage).NEAREST


def load_model_from_checkpoint(
    checkpoint_path: Union[Path, str],
    model_cls,
    device: torch.device,
):
    """Rebuild the architecture from a Trainer checkpoint and load it strictly.

    ``model_cls`` must expose ``from_config(model_config)`` (the project's
    :class:`src.models.Autoencoder`). Returns ``(model, payload)`` with the
    model in ``eval()`` mode on *device*.
    """
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(
            f"{checkpoint_path} is not a Trainer checkpoint dict "
            "(missing 'model_state_dict')"
        )
    from ..config import load_experiment_config

    cfg = load_experiment_config(overrides=payload["config"])
    model = model_cls.from_config(cfg.model)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model, payload


def _load_masks(mask_paths: Sequence[Optional[Path]], height: int, width: int) -> np.ndarray:
    masks = np.zeros((len(mask_paths), height, width), dtype=bool)
    for index, path in enumerate(mask_paths):
        if path is None:
            continue
        with PILImage.open(path) as mask_img:
            resized = mask_img.convert("L").resize((width, height), _RESAMPLE_NEAREST)
        masks[index] = np.asarray(resized) > 127
    return masks


def evaluate_configuration(
    model: torch.nn.Module,
    test_manifest: Sequence[dict],
    val_paths: Sequence[Union[Path, str]],
    transform,
    device: torch.device,
    score_fn: ScoreFn,
    *,
    pixel_percentile: float = 99.5,
    image_percentile: float = 95.0,
    batch_size: int = 16,
) -> Dict[str, object]:
    """Score one configuration end to end; see module docstring for the protocol.

    Parameters
    ----------
    model:
        Trained autoencoder in any mode; ``reconstruct`` switches to eval and
        restores the previous mode.
    test_manifest:
        Sequence of dicts with keys ``path`` (image), ``name``, ``defect``,
        ``is_defect`` (0/1) and ``mask_path`` (Path or None) - exactly what
        notebook 04's manifest cell builds from ``scan_category``.
    val_paths:
        Held-out normal images (all under ``train/good``); the threshold
        source. Must be non-empty.
    score_fn:
        ``(reconstruction, images) -> (B, H, W)`` anomaly scores.
    """
    if not val_paths:
        raise ValueError("val_paths is empty - thresholds need normal validation images")
    for path in val_paths:
        posix = Path(path).as_posix()
        if "/train/good/" not in posix:
            raise ValueError(
                f"threshold source must live under train/good, got {path} - leakage risk"
            )
    if not test_manifest:
        raise ValueError("test_manifest is empty")

    names = [entry["name"] for entry in test_manifest]
    labels = np.array([int(entry["is_defect"]) for entry in test_manifest], dtype=np.int64)
    paths = [entry["path"] for entry in test_manifest]
    mask_paths = [entry.get("mask_path") for entry in test_manifest]

    # ---- threshold source: normal validation scores only (same score_fn) ----
    val_dataset = NormalImageDataset(list(val_paths), transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    val_chunks = []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            reconstructions = reconstruct(model, images)
            val_chunks.append(score_fn(reconstructions, images).cpu())
    val_score_maps = torch.cat(val_chunks)
    pixel_threshold = select_percentile_threshold(
        val_score_maps.flatten().numpy(), pixel_percentile
    )
    image_threshold = select_percentile_threshold(
        image_score(val_score_maps).numpy(), image_percentile
    )

    # ---- test inference: images only ----
    test_dataset = NormalImageDataset(paths, transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    started = time.perf_counter()
    score_chunks = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            reconstructions = reconstruct(model, images)
            score_chunks.append(score_fn(reconstructions, images).cpu())
    elapsed = time.perf_counter() - started
    score_maps = torch.cat(score_chunks).numpy().astype(np.float32)
    image_scores = image_score(torch.from_numpy(score_maps)).numpy()

    height, width = score_maps.shape[-2:]
    gt_masks = _load_masks(mask_paths, height, width)
    pred_masks = apply_threshold(score_maps, pixel_threshold)

    defective = labels == 1
    normal = labels == 0
    n_defective = int(defective.sum())
    n_normal = int(normal.sum())
    if n_defective == 0 or n_normal == 0:
        raise ValueError(
            f"test split must contain both classes for AUROC "
            f"(normal={n_normal}, defective={n_defective})"
        )

    dice_all = dice_score(pred_masks, gt_masks)
    iou_all = iou_score(pred_masks, gt_masks)
    normal_area = pred_masks[normal].sum(axis=(1, 2))
    missed = np.where(defective & (iou_all == 0.0))[0]

    per_type: Dict[str, Dict[str, float]] = {}
    for defect in sorted({e["defect"] for e in test_manifest if e["is_defect"]}):
        idx = np.array(
            [i for i, e in enumerate(test_manifest) if e["defect"] == defect], dtype=int
        )
        per_type[defect] = {
            "n": int(idx.size),
            "dice_mean": float(dice_all[idx].mean()),
            "iou_mean": float(iou_all[idx].mean()),
        }

    metrics = {
        "image_auroc": float(image_auroc(image_scores, labels)),
        "pixel_auroc": float(pixel_auroc(score_maps, gt_masks)),
        "pixel_ap": float(pixel_average_precision(score_maps, gt_masks)),
        "dice_defective_mean": float(dice_all[defective].mean()),
        "iou_defective_mean": float(iou_all[defective].mean()),
        "dice_all_mean": float(dice_all.mean()),
        "iou_all_mean": float(iou_all.mean()),
        "n_normal": n_normal,
        "n_defective": n_defective,
        "fp_normal_images": int((normal_area > 0).sum()),
        "fp_normal_pixels": int(normal_area.sum()),
        "missed_defects_iou_zero": [names[i] for i in missed],
        "per_defect_type": per_type,
    }

    return {
        "score_maps": score_maps,
        "image_scores": image_scores,
        "pred_masks": pred_masks,
        "gt_masks": gt_masks,
        "labels": labels,
        "names": names,
        "paths": [str(p) for p in paths],
        "pixel_threshold": float(pixel_threshold),
        "image_threshold": float(image_threshold),
        "pixel_percentile": float(pixel_percentile),
        "image_percentile": float(image_percentile),
        "n_val_source": len(val_paths),
        "inference_seconds": float(elapsed),
        "metrics": metrics,
    }
