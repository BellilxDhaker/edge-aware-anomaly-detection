"""Dataset discovery, validation and PyTorch loading for MVTec AD.

Inspection helpers (Stages 1-2, used by ``notebooks/02_dataset_inspection.ipynb``):

* :func:`find_dataset_root` - locate the folder that directly contains the
  category directories (``bottle/``, ``cable/``, ...). The extraction layout is
  probed, never assumed.
* :func:`scan_category` - enumerate normal/defective images and ground-truth
  masks for one category and verify every file (readable, sizes consistent,
  mask present).
* :func:`summarize_layout` - per-split, per-defect file counts for every
  category.

Training data pipeline (Stage 3, used by ``notebooks/03_baseline_autoencoder.ipynb``):

* :func:`list_image_files` / :func:`split_train_val` - deterministic file
  listing and reproducible train/validation splitting of NORMAL images only.
* :class:`NormalImageDataset` - map-style ``torch.utils.data.Dataset`` that
  loads RGB images, resizes them and returns float32 tensors in ``[0, 1]``.
* :func:`create_dataloaders` - builds the training (and optional validation)
  ``DataLoader``s straight from ``<category>/train/good``.

Rules honoured by this module:

* The inspection helpers must stay importable before dependencies are
  verified, so PyTorch/torchvision are imported behind a guard: the
  training dataset classes fail fast with a clear message when torch is
  missing, while discovery/validation keep working.
* Problems are returned in the report - nothing is ever skipped silently.
* The official split is never modified: training material comes from
  ``train/good`` only; ``test/`` and ``ground_truth/`` are for inspection and
  later evaluation only.
"""

from __future__ import annotations

import os
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from PIL import Image

try:  # torch is optional at import time; needed only for the training pipeline
    import torch
    from torch.utils.data import DataLoader, Dataset as _TorchDataset
except ImportError:  # pragma: no cover - exercised only in torch-less environments
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    _TorchDataset = object  # type: ignore[assignment,misc]

CATEGORIES: Tuple[str, ...] = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)

GOOD = "good"
SPLITS: Tuple[str, ...] = ("train", "test", "ground_truth")

# Directories the archive or the Kaggle download may nest the data in.
_WRAPPER_DIRS: Tuple[str, ...] = (
    "mvtec_anomaly_detection",
    "MVTecAD",
    "mvtec_ad",
    "extracted",
    "kaggle",
)

# Fallback locations when neither MVTEC_ROOT nor a start path was given.
_KNOWN_STARTS: Tuple[Path, ...] = (
    Path("/content/datasets/mvtec"),
    Path("/content/drive/MyDrive/datasets/mvtec"),
    Path("/content"),
)


def looks_like_dataset_root(path: Path) -> bool:
    """True when *path* directly contains a plausible MVTec AD category tree.

    Requires at least three known category names plus a complete ``bottle``
    category (``train/good`` and ``test``), which every official copy has.
    """
    try:
        entries = {p.name for p in path.iterdir() if p.is_dir()}
    except OSError:
        return False
    matched = set(entries) & set(CATEGORIES)
    if len(matched) < 3:
        return False
    bottle = path / "bottle"
    return (bottle / "train" / GOOD).is_dir() and (bottle / "test").is_dir()


def _iter_candidates(bases: Iterable[Path], max_depth: int = 3) -> Iterator[Path]:
    """Breadth-first walk (bounded depth) over *bases* and their subdirectories."""
    seen: set = set()
    queue: List[Tuple[Path, int]] = [(Path(b), 0) for b in bases]
    while queue:
        current, depth = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        yield current
        if depth >= max_depth:
            continue
        try:
            children = sorted(p for p in current.iterdir() if p.is_dir())
        except OSError:
            continue
        for child in children:
            queue.append((child, depth + 1))


def find_dataset_root(start: Optional[Path] = None) -> Path:
    """Locate the directory that directly contains the MVTec AD categories.

    Search order:

    1. ``MVTEC_ROOT`` environment variable (explicit override, same mechanism as
       :attr:`src.config.Paths.data`).
    2. *start* (e.g. the extraction folder) and a bounded walk below and above it.
    3. The current working directory and its parents.
    4. Known Colab locations (``/content/datasets/mvtec``, a Drive copy, ``/content``).

    Raises :class:`FileNotFoundError` with the searched locations when nothing
    matches - never a guess.
    """
    bases: List[Path] = []
    env = os.environ.get("MVTEC_ROOT")
    if env:
        bases.append(Path(env))
    if start is not None:
        bases.append(Path(start))
    bases.append(Path.cwd())
    bases.extend(_KNOWN_STARTS)

    searched: List[Path] = []
    for candidate in _iter_candidates(bases):
        searched.append(candidate)
        if looks_like_dataset_root(candidate):
            return candidate

    preview = "\n".join(f"  - {p}" for p in searched[:15])
    more = f"\n  ... and {len(searched) - 15} more" if len(searched) > 15 else ""
    raise FileNotFoundError(
        "Could not find the MVTec AD dataset root (no folder with bottle/train/good, ...).\n"
        f"Searched:\n{preview}{more}\n"
        "Fix one of:\n"
        "  1. run the download cell (notebook 02, cell 1),\n"
        "  2. run the extraction cell,\n"
        "  3. or set MVTEC_ROOT to the folder that contains 'bottle', 'cable', ..."
    )


def summarize_layout(root: Path) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Count files per split and defect type for every category under *root*.

    Returns ``{category: {split: {defect: count}}}`` based on what is actually
    on disk - the structure is inspected, not assumed.
    """
    root = Path(root)
    layout: Dict[str, Dict[str, Dict[str, int]]] = {}
    if not root.is_dir():
        return layout
    for cat_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        splits: Dict[str, Dict[str, int]] = {}
        for split in SPLITS:
            split_dir = cat_dir / split
            if not split_dir.is_dir():
                continue
            counts: Dict[str, int] = {}
            for entry in sorted(split_dir.iterdir()):
                if entry.is_dir():
                    counts[entry.name] = sum(1 for f in entry.iterdir() if f.is_file())
                elif entry.is_file():
                    counts["(loose file)"] = counts.get("(loose file)", 0) + 1
            if counts:
                splits[split] = counts
        if splits:
            layout[cat_dir.name] = splits
    return layout


@dataclass
class CategoryReport:
    """Everything found (and everything wrong) for one category."""

    category: str
    root: Path
    train_good: List[Path] = field(default_factory=list)
    test_good: List[Path] = field(default_factory=list)
    test_by_defect: Dict[str, List[Path]] = field(default_factory=dict)
    masks_by_defect: Dict[str, List[Path]] = field(default_factory=dict)
    missing_masks: List[Path] = field(default_factory=list)
    orphan_masks: List[Path] = field(default_factory=list)
    unreadable: List[Tuple[Path, str]] = field(default_factory=list)
    empty_masks: List[Path] = field(default_factory=list)
    size_mismatch: List[Tuple[Path, str]] = field(default_factory=list)
    unexpected: List[str] = field(default_factory=list)
    formats: Counter = field(default_factory=Counter)
    modes: Counter = field(default_factory=Counter)
    sizes: Counter = field(default_factory=Counter)
    checked_images: int = 0
    checked_masks: int = 0

    @property
    def defect_names(self) -> List[str]:
        return sorted(self.test_by_defect)

    @property
    def n_test_defective(self) -> int:
        return sum(len(v) for v in self.test_by_defect.values())

    @property
    def errors(self) -> List[str]:
        """Hard problems: must be fixed before anything is trained or scored."""
        lines: List[str] = []
        for path in self.missing_masks:
            lines.append(f"missing ground-truth mask for {path}")
        for path, why in self.unreadable:
            lines.append(f"unreadable file {path} ({why})")
        for path, why in self.size_mismatch:
            lines.append(f"size mismatch: {path} ({why})")
        for path in self.orphan_masks:
            lines.append(f"mask without a matching test image: {path}")
        for line in self.unexpected:
            lines.append(f"unexpected entry in the official layout: {line}")
        return lines

    @property
    def warnings(self) -> List[str]:
        """Suspicious but not fatal (reported, never silently dropped)."""
        return [f"mask has no positive pixel: {p}" for p in self.empty_masks]

    @property
    def ok(self) -> bool:
        return not self.errors

    def counts(self) -> Dict[str, object]:
        return {
            "train_good": len(self.train_good),
            "test_good": len(self.test_good),
            "test_defective": self.n_test_defective,
            "masks": sum(len(v) for v in self.masks_by_defect.values()),
            "errors": len(self.errors),
            "warnings": len(self.warnings),
        }

    def summary(self) -> str:
        lines = [
            f"category          : {self.category}",
            f"train (good)      : {len(self.train_good)}",
            f"test  (good)      : {len(self.test_good)}",
            f"test  (defective) : {self.n_test_defective}",
            f"ground-truth masks: {sum(len(v) for v in self.masks_by_defect.values())}",
            f"images decoded    : {self.checked_images}",
            f"masks decoded     : {self.checked_masks}",
        ]
        if self.sizes:
            pretty = ", ".join(f"{w}x{h} ({n})" for (w, h), n in self.sizes.most_common())
            lines.append(f"image sizes       : {pretty}")
        if self.formats:
            lines.append(f"formats           : {dict(self.formats)}")
        lines.append(f"errors            : {len(self.errors)}")
        lines.append(f"warnings          : {len(self.warnings)}")
        return "\n".join(lines)


def _list_files(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file())


def _verify_image(path: Path, report: CategoryReport) -> Optional[Tuple[int, int]]:
    try:
        with Image.open(path) as img:
            img.load()
            size = img.size
            report.formats[str(img.format)] += 1
            report.modes[str(img.mode)] += 1
            report.sizes[size] += 1
            report.checked_images += 1
            return size
    except Exception as exc:  # noqa: BLE001 - every failure mode must be reported
        report.unreadable.append((path, f"{type(exc).__name__}: {exc}"))
        return None


def _verify_mask(path: Path, expected_size: Optional[Tuple[int, int]], report: CategoryReport) -> None:
    try:
        with Image.open(path) as img:
            img.load()
            if expected_size is not None and img.size != expected_size:
                report.size_mismatch.append(
                    (path, f"mask {img.size[0]}x{img.size[1]} vs image {expected_size[0]}x{expected_size[1]}")
                )
            report.checked_masks += 1
            extrema = img.convert("L").getextrema()
            if isinstance(extrema, tuple) and extrema[1] == 0:
                report.empty_masks.append(path)
    except Exception as exc:  # noqa: BLE001
        report.unreadable.append((path, f"mask {type(exc).__name__}: {exc}"))


def scan_category(root: Path, category: str, *, verify_images: bool = True) -> CategoryReport:
    """Enumerate and verify one category's files.

    * ``train/good`` -> :attr:`CategoryReport.train_good` (the ONLY training material)
    * ``test/good`` -> :attr:`CategoryReport.test_good`
    * every other folder under ``test/`` -> :attr:`CategoryReport.test_by_defect`
    * ``ground_truth/<defect>/<name>_mask.png`` must match every defective image

    With ``verify_images=False`` only the file lists are built (fast); with the
    default every image and mask is decoded, and unreadable files, size
    mismatches and empty masks are recorded instead of being skipped.
    """
    cat_dir = Path(root) / category
    if not cat_dir.is_dir():
        available = sorted(p.name for p in Path(root).iterdir() if p.is_dir()) if Path(root).is_dir() else []
        raise FileNotFoundError(
            f"Category '{category}' not found under {root}. Available: {available}"
        )

    train_dir = cat_dir / "train"
    test_dir = cat_dir / "test"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Missing {train_dir} - the official split requires train/ per category.")
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Missing {test_dir} - the official split requires test/ per category.")

    report = CategoryReport(category=category, root=cat_dir)

    for entry in sorted(train_dir.iterdir()):
        if entry.is_dir() and entry.name == GOOD:
            report.train_good = _list_files(entry)
        elif entry.is_dir():
            report.unexpected.append(f"train/{entry.name}/ (training must be defect-free: only train/good)")
        elif entry.is_file():
            report.unexpected.append(f"train/{entry.name} (loose file)")

    for entry in sorted(test_dir.iterdir()):
        if entry.is_file():
            report.unexpected.append(f"test/{entry.name} (loose file)")
            continue
        files = _list_files(entry)
        if entry.name == GOOD:
            report.test_good = files
        else:
            report.test_by_defect[entry.name] = files

    gt_dir = cat_dir / "ground_truth"
    for defect, images in report.test_by_defect.items():
        defect_masks = _list_files(gt_dir / defect)
        report.masks_by_defect[defect] = defect_masks
        mask_stems = {m.stem for m in defect_masks}
        image_stems = {img.stem for img in images}
        for img in images:
            if f"{img.stem}_mask" not in mask_stems:
                report.missing_masks.append(img)
        for mask in defect_masks:
            base = mask.stem[: -len("_mask")] if mask.stem.endswith("_mask") else mask.stem
            if base not in image_stems:
                report.orphan_masks.append(mask)

    if gt_dir.is_dir():
        for entry in sorted(gt_dir.iterdir()):
            if entry.is_dir() and entry.name not in report.test_by_defect:
                report.unexpected.append(f"ground_truth/{entry.name}/ (no matching test defect folder)")
            elif entry.is_file():
                report.unexpected.append(f"ground_truth/{entry.name} (loose file)")

    if verify_images:
        image_sizes: Dict[Path, Tuple[int, int]] = {}
        for path in [*report.train_good, *report.test_good]:
            size = _verify_image(path, report)
            if size is not None:
                image_sizes[path] = size
        for defect, images in report.test_by_defect.items():
            for path in images:
                size = _verify_image(path, report)
                if size is not None:
                    image_sizes[path] = size
            for mask in report.masks_by_defect.get(defect, []):
                partner = next(
                    (p for p in images if p.stem == mask.stem[: -len("_mask")] if mask.stem.endswith("_mask")),
                    None,
                )
                expected = image_sizes.get(partner) if partner else None
                _verify_mask(mask, expected, report)

    return report


# ---------------------------------------------------------------------------
# Stage 3 - PyTorch datasets for training on NORMAL images only
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for the training data pipeline but is not "
            "importable. On Colab: reconnect to a GPU runtime. Locally: "
            "pip install torch torchvision (see requirements.txt)."
        )


def list_image_files(directory: Union[Path, str]) -> List[Path]:
    """Return the sorted image files directly inside *directory*.

    Raises :class:`FileNotFoundError` when the folder is missing or holds no
    recognisable image files - an empty training set must never pass silently.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    files = sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(
            f"No image files {IMAGE_EXTENSIONS} found in {directory}. "
            "Check the dataset extraction (notebook 02) before training."
        )
    return files


def split_train_val(
    image_paths: Sequence[Union[Path, str]],
    val_fraction: float,
    seed: int,
) -> Tuple[List[Path], List[Path]]:
    """Split *image_paths* into (train, validation) with a reproducible seed.

    The split is done on sorted file paths with ``random.Random(seed)``, so it
    depends only on the path list and the seed - not on DataLoader worker
    count, filesystem order or wall-clock time. Only pass NORMAL images here;
    the official test set must never be split off for validation.
    """
    if not 0.0 <= val_fraction < 0.5:
        raise ValueError(f"val_fraction must be in [0.0, 0.5), got {val_fraction!r}")
    paths = sorted(Path(p) for p in image_paths)
    n = len(paths)
    if n == 0:
        raise ValueError("split_train_val received an empty path list")
    if val_fraction == 0.0 or n == 1:
        return paths, []
    n_val = max(1, min(int(round(n * val_fraction)), n - 1))
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    val_indices = sorted(indices[:n_val])
    train_indices = sorted(indices[n_val:])
    return [paths[i] for i in train_indices], [paths[i] for i in val_indices]


def build_transform(image_size: int):
    """PIL image -> float32 CHW tensor of shape ``(3, image_size, image_size)``.

    Normalisation strategy (documented on purpose, the decoder must match it):

    * ``Resize`` to ``(image_size, image_size)`` - MVTec AD images are square
      (e.g. 900x900 for bottle), so no aspect ratio is distorted.
    * ``ToTensor`` scales uint8 pixels ``[0, 255]`` to float32 ``[0, 1]`` and
      moves the channel axis from last (HWC) to first (CHW).

    No mean/std standardisation is applied: reconstruction targets and model
    outputs then share the exact ``[0, 1]`` range, which pairs with a sigmoid
    output activation and keeps pixel-wise error maps directly interpretable.
    """
    _require_torch()
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
        ]
    )


class NormalImageDataset(_TorchDataset):
    """Map-style dataset of defect-free training images.

    Each item is a dict ``{"image": FloatTensor[3, H, W] in [0, 1], "path": str}``
    so notebooks can show reconstructions next to their source files. A file
    that fails to decode raises :class:`RuntimeError` naming the path - files
    are never skipped silently (notebook 02 already verified every image).
    """

    def __init__(self, image_paths: Sequence[Union[Path, str]], transform) -> None:
        _require_torch()
        self.image_paths: List[Path] = [Path(p) for p in image_paths]
        if not self.image_paths:
            raise ValueError("NormalImageDataset requires at least one image path")
        if transform is None:
            raise ValueError("NormalImageDataset requires a transform (see build_transform)")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, object]:
        path = self.image_paths[index]
        try:
            with Image.open(path) as img:
                image = img.convert("RGB")  # grayscale/RGBA files -> consistent 3 channels
                tensor = self.transform(image)
        except Exception as exc:  # noqa: BLE001 - the path must appear in the error
            raise RuntimeError(f"Failed to load image {path}: {exc}") from exc
        if not torch.is_tensor(tensor) or tensor.ndim != 3 or tensor.shape[0] != 3:
            raise RuntimeError(
                f"Transform of {path} produced {type(tensor).__name__} "
                f"with shape {getattr(tensor, 'shape', None)}; expected a CHW tensor with 3 channels."
            )
        return {"image": tensor, "path": str(path)}


def create_dataloaders(
    dataset_root: Union[Path, str],
    category: str = "bottle",
    image_size: int = 256,
    batch_size: int = 16,
    val_fraction: float = 0.1,
    seed: int = 42,
    num_workers: int = 2,
    pin_memory: bool = False,
) -> Tuple["DataLoader", Optional["DataLoader"], Dict[str, object]]:
    """Build train/validation dataloaders from ``<root>/<category>/train/good``.

    Only the normal training folder is read - ``test/`` and ``ground_truth/``
    are never opened here, so anomalous images cannot leak into training or
    into model selection. The validation split (when ``val_fraction > 0``)
    consists solely of held-out normal images drawn with *seed*.

    Returns ``(train_loader, val_loader_or_None, split_info)`` where
    ``split_info`` records the split for reproducibility/auditing.

    Notes:
        * ``drop_last=True`` on the training loader keeps every batch the same
          size (BatchNorm statistics stay comparable); the remainder images are
          still used across epochs because the sampler reshuffles. The
          validation loader keeps its (possibly shorter) final batch.
        * On Windows local runs, set ``num_workers=0`` (spawned workers and
          notebooks mix poorly); Colab (Linux) is fine with 2.
    """
    _require_torch()
    root = Path(dataset_root)
    train_good_dir = root / category / "train" / GOOD
    all_paths = list_image_files(train_good_dir)
    train_paths, val_paths = split_train_val(all_paths, val_fraction, seed)

    transform = build_transform(image_size)
    train_dataset = NormalImageDataset(train_paths, transform)
    val_dataset = NormalImageDataset(val_paths, transform) if val_paths else None

    if len(train_dataset) < batch_size:
        raise ValueError(
            f"Training set has {len(train_dataset)} images but batch_size={batch_size} "
            "with drop_last=True would yield zero batches. Reduce data.batch_size "
            f"(must be <= {len(train_dataset)})."
        )

    generator = torch.Generator()
    generator.manual_seed(seed)
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,  # epoch-wise order changes; seeded generator makes it reproducible
        drop_last=True,
        generator=generator,
        **common,
    )
    val_loader = (
        DataLoader(val_dataset, shuffle=False, drop_last=False, **common)
        if val_dataset is not None
        else None
    )

    split_info: Dict[str, object] = {
        "dataset_root": str(root),
        "category": category,
        "image_size": image_size,
        "batch_size": batch_size,
        "val_fraction": val_fraction,
        "seed": seed,
        "n_all_train_good": len(all_paths),
        "n_train": len(train_paths),
        "n_val": len(val_paths),
        "train_paths": [str(p) for p in train_paths],
        "val_paths": [str(p) for p in val_paths],
    }
    return train_loader, val_loader, split_info
