"""Dataset discovery and validation helpers for MVTec AD (Stage 2 inspection).

The inspection notebook (``notebooks/02_dataset_inspection.ipynb``) calls these
helpers instead of hard-coding paths:

* :func:`find_dataset_root` - locate the folder that directly contains the
  category directories (``bottle/``, ``cable/``, ...). The extraction layout is
  probed, never assumed.
* :func:`scan_category` - enumerate normal/defective images and ground-truth
  masks for one category and verify every file (readable, sizes consistent,
  mask present).
* :func:`summarize_layout` - per-split, per-defect file counts for every
  category.

Rules honoured by this module:

* PyTorch is never imported at module load time (same rule as ``src/config.py``),
  so a fresh runtime can import it before dependencies are verified.
* Problems are returned in the report - nothing is ever skipped silently.
* The official split is never modified: training material comes from
  ``train/good`` only; ``test/`` and ``ground_truth/`` are for inspection and
  later evaluation only.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from PIL import Image

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
