"""Project-wide configuration: paths, seeds, and small runtime helpers.

This module must stay importable with the standard library alone (no numpy or
torch import at module load time) so notebooks can load it before dependencies
have been verified on a fresh Colab runtime.
"""

from __future__ import annotations

import copy
import hashlib
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

SEED = 42
"""Global seed used by every entry point in this project."""

REPO_DIR_NAME = "edge-aware-anomaly-detection"

# Locations that may hold a repo copy when a notebook runs on a Colab runtime.
# A Colab kernel runs on Google's servers and cannot see the local disk; the
# repo gets there only via "Upload to Colab" (VS Code extension) or a Drive mount.
_RUNTIME_ROOT_CANDIDATES = (
    Path("/content") / REPO_DIR_NAME,
    Path("/content/drive/MyDrive") / REPO_DIR_NAME,
    Path("/content"),
)


def _looks_like_repo_root(path: Path) -> bool:
    return (path / "src" / "config.py").is_file() and (path / "src" / "__init__.py").is_file()


def find_repo_root(start: Optional[Union[Path, str]] = None) -> Path:
    """Locate the repository root (the directory that contains ``src/``).

    Search order:

    1. The package location of this file (``src/config.py`` -> parent of ``src``).
       This works both locally and on a runtime after the repo was uploaded.
    2. ``start`` (defaults to the current working directory) and all of its parents.
    3. Known Colab runtime locations (uploaded repo folder, mounted Drive copy, ``/content``).
    """
    here = Path(__file__).resolve()
    if _looks_like_repo_root(here.parents[1]):
        return here.parents[1]

    search_from = Path(start) if start is not None else Path.cwd()
    for candidate in (search_from, *search_from.parents):
        if _looks_like_repo_root(candidate):
            return candidate

    for root in _RUNTIME_ROOT_CANDIDATES:
        if _looks_like_repo_root(root):
            return root

    raise FileNotFoundError(
        "Could not locate the repository root (no src/config.py found). "
        "On a Colab runtime: in the VS Code Explorer, right-click the repository "
        "folder and choose 'Upload to Colab', then re-run the notebook cell."
    )


@dataclass(frozen=True)
class Paths:
    """Resolved project directories.

    ``data`` and ``outputs`` can grow large and are covered by ``.gitignore``;
    they must never be committed. ``data`` can be redirected with the
    ``MVTEC_ROOT`` environment variable (useful for keeping the dataset on a
    mounted Google Drive instead of the ephemeral ``/content`` filesystem).
    """

    repo_root: Path

    @property
    def src(self) -> Path:
        return self.repo_root / "src"

    @property
    def notebooks(self) -> Path:
        return self.repo_root / "notebooks"

    @property
    def configs(self) -> Path:
        return self.repo_root / "configs"

    @property
    def tests(self) -> Path:
        return self.repo_root / "tests"

    @property
    def outputs(self) -> Path:
        return self.repo_root / "outputs"

    @property
    def data(self) -> Path:
        override = os.environ.get("MVTEC_ROOT")
        return Path(override) if override else self.repo_root / "data"


def get_paths(start: Optional[Union[Path, str]] = None) -> Paths:
    """Return a :class:`Paths` instance rooted at the discovered repo root."""
    return Paths(repo_root=find_repo_root(start))


def set_seed(seed: Optional[int] = None) -> int:
    """Seed python, numpy (if installed) and torch (if installed).

    Also forces cuDNN toward deterministic kernels (``deterministic=True``,
    ``benchmark=False``): slower, but reproducible enough for research runs.
    Returns the seed that was applied.
    """
    seed = SEED if seed is None else int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed


def get_device(prefer_cuda: bool = True):
    """Return ``torch.device('cuda')`` when usable, else ``torch.device('cpu')``."""
    import torch  # deferred: verified by the environment notebook, not at import time

    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def file_sha256(path: Union[Path, str], length: int = 12) -> str:
    """Short content hash, used to detect stale uploads of ``src`` files on a runtime."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()[:length]


# ---------------------------------------------------------------------------
# Experiment configuration (Stage 3: baseline autoencoder)
# ---------------------------------------------------------------------------
#
# A frozen dataclass tree (ExperimentConfig -> DataConfig / ModelConfig /
# TrainConfig) is the typed view the code uses; YAML files under ``configs/``
# and inline dict overrides are merged into it by :func:`load_experiment_config`.
# pyyaml is imported lazily inside the loader/saver so this module keeps
# working - and stays importable - with the standard library alone.

DEFAULT_EXPERIMENT: Dict[str, Any] = {
    "experiment_name": "baseline_bottle",
    "category": "bottle",
    "seed": SEED,
    "data": {
        "image_size": 256,
        "val_fraction": 0.1,
        "batch_size": 16,
        "num_workers": 2,
    },
    "model": {
        "in_channels": 3,
        "base_channels": 32,
        "channel_mults": [1, 2, 4, 8, 8],
        "latent_channels": 256,
        "num_convs_per_stage": 2,
    },
    "train": {
        "epochs": 30,
        "optimizer": "adam",
        "lr": 0.001,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
    },
}

_OPTIMIZER_NAMES = ("adam", "adamw")


@dataclass(frozen=True)
class DataConfig:
    """Data-loading hyper-parameters for the baseline experiment.

    ``image_size`` must be divisible by ``2 ** len(ModelConfig.channel_mults)``
    because the encoder pools once per stage; this is validated when the full
    experiment config is built.
    """

    image_size: int = 256
    val_fraction: float = 0.1
    batch_size: int = 16
    num_workers: int = 2


@dataclass(frozen=True)
class ModelConfig:
    """Architecture hyper-parameters of the baseline convolutional autoencoder.

    The spatial resolution is divided by ``2 ** num_downsamples`` inside the
    encoder; ``channel_mults[i]`` sets the output channels of stage *i* as
    ``base_channels * channel_mults[i]``.
    """

    in_channels: int = 3
    base_channels: int = 32
    channel_mults: Tuple[int, ...] = (1, 2, 4, 8, 8)
    latent_channels: int = 256
    num_convs_per_stage: int = 2

    @property
    def num_downsamples(self) -> int:
        return len(self.channel_mults)

    @property
    def downsample_factor(self) -> int:
        return 2 ** self.num_downsamples


@dataclass(frozen=True)
class TrainConfig:
    """Optimization hyper-parameters.

    ``grad_clip`` is the max gradient norm (``0.0`` disables clipping).
    """

    epochs: int = 30
    optimizer: str = "adam"
    lr: float = 0.001
    weight_decay: float = 0.0
    grad_clip: float = 1.0


@dataclass(frozen=True)
class ExperimentConfig:
    """Fully validated configuration of one experiment run."""

    experiment_name: str
    category: str
    seed: int
    data: DataConfig
    model: ModelConfig
    train: TrainConfig

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict view (YAML/JSON friendly) used for saving and logging."""
        return {
            "experiment_name": self.experiment_name,
            "category": self.category,
            "seed": self.seed,
            "data": asdict(self.data),
            "model": {
                **asdict(self.model),
                "channel_mults": list(self.model.channel_mults),
            },
            "train": asdict(self.train),
        }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _merge_dict(base: Mapping[str, Any], extra: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge *extra* onto *base* (extra wins); returns a new dict."""
    merged: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _check_keys(section: str, raw: Mapping[str, Any], allowed: Tuple[str, ...]) -> None:
    unknown = sorted(set(raw) - set(allowed))
    if unknown:
        raise ValueError(
            f"Unknown key(s) in '{section}': {unknown}. Allowed: {sorted(allowed)}"
        )


def _build_experiment(raw: Mapping[str, Any]) -> ExperimentConfig:
    """Validate a merged config dict and build the typed configuration."""
    _check_keys(
        "top level",
        raw,
        ("experiment_name", "category", "seed", "data", "model", "train"),
    )
    data_raw = raw.get("data") or {}
    model_raw = raw.get("model") or {}
    train_raw = raw.get("train") or {}
    _check_keys("data", data_raw, ("image_size", "val_fraction", "batch_size", "num_workers"))
    _check_keys(
        "model",
        model_raw,
        ("in_channels", "base_channels", "channel_mults", "latent_channels", "num_convs_per_stage"),
    )
    _check_keys("train", train_raw, ("epochs", "optimizer", "lr", "weight_decay", "grad_clip"))

    name = raw.get("experiment_name", DEFAULT_EXPERIMENT["experiment_name"])
    category = raw.get("category", DEFAULT_EXPERIMENT["category"])
    seed = raw.get("seed", SEED)
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"'experiment_name' must be a non-empty string, got {name!r}")
    if not isinstance(category, str) or not category.strip():
        raise ValueError(f"'category' must be a non-empty string, got {category!r}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"'seed' must be an integer, got {seed!r}")

    data_defaults = DEFAULT_EXPERIMENT["data"]
    image_size = data_raw.get("image_size", data_defaults["image_size"])
    val_fraction = data_raw.get("val_fraction", data_defaults["val_fraction"])
    batch_size = data_raw.get("batch_size", data_defaults["batch_size"])
    num_workers = data_raw.get("num_workers", data_defaults["num_workers"])
    if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size < 32:
        raise ValueError(f"'data.image_size' must be an int >= 32, got {image_size!r}")
    if not _is_number(val_fraction) or not 0.0 <= float(val_fraction) < 0.5:
        raise ValueError(
            f"'data.val_fraction' must be a number in [0.0, 0.5), got {val_fraction!r}"
        )
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError(f"'data.batch_size' must be an int >= 1, got {batch_size!r}")
    if not isinstance(num_workers, int) or isinstance(num_workers, bool) or num_workers < 0:
        raise ValueError(f"'data.num_workers' must be an int >= 0, got {num_workers!r}")

    model_defaults = DEFAULT_EXPERIMENT["model"]
    in_channels = model_raw.get("in_channels", model_defaults["in_channels"])
    base_channels = model_raw.get("base_channels", model_defaults["base_channels"])
    mults_raw = model_raw.get("channel_mults", model_defaults["channel_mults"])
    latent_channels = model_raw.get("latent_channels", model_defaults["latent_channels"])
    num_convs = model_raw.get(
        "num_convs_per_stage", model_defaults["num_convs_per_stage"]
    )
    if not isinstance(mults_raw, (list, tuple)) or not mults_raw:
        raise ValueError(
            f"'model.channel_mults' must be a non-empty list of ints, got {mults_raw!r}"
        )
    if not all(isinstance(m, int) and not isinstance(m, bool) and m >= 1 for m in mults_raw):
        raise ValueError(f"'model.channel_mults' entries must be ints >= 1, got {mults_raw!r}")
    channel_mults = tuple(int(m) for m in mults_raw)
    if not isinstance(in_channels, int) or isinstance(in_channels, bool) or in_channels < 1:
        raise ValueError(f"'model.in_channels' must be an int >= 1, got {in_channels!r}")
    if not isinstance(base_channels, int) or isinstance(base_channels, bool) or base_channels < 1:
        raise ValueError(f"'model.base_channels' must be an int >= 1, got {base_channels!r}")
    if not isinstance(latent_channels, int) or isinstance(latent_channels, bool) or latent_channels < 1:
        raise ValueError(f"'model.latent_channels' must be an int >= 1, got {latent_channels!r}")
    if not isinstance(num_convs, int) or isinstance(num_convs, bool) or num_convs < 1:
        raise ValueError(
            f"'model.num_convs_per_stage' must be an int >= 1, got {num_convs!r}"
        )

    downsample_factor = 2 ** len(channel_mults)
    if image_size % downsample_factor != 0:
        raise ValueError(
            f"'data.image_size' ({image_size}) must be divisible by "
            f"2 ** len(model.channel_mults) = {downsample_factor}; "
            f"pick e.g. {downsample_factor * max(1, image_size // downsample_factor)}."
        )

    train_defaults = DEFAULT_EXPERIMENT["train"]
    epochs = train_raw.get("epochs", train_defaults["epochs"])
    optimizer = train_raw.get("optimizer", train_defaults["optimizer"])
    lr = train_raw.get("lr", train_defaults["lr"])
    weight_decay = train_raw.get("weight_decay", train_defaults["weight_decay"])
    grad_clip = train_raw.get("grad_clip", train_defaults["grad_clip"])
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError(f"'train.epochs' must be an int >= 1, got {epochs!r}")
    if not isinstance(optimizer, str) or optimizer.lower() not in _OPTIMIZER_NAMES:
        raise ValueError(
            f"'train.optimizer' must be one of {_OPTIMIZER_NAMES}, got {optimizer!r}"
        )
    if not _is_number(lr) or float(lr) <= 0.0:
        raise ValueError(f"'train.lr' must be a positive number, got {lr!r}")
    if not _is_number(weight_decay) or float(weight_decay) < 0.0:
        raise ValueError(f"'train.weight_decay' must be >= 0, got {weight_decay!r}")
    if not _is_number(grad_clip) or float(grad_clip) < 0.0:
        raise ValueError(
            f"'train.grad_clip' must be >= 0 (0 disables clipping), got {grad_clip!r}"
        )

    return ExperimentConfig(
        experiment_name=name.strip(),
        category=category.strip(),
        seed=seed,
        data=DataConfig(
            image_size=image_size,
            val_fraction=float(val_fraction),
            batch_size=batch_size,
            num_workers=num_workers,
        ),
        model=ModelConfig(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            latent_channels=latent_channels,
            num_convs_per_stage=num_convs,
        ),
        train=TrainConfig(
            epochs=epochs,
            optimizer=optimizer.lower(),
            lr=float(lr),
            weight_decay=float(weight_decay),
            grad_clip=float(grad_clip),
        ),
    )


def load_experiment_config(
    path: Optional[Union[Path, str]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> ExperimentConfig:
    """Build an :class:`ExperimentConfig` from defaults + YAML + overrides.

    Precedence (later wins): ``DEFAULT_EXPERIMENT`` -> YAML file at *path*
    (if given) -> *overrides* mapping (nested dicts merge section by section).

    Unknown keys and invalid values raise :class:`ValueError` with the offending
    key path - nothing is silently coerced.
    """
    raw = copy.deepcopy(DEFAULT_EXPERIMENT)
    if path is not None:
        import yaml  # lazy: keeps this module stdlib-only at import time

        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Experiment config not found: {config_path}\n"
                "Expected a YAML file such as configs/default.yaml."
            )
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, Mapping):
            raise ValueError(
                f"Config file {config_path} must contain a YAML mapping, "
                f"got {type(loaded).__name__}"
            )
        raw = _merge_dict(raw, loaded)
    if overrides:
        raw = _merge_dict(raw, overrides)
    return _build_experiment(raw)


def save_experiment_config(config: ExperimentConfig, path: Union[Path, str]) -> Path:
    """Write *config* to *path* as YAML (creates parent directories)."""
    import yaml  # lazy: same rationale as in the loader

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
    return out
