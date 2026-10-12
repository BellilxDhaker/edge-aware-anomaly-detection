"""Project-wide configuration: paths, seeds, and small runtime helpers.

This module must stay importable with the standard library alone (no numpy or
torch import at module load time) so notebooks can load it before dependencies
have been verified on a fresh Colab runtime.
"""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

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
