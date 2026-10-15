"""Training loop for the baseline autoencoder (Phase 2).

Responsibilities kept here so notebooks stay thin orchestration layers:

* configurable epochs / optimizer (Adam or AdamW) / learning rate,
* GPU acceleration when available (batches moved with ``non_blocking=True``),
* per-epoch train (and optional validation) loss, weighted by batch size,
* explicit non-finite loss detection that stops the run with context,
* wall-clock timing per epoch and in total,
* checkpointing (``last.pt`` every epoch, ``best.pt`` on improvement) with
  model/optimizer state, epoch counter, history and RNG states so training
  can be resumed reproducibly,
* ``history.json`` and ``metadata.json`` written into the run directory.

Only normal images reach this trainer: the dataloaders are built from
``train/good`` by :func:`src.data.dataset.create_dataloaders`, and the
validation loader (if any) also holds only normal held-out images.
"""

from __future__ import annotations

import json
import platform
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from torch import nn


@dataclass
class EpochRecord:
    """One finished epoch: losses (arithmetic means), duration and learning rate."""

    epoch: int  # 1-based
    train_loss: float
    val_loss: Optional[float] = None
    seconds: float = 0.0
    lr: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "EpochRecord":
        return cls(
            epoch=int(raw["epoch"]),
            train_loss=float(raw["train_loss"]),
            val_loss=None if raw.get("val_loss") is None else float(raw["val_loss"]),
            seconds=float(raw.get("seconds", 0.0)),
            lr=float(raw.get("lr", 0.0)),
        )


@dataclass
class TrainingResult:
    """Summary returned by :meth:`Trainer.fit`."""

    records: List[EpochRecord]
    total_seconds: float
    best_metric: Optional[float]
    run_dir: Path
    checkpoint_paths: Dict[str, Path] = field(default_factory=dict)

    @property
    def history(self) -> List[Dict[str, Any]]:
        return [record.to_dict() for record in self.records]


def _images_from_batch(batch: Any) -> torch.Tensor:
    """Accept the project batch dict ``{"image": ..., "path": ...}`` or a raw tensor."""
    if isinstance(batch, dict):
        if "image" not in batch:
            raise KeyError(f"Batch dict has no 'image' key; keys={sorted(batch)}")
        return batch["image"]
    if isinstance(batch, (tuple, list)):
        return batch[0]
    if torch.is_tensor(batch):
        return batch
    raise TypeError(f"Unsupported batch type {type(batch).__name__}")


def _rng_states() -> Dict[str, Any]:
    """Snapshot python/torch/CUDA RNG so a resumed run continues the stream."""
    states: Dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def _restore_rng_states(states: Dict[str, Any]) -> None:
    if "python" in states:
        random.setstate(states["python"])
    if "torch" in states:
        torch.set_rng_state(torch.as_tensor(states["torch"], dtype=torch.uint8))
    if "cuda" in states and torch.cuda.is_available():
        saved = states["cuda"]
        if len(saved) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([torch.as_tensor(s, dtype=torch.uint8) for s in saved])


def write_run_metadata(run_dir: Union[Path, str], **fields: Any) -> Path:
    """Write ``metadata.json`` describing the runtime plus any extra fields."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    payload: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
    }
    payload.update(fields)
    out = run_dir / "metadata.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out


def write_history(records: Sequence[EpochRecord], run_dir: Union[Path, str]) -> Path:
    """Write the full training history to ``history.json`` inside *run_dir*."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "history.json"
    out.write_text(
        json.dumps([record.to_dict() for record in records], indent=2), encoding="utf-8"
    )
    return out


class Trainer:
    """Minimal, explicit training loop for a reconstruction model.

    Parameters
    ----------
    model:
        Any module mapping images to reconstructions of the same shape
        (the baseline :class:`src.models.Autoencoder`).
    criterion:
        Callable ``(reconstruction, target) -> scalar tensor``.
    optimizer:
        A ``torch.optim.Optimizer`` already configured with the model
        parameters (see :func:`build_optimizer`).
    device:
        ``torch.device`` (or string) the model and batches are moved to.
    run_dir:
        Directory for checkpoints/history; created if missing.
    grad_clip:
        Max gradient norm; ``0.0`` disables clipping.
    config:
        Optional plain-dict view of the experiment config; stored inside
        checkpoints so a resumed run can verify it loaded the right weights.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        run_dir: Union[Path, str],
        grad_clip: float = 0.0,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if grad_clip < 0.0:
            raise ValueError(f"grad_clip must be >= 0, got {grad_clip}")
        self.grad_clip = float(grad_clip)
        self.config = config
        self.best_metric: Optional[float] = None
        self._records: List[EpochRecord] = []
        self._current_lr = self._get_lr()

    # ------------------------------------------------------------------ utils
    def _get_lr(self) -> float:
        for group in self.optimizer.param_groups:
            return float(group["lr"])
        return 0.0

    def _assert_finite(self, loss: torch.Tensor, context: str) -> None:
        if not bool(torch.isfinite(loss).all()):
            raise FloatingPointError(
                f"Non-finite loss ({context}). The run is stopped instead of "
                "silently writing NaN checkpoints. Likely causes: learning rate "
                "too high, a corrupted batch, or numerical overflow in the model. "
                f"Last lr={self._get_lr()}."
            )

    # ----------------------------------------------------------------- epochs
    def train_one_epoch(self, loader, epoch: int) -> float:
        """One pass over *loader* with gradients; returns the mean batch loss."""
        self.model.train()
        total_loss, total_images = 0.0, 0
        for batch_index, batch in enumerate(loader):
            images = _images_from_batch(batch).to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            reconstruction = self.model(images)
            loss = self.criterion(reconstruction, images)
            self._assert_finite(loss, f"train epoch {epoch} batch {batch_index}")
            loss.backward()
            if self.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            batch_size = images.shape[0]
            total_loss += float(loss.detach()) * batch_size
            total_images += batch_size
        if total_images == 0:
            raise RuntimeError("Training loader yielded no batches - check batch_size/dataset size")
        return total_loss / total_images

    @torch.no_grad()
    def validate(self, loader, epoch: int) -> float:
        """One pass over *loader* without gradients; returns the mean loss."""
        self.model.eval()
        total_loss, total_images = 0.0, 0
        for batch_index, batch in enumerate(loader):
            images = _images_from_batch(batch).to(self.device, non_blocking=True)
            reconstruction = self.model(images)
            loss = self.criterion(reconstruction, images)
            self._assert_finite(loss, f"val epoch {epoch} batch {batch_index}")
            batch_size = images.shape[0]
            total_loss += float(loss) * batch_size
            total_images += batch_size
        if total_images == 0:
            raise RuntimeError("Validation loader yielded no batches")
        return total_loss / total_images

    # ------------------------------------------------------------ checkpoints
    def save_checkpoint(self, filename: str, epoch: int, records: Sequence[EpochRecord]) -> Path:
        """Persist model/optimizer/RNG state plus history into ``checkpoints/``."""
        payload = {
            "epoch": epoch,  # number of completed epochs
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "records": [record.to_dict() for record in records],
            "best_metric": self.best_metric,
            "config": self.config,
            "rng": _rng_states(),
            "torch_version": torch.__version__,
        }
        path = self.checkpoint_dir / filename
        torch.save(payload, path)
        return path

    def load_checkpoint(self, path: Union[Path, str]) -> int:
        """Restore state from *path*; returns the next epoch index (0-based)."""
        # weights_only=False: our payload embeds python/torch RNG states (tuples),
        # which the restricted unpickler would refuse. Checkpoints are our own files.
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_metric = checkpoint.get("best_metric")
        if "rng" in checkpoint:
            try:
                _restore_rng_states(checkpoint["rng"])
            except Exception as exc:  # noqa: BLE001 - RNG restore is best-effort
                print(f"Warning: could not restore RNG state from checkpoint ({exc})")
        self._records = [EpochRecord.from_dict(r) for r in checkpoint.get("records", [])]
        return int(checkpoint["epoch"])

    # -------------------------------------------------------------------- fit
    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: Optional[int] = None,
        resume_from: Optional[Union[Path, str]] = None,
    ) -> TrainingResult:
        """Train until *epochs* total epochs have been completed.

        ``epochs`` counts epochs overall (not extra epochs): resuming a run
        that finished 10 of 30 epochs with ``fit(..., epochs=30)`` runs epochs
        11-30. The returned :class:`TrainingResult` includes the pre-resume
        records as well.
        """
        if epochs is None or epochs < 1:
            raise ValueError(f"epochs must be an int >= 1, got {epochs!r}")
        records: List[EpochRecord] = []
        start_epoch = 0
        if resume_from is not None:
            start_epoch = self.load_checkpoint(resume_from)
            records = self._records
            print(f"Resumed from {resume_from} (completed {start_epoch} epochs)")
        if start_epoch >= epochs:
            print(f"Checkpoint already covers {start_epoch} >= {epochs} epochs - nothing to do")
            return TrainingResult(
                records=records,
                total_seconds=0.0,
                best_metric=self.best_metric,
                run_dir=self.run_dir,
                checkpoint_paths={
                    name: self.checkpoint_dir / name for name in ("last.pt", "best.pt")
                },
            )

        run_started = time.perf_counter()
        for epoch_index in range(start_epoch, epochs):
            epoch_number = epoch_index + 1
            epoch_started = time.perf_counter()
            self._current_lr = self._get_lr()
            train_loss = self.train_one_epoch(train_loader, epoch_number)
            val_loss = (
                self.validate(val_loader, epoch_number) if val_loader is not None else None
            )
            seconds = time.perf_counter() - epoch_started
            record = EpochRecord(
                epoch=epoch_number,
                train_loss=train_loss,
                val_loss=val_loss,
                seconds=seconds,
                lr=self._current_lr,
            )
            records.append(record)

            metric = val_loss if val_loss is not None else train_loss
            improved = self.best_metric is None or metric < self.best_metric
            if improved:
                self.best_metric = metric
            self.save_checkpoint("last.pt", epoch_number, records)
            if improved:
                self.save_checkpoint("best.pt", epoch_number, records)
            write_history(records, self.run_dir)

            metric_name = "val_loss" if val_loss is not None else "train_loss"
            marker = " *" if improved else ""
            val_text = f"  val_loss {val_loss:.6f}" if val_loss is not None else ""
            print(
                f"epoch {epoch_number:3d}/{epochs}  "
                f"train_loss {train_loss:.6f}{val_text}  "
                f"{seconds:6.1f}s  ({metric_name} {metric:.6f}{marker})"
            )

        total_seconds = time.perf_counter() - run_started
        ran = max(1, epochs - start_epoch)
        summary = f"Training finished: {epochs} total epochs in {total_seconds:.1f}s ({total_seconds / ran:.1f}s/epoch)"
        if self.best_metric is not None:
            summary += f". Best metric: {self.best_metric:.6f}"
        print(summary)
        return TrainingResult(
            records=records,
            total_seconds=total_seconds,
            best_metric=self.best_metric,
            run_dir=self.run_dir,
            checkpoint_paths={
                name: self.checkpoint_dir / name for name in ("last.pt", "best.pt")
            },
        )


def build_optimizer(
    model: nn.Module,
    name: str,
    lr: float,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """Construct Adam or AdamW over all trainable parameters (baseline choice).

    Adam is the default for the baseline: it adapts per-parameter step sizes,
    which suits a from-scratch CNN trained for a few dozen epochs. AdamW adds
    decoupled weight decay; both are supported so the config can switch with
    one key.
    """
    if lr <= 0.0:
        raise ValueError(f"lr must be positive, got {lr}")
    if weight_decay < 0.0:
        raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("Model has no trainable parameters")
    key = name.lower()
    if key == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if key == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer {name!r}; expected 'adam' or 'adamw'")
