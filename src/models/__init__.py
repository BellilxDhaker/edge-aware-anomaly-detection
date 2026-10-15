"""Model building blocks for the baseline autoencoder (no pretrained weights)."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from .autoencoder import (  # noqa: F401
    Autoencoder,
    count_parameters_by_part,
    count_trainable_parameters,
)
from .decoder import Decoder  # noqa: F401
from .encoder import ConvBlock, Encoder  # noqa: F401


def tensor_shape_report(
    model: nn.Module,
    input_shape: Sequence[int] = (1, 3, 256, 256),
    device: Optional[Union[str, torch.device]] = None,
) -> List[str]:
    """Human-readable table of per-stage output shapes and parameter counts.

    Runs one dummy forward pass with forward hooks registered on the model's
    submodules up to depth 2 (encoder/decoder stages, pooling, output head).
    A reused module (the shared max-pool) appears once per invocation, which
    makes the spatial halving visible line by line.

    Returns one line per module invocation, e.g.::

        encoder.stages.0    output (1, 32, 256, 256)   trainable params ...
    """
    was_training = model.training
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    records: List[Tuple[str, Tuple[int, ...], int]] = []

    def make_hook(name: str):
        def hook(_module: nn.Module, _inputs, output: torch.Tensor) -> None:
            params = sum(p.numel() for p in _module.parameters() if p.requires_grad)
            records.append((name, tuple(output.shape), params))

        return hook

    handles = []
    for name, module in model.named_modules():
        # Depth filter: hook encoder/decoder and their stage blocks
        # (encoder.stages.0) but skip deeper internals (encoder.stages.0.block).
        # A hook on the ModuleList itself never fires (forward iterates its
        # children individually) - harmless, and it keeps the code simple.
        if not name or name.count(".") >= 3:
            continue
        handles.append(module.register_forward_hook(make_hook(name)))

    try:
        dummy = torch.zeros(tuple(input_shape), device=device)
        with torch.no_grad():
            model(dummy)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    lines = []
    for name, shape, params in records:
        shape_str = "(" + ", ".join(str(s) for s in shape) + ")"
        lines.append(f"{name:<22s} output {shape_str:<22s} module params {params:>9,}")
    total = count_trainable_parameters(model)
    lines.append(f"{'TOTAL':<22s} trainable parameters {total:>20,}")
    return lines
