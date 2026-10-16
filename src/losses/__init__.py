"""Loss functions: baseline MSE (Phase 2) and edge-aware variant (Phase 4)."""

from .edge_aware import (  # noqa: F401
    EdgeAwareReconstructionLoss,
    gradient_magnitude_mse,
    spatial_gradient_magnitude,
)
from .reconstruction import (  # noqa: F401
    ReconstructionLoss,
    absolute_error_map,
    mse_reconstruction_loss,
    per_image_mse,
)

__all__ = [
    "EdgeAwareReconstructionLoss",
    "ReconstructionLoss",
    "absolute_error_map",
    "gradient_magnitude_mse",
    "mse_reconstruction_loss",
    "per_image_mse",
    "spatial_gradient_magnitude",
]
