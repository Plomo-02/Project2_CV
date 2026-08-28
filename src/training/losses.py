"""Loss functions for monocular metric-depth training."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def resize_depth_target(
    target: torch.Tensor,
    output_size: tuple[int, int],
    *,
    min_depth: float = 1e-3,
    max_depth: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize depth and its validity mask to a prediction resolution."""
    if target.ndim != 4 or target.shape[1] != 1:
        raise ValueError(f"Expected target [B, 1, H, W], got {tuple(target.shape)}")
    valid = (
        torch.isfinite(target)
        & (target >= min_depth)
        & (target <= max_depth)
    )
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    resized_target = functional.interpolate(
        safe_target, size=output_size, mode="bilinear", align_corners=False
    )
    resized_valid = functional.interpolate(
        valid.float(), size=output_size, mode="nearest"
    ).bool()
    return resized_target, resized_valid


def masked_l1_depth_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    min_depth: float = 1e-3,
    max_depth: float = 10.0,
) -> torch.Tensor:
    """Mean absolute metric-depth error over valid target pixels."""
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError(
            f"Expected prediction [B, 1, H, W], got {tuple(prediction.shape)}"
        )
    resized_target, valid = resize_depth_target(
        target,
        prediction.shape[-2:],
        min_depth=min_depth,
        max_depth=max_depth,
    )
    valid &= torch.isfinite(prediction)
    if not torch.any(valid):
        raise ValueError("The batch does not contain valid depth pixels.")
    return torch.abs(prediction[valid] - resized_target[valid]).mean()

