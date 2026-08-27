"""Depth-estimation metrics shared by training and evaluation."""

from __future__ import annotations

from collections.abc import Mapping

import torch


DEPTH_METRIC_NAMES = ("rmse", "abs_rel", "delta1", "delta2", "delta3")


def _as_bhw(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Convert a depth tensor to [B, H, W] without changing its values."""
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim == 3:
        return tensor
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        return tensor[:, 0]
    raise ValueError(
        f"{name} must have shape [H, W], [B, H, W], or [B, 1, H, W]; "
        f"got {tuple(tensor.shape)}"
    )


@torch.no_grad()
def compute_depth_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    min_depth: float = 1e-3,
    max_depth: float = 10.0,
) -> dict[str, float]:
    """Compute common MDE metrics, averaging valid per-image results.

    Valid pixels have finite prediction/target values and target depth inside
    ``[min_depth, max_depth]``. Predictions are clipped to that same interval.
    Images without valid pixels are ignored; an all-invalid batch is rejected.
    """
    if min_depth <= 0 or max_depth <= min_depth:
        raise ValueError("Expected 0 < min_depth < max_depth.")

    prediction = _as_bhw(prediction.detach(), "prediction").float()
    target = _as_bhw(target.detach(), "target").float()
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: {tuple(prediction.shape)} vs "
            f"{tuple(target.shape)}"
        )

    per_image: dict[str, list[torch.Tensor]] = {
        name: [] for name in DEPTH_METRIC_NAMES
    }

    for predicted_depth, target_depth in zip(prediction, target):
        valid = (
            torch.isfinite(predicted_depth)
            & torch.isfinite(target_depth)
            & (target_depth >= min_depth)
            & (target_depth <= max_depth)
        )
        if not torch.any(valid):
            continue

        pred = predicted_depth[valid].clamp(min=min_depth, max=max_depth)
        true = target_depth[valid]
        ratio = torch.maximum(true / pred, pred / true)

        per_image["rmse"].append(torch.sqrt(torch.mean((pred - true) ** 2)))
        per_image["abs_rel"].append(torch.mean(torch.abs(pred - true) / true))
        per_image["delta1"].append(torch.mean((ratio < 1.25).float()))
        per_image["delta2"].append(torch.mean((ratio < 1.25**2).float()))
        per_image["delta3"].append(torch.mean((ratio < 1.25**3).float()))

    if not per_image["rmse"]:
        raise ValueError("The batch does not contain any valid depth pixels.")

    return {
        name: torch.stack(values).mean().item()
        for name, values in per_image.items()
    }


class DepthMetricAccumulator:
    """Accumulate batch metrics using the number of images as weight."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._totals = {name: 0.0 for name in DEPTH_METRIC_NAMES}
        self._images = 0

    def update(self, metrics: Mapping[str, float], batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        missing = set(DEPTH_METRIC_NAMES) - set(metrics)
        if missing:
            raise KeyError(f"Missing depth metrics: {sorted(missing)}")
        for name in DEPTH_METRIC_NAMES:
            self._totals[name] += float(metrics[name]) * batch_size
        self._images += batch_size

    def compute(self) -> dict[str, float]:
        if self._images == 0:
            raise ValueError("No depth metrics have been accumulated.")
        return {
            name: total / self._images for name, total in self._totals.items()
        }

