"""Reusable training and validation loops for monocular depth estimation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch

from src.metrics.depth import DepthMetricAccumulator, compute_depth_metrics
from src.training.losses import masked_l1_depth_loss, resize_depth_target


Batch = dict[str, Any]


def train_one_epoch(
    model: torch.nn.Module,
    loader: Iterable[Batch],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    normalize: Callable[[torch.Tensor], torch.Tensor] | None = None,
    use_amp: bool = True,
) -> float:
    """Train for one epoch and return the sample-weighted mean loss."""
    model.train()
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["depth"].to(device, non_blocking=True)
        inputs = normalize(images) if normalize is not None else images
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            predictions = model(inputs)
            loss = masked_l1_depth_loss(predictions, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.shape[0]
        total_loss += loss.detach().item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("The training loader is empty.")
    return total_loss / total_samples


@torch.no_grad()
def evaluate_depth(
    model: torch.nn.Module,
    loader: Iterable[Batch],
    device: torch.device,
    *,
    normalize: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[float, dict[str, float]]:
    """Evaluate loss and standard NYU depth metrics."""
    model.eval()
    accumulator = DepthMetricAccumulator()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["depth"].to(device, non_blocking=True)
        inputs = normalize(images) if normalize is not None else images
        predictions = model(inputs)
        loss = masked_l1_depth_loss(predictions, targets)
        resized_targets, _ = resize_depth_target(targets, predictions.shape[-2:])
        metrics = compute_depth_metrics(predictions, resized_targets)

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        accumulator.update(metrics, batch_size)

    if total_samples == 0:
        raise ValueError("The validation loader is empty.")
    return total_loss / total_samples, accumulator.compute()


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_validation_loss: float,
    history: list[dict[str, float]],
) -> None:
    """Persist all state required to resume training."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_validation_loss": best_validation_loss,
            "history": history,
        },
        destination,
    )
