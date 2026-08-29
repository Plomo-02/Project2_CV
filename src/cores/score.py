"""CORES response statistics from Tang et al., CVPR 2024."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CoresComponents:
    """Per-sample CORES components and numerically stable log-score."""

    response_magnitude_positive: torch.Tensor
    response_magnitude_negative: torch.Tensor
    response_frequency_positive: torch.Tensor
    response_frequency_negative: torch.Tensor
    log_score: torch.Tensor


def single_layer_cores(
    responses: torch.Tensor,
    *,
    tau_positive: float,
    tau_negative: float,
    lambda_magnitude: float = 10.0,
    lambda_frequency: float = 1.0,
    epsilon: float = 1e-12,
) -> CoresComponents:
    """Compute Eq. (3)-(5) of CORES for a convolutional response tensor.

    The returned log-score is ranking-equivalent to the paper's multiplicative
    score and avoids numerical underflow when ``lambda_magnitude=10``. A higher
    value means more ID-like, following the original CORES convention.
    """
    if responses.ndim != 4:
        raise ValueError(f"Expected responses [B,C,H,W], got {tuple(responses.shape)}")
    if responses.shape[1] == 0 or responses.shape[2] == 0 or responses.shape[3] == 0:
        raise ValueError("CORES responses cannot contain empty dimensions.")
    if tau_negative >= tau_positive:
        raise ValueError("Expected tau_negative < tau_positive.")
    if lambda_magnitude < 0 or lambda_frequency < 0:
        raise ValueError("CORES exponents must be non-negative.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    channel_maxima = responses.amax(dim=(-2, -1))
    channel_minima = responses.amin(dim=(-2, -1))
    rm_positive = torch.relu(channel_maxima - tau_positive).mean(dim=1)
    rm_negative = torch.relu(tau_negative - channel_minima).mean(dim=1)
    rf_positive = (channel_maxima > tau_positive).float().mean(dim=1)
    rf_negative = (channel_minima < tau_negative).float().mean(dim=1)

    log_score = lambda_magnitude * (
        torch.log(rm_positive.clamp_min(epsilon))
        + torch.log(rm_negative.clamp_min(epsilon))
    ) + lambda_frequency * (
        torch.log(rf_positive.clamp_min(epsilon))
        + torch.log(rf_negative.clamp_min(epsilon))
    )
    return CoresComponents(
        rm_positive, rm_negative, rf_positive, rf_negative, log_score
    )


def response_selected_cores(
    responses: torch.Tensor,
    *,
    tau_positive: float,
    tau_negative: float,
    selection_fraction: float = 0.20,
    lambda_magnitude: float = 10.0,
    lambda_frequency: float = 1.0,
    epsilon: float = 1e-12,
) -> CoresComponents:
    """CORES using per-sample extreme-response channel selection.

    This is an explicit dense-prediction adaptation, not the classifier-logit
    backtracking in the original paper. Positive and negative branches select
    the channels with the largest maxima and smallest minima respectively.
    """
    if responses.ndim != 4:
        raise ValueError(f"Expected responses [B,C,H,W], got {tuple(responses.shape)}")
    if not 0 < selection_fraction <= 1:
        raise ValueError("selection_fraction must be in (0, 1].")
    channel_count = responses.shape[1]
    selected_count = max(1, round(channel_count * selection_fraction))
    maxima = responses.amax(dim=(-2, -1))
    minima = responses.amin(dim=(-2, -1))
    positive_indices = maxima.topk(selected_count, dim=1).indices
    negative_indices = minima.topk(selected_count, dim=1, largest=False).indices
    spatial_shape = responses.shape[-2:]
    positive = responses.gather(
        1, positive_indices[..., None, None].expand(-1, -1, *spatial_shape)
    )
    negative = responses.gather(
        1, negative_indices[..., None, None].expand(-1, -1, *spatial_shape)
    )
    positive_components = single_layer_cores(
        positive,
        tau_positive=tau_positive,
        tau_negative=tau_negative,
        lambda_magnitude=lambda_magnitude,
        lambda_frequency=lambda_frequency,
        epsilon=epsilon,
    )
    negative_components = single_layer_cores(
        negative,
        tau_positive=tau_positive,
        tau_negative=tau_negative,
        lambda_magnitude=lambda_magnitude,
        lambda_frequency=lambda_frequency,
        epsilon=epsilon,
    )
    rm_positive = positive_components.response_magnitude_positive
    rf_positive = positive_components.response_frequency_positive
    rm_negative = negative_components.response_magnitude_negative
    rf_negative = negative_components.response_frequency_negative
    log_score = lambda_magnitude * (
        torch.log(rm_positive.clamp_min(epsilon))
        + torch.log(rm_negative.clamp_min(epsilon))
    ) + lambda_frequency * (
        torch.log(rf_positive.clamp_min(epsilon))
        + torch.log(rf_negative.clamp_min(epsilon))
    )
    return CoresComponents(
        rm_positive, rm_negative, rf_positive, rf_negative, log_score
    )
