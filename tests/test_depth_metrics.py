import math

import pytest
import torch

from src.metrics.depth import DepthMetricAccumulator, compute_depth_metrics


def test_perfect_prediction() -> None:
    target = torch.tensor([[[[1.0, 2.0], [4.0, 8.0]]]])

    metrics = compute_depth_metrics(target.clone(), target)

    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["abs_rel"] == pytest.approx(0.0)
    assert metrics["delta1"] == pytest.approx(1.0)
    assert metrics["delta2"] == pytest.approx(1.0)
    assert metrics["delta3"] == pytest.approx(1.0)


def test_known_scaled_prediction() -> None:
    target = torch.ones(1, 1, 2, 2)
    prediction = target * 2.0

    metrics = compute_depth_metrics(prediction, target)

    assert metrics["rmse"] == pytest.approx(1.0)
    assert metrics["abs_rel"] == pytest.approx(1.0)
    assert metrics["delta1"] == pytest.approx(0.0)
    assert metrics["delta2"] == pytest.approx(0.0)
    assert metrics["delta3"] == pytest.approx(0.0)


def test_invalid_target_pixels_are_ignored() -> None:
    target = torch.tensor([[[[1.0, 0.0], [float("nan"), 2.0]]]])
    prediction = torch.tensor([[[[1.0, 100.0], [100.0, 2.0]]]])

    metrics = compute_depth_metrics(prediction, target)

    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["abs_rel"] == pytest.approx(0.0)


def test_all_invalid_batch_is_rejected() -> None:
    target = torch.zeros(1, 1, 2, 2)
    with pytest.raises(ValueError, match="valid depth pixels"):
        compute_depth_metrics(torch.ones_like(target), target)


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_depth_metrics(torch.ones(1, 2, 2), torch.ones(1, 3, 3))


def test_accumulator_uses_batch_size_weighting() -> None:
    accumulator = DepthMetricAccumulator()
    first = {name: 1.0 for name in ("rmse", "abs_rel", "delta1", "delta2", "delta3")}
    second = {name: 3.0 for name in first}

    accumulator.update(first, batch_size=1)
    accumulator.update(second, batch_size=3)
    result = accumulator.compute()

    assert all(math.isclose(value, 2.5) for value in result.values())

