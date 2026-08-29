import numpy as np
import pytest

from src.metrics.ood import bootstrap_ood_metrics, compute_ood_metrics


def test_perfect_ood_separation() -> None:
    metrics = compute_ood_metrics(np.array([3.0, 4.0]), np.array([1.0, 2.0]))
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["fpr95"] == pytest.approx(0.0)


def test_bootstrap_is_deterministic_and_contains_point_estimate() -> None:
    id_scores = np.array([0.5, 1.0, 1.5, 2.0])
    ood_scores = np.array([-1.0, 0.0, 0.7, 1.2])
    first = bootstrap_ood_metrics(id_scores, ood_scores, repetitions=50, seed=7)
    second = bootstrap_ood_metrics(id_scores, ood_scores, repetitions=50, seed=7)
    assert first == second
    assert first["auroc_low"] <= first["auroc"] <= first["auroc_high"]


def test_rejects_empty_scores() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        compute_ood_metrics(np.array([]), np.array([1.0]))
