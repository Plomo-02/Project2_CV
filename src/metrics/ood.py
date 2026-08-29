"""OOD metrics and deterministic bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_ood_metrics(
    id_scores: np.ndarray, ood_scores: np.ndarray
) -> dict[str, float]:
    """Compute AUROC and FPR95 when a higher score means more ID-like."""
    id_scores = np.asarray(id_scores, dtype=np.float64).reshape(-1)
    ood_scores = np.asarray(ood_scores, dtype=np.float64).reshape(-1)
    if id_scores.size == 0 or ood_scores.size == 0:
        raise ValueError("Both ID and OOD score arrays must be non-empty.")
    if not np.isfinite(id_scores).all() or not np.isfinite(ood_scores).all():
        raise ValueError("OOD scores must be finite.")
    labels = np.concatenate([np.ones(id_scores.size), np.zeros(ood_scores.size)])
    scores = np.concatenate([id_scores, ood_scores])
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    candidates = np.flatnonzero(tpr >= 0.95)
    if candidates.size == 0:
        raise RuntimeError("ROC curve did not reach 95% true-positive rate.")
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "fpr95": float(fpr[candidates[0]]),
    }


def bootstrap_ood_metrics(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    *,
    repetitions: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Stratified bootstrap confidence intervals for AUROC and FPR95."""
    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1).")
    id_scores = np.asarray(id_scores, dtype=np.float64).reshape(-1)
    ood_scores = np.asarray(ood_scores, dtype=np.float64).reshape(-1)
    point = compute_ood_metrics(id_scores, ood_scores)
    generator = np.random.default_rng(seed)
    samples = {"auroc": [], "fpr95": []}
    for _ in range(repetitions):
        sampled_id = id_scores[generator.integers(0, id_scores.size, id_scores.size)]
        sampled_ood = ood_scores[
            generator.integers(0, ood_scores.size, ood_scores.size)
        ]
        metrics = compute_ood_metrics(sampled_id, sampled_ood)
        for name in samples:
            samples[name].append(metrics[name])
    alpha = (1.0 - confidence) / 2.0
    result: dict[str, float] = {}
    for name, values in samples.items():
        result[name] = point[name]
        result[f"{name}_low"] = float(np.quantile(values, alpha))
        result[f"{name}_high"] = float(np.quantile(values, 1.0 - alpha))
    return result
