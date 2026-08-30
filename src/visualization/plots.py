"""Generate the publication-ready figures used in the report."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


DISPLAY_NAMES = {
    "encoder_early": "Early encoder",
    "encoder_middle": "Middle encoder",
    "encoder_late": "Late encoder",
    "decoder_late": "Late decoder",
    "multi_layer_mean": "Multi-layer mean",
}


def _save(fig: plt.Figure, destination: Path, name: str) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / name
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _metric_bars(frame: pd.DataFrame, label: str, destination: Path, name: str) -> Path:
    data = frame.reset_index()
    label_column = data.columns[0]
    positions = np.arange(len(data))
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, 0.52 * len(data))))
    for axis, metric, title in zip(axes, ("auroc", "fpr95"), ("AUROC ↑", "FPR95 ↓")):
        axis.barh(positions, data[metric], color="#4472C4")
        axis.set_yticks(positions, data[label_column].map(lambda x: DISPLAY_NAMES.get(x, x)))
        axis.invert_yaxis()
        axis.set_xlim(0, 1.02)
        axis.set_xlabel(title)
        axis.grid(axis="x", alpha=0.25)
        for position, value in zip(positions, data[metric]):
            axis.text(min(value + 0.015, 0.94), position, f"{value:.3f}", va="center")
    fig.suptitle(label)
    fig.tight_layout()
    return _save(fig, destination, name)


def plot_training(history: pd.DataFrame, destination: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(history["epoch"], history["train_loss"], label="Train L1")
    axes[0].plot(history["epoch"], history["validation_loss"], label="Validation L1")
    axes[0].set(xlabel="Epoch", ylabel="Loss (m)", title="FastDepth learning curves")
    axes[0].legend()
    axes[1].plot(history["epoch"], history["rmse"], label="RMSE")
    axes[1].plot(history["epoch"], history["abs_rel"], label="AbsRel")
    axes[1].plot(history["epoch"], history["delta1"], label="δ1")
    axes[1].set(xlabel="Epoch", title="Validation depth metrics")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    return _save(fig, destination, "01_fastdepth_training.png")


def plot_depth_metrics(metrics: pd.DataFrame, destination: Path) -> Path:
    selected = metrics[["rmse", "abs_rel", "delta1"]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for axis, (metric, colour, title) in zip(axes, (
        ("rmse", "#4472C4", "RMSE ↓"),
        ("abs_rel", "#ED7D31", "AbsRel ↓"),
        ("delta1", "#70AD47", "δ1 ↑"),
    )):
        selected[metric].plot.bar(ax=axis, color=colour)
        axis.set(title=title, ylabel="Metric value", xlabel="")
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("FastDepth evaluation across ID and OOD domains")
    fig.tight_layout()
    return _save(fig, destination, "02_fastdepth_depth_metrics.png")


def plot_score_distributions(scores: pd.DataFrame, destination: Path) -> Path:
    layers = list(scores["layer"].drop_duplicates())
    fig, axes = plt.subplots(1, len(layers), figsize=(4.3 * len(layers), 4), squeeze=False)
    for axis, layer in zip(axes[0], layers):
        subset = scores[scores["layer"] == layer]
        for domain, colour in (("NYU_test_ID", "#4472C4"), ("KITTI_OOD", "#ED7D31")):
            values = subset.loc[subset["domain"] == domain, "score"]
            axis.hist(values, bins=35, density=True, alpha=0.62, label=domain, color=colour)
        axis.set(title=DISPLAY_NAMES.get(layer, layer), xlabel="CORES score (higher = ID)")
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend()
    fig.tight_layout()
    return _save(fig, destination, "03_cores_score_distributions.png")


def plot_roc_curves(scores: pd.DataFrame, destination: Path) -> Path:
    fig, axis = plt.subplots(figsize=(6.4, 5.4))
    for layer in scores["layer"].drop_duplicates():
        subset = scores[scores["layer"] == layer]
        labels = (subset["domain"] == "NYU_test_ID").astype(int)
        fpr, tpr, _ = roc_curve(labels, subset["score"])
        axis.plot(fpr, tpr, label=f"{DISPLAY_NAMES.get(layer, layer)} ({auc(fpr, tpr):.4f})")
    axis.plot([0, 1], [0, 1], "--", color="0.5", label="Random")
    axis.set(xlabel="False-positive rate", ylabel="True-positive rate", title="NYU ID vs KITTI OOD")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return _save(fig, destination, "04_cores_roc_curves.png")


def plot_controls(results: Mapping[str, pd.DataFrame], destination: Path) -> Path:
    frames = []
    for key, heading in (
        ("cores_training_effect_control", "Training effect"),
        ("cores_nyu_corruption_detection", "Near-OOD corruptions"),
        ("cores_threshold_calibration_comparison", "Threshold calibration"),
    ):
        data = results[key].copy()
        data["group"] = heading
        data["configuration"] = data.iloc[:, 0].astype(str)
        frames.append(data[["group", "configuration", "auroc", "fpr95"]])
    combined = pd.concat(frames, ignore_index=True)
    positions = np.arange(len(combined))
    colours = combined["group"].map({
        "Training effect": "#4472C4",
        "Near-OOD corruptions": "#ED7D31",
        "Threshold calibration": "#70AD47",
    })
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4))
    for axis, metric, title in zip(axes, ("auroc", "fpr95"), ("AUROC ↑", "FPR95 ↓")):
        axis.barh(positions, combined[metric], color=colours)
        axis.set_yticks(positions, combined["configuration"])
        axis.invert_yaxis()
        axis.set_xlim(0, 1.02)
        axis.set_xlabel(title)
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle("CORES robustness controls")
    fig.tight_layout()
    return _save(fig, destination, "06_cores_robustness_controls.png")


def plot_stability(frame: pd.DataFrame, destination: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for seed, group in frame.groupby("seed"):
        axes[0].plot(group["calibration_samples"], (1 - group["auroc"]) * 1e6,
                     marker="o", alpha=0.75, label=f"seed {seed}")
        axes[1].plot(group["calibration_samples"], group["tau_positive"],
                     marker="o", alpha=0.75, label=f"positive, seed {seed}")
        axes[1].plot(group["calibration_samples"], group["tau_negative"],
                     marker="x", linestyle="--", alpha=0.65, label=f"negative, seed {seed}")
    axes[0].set(xlabel="Calibration samples", ylabel="1 - AUROC (ppm)",
                title="Detection stability; FPR95 = 0 throughout")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Calibration samples", ylabel="Response threshold",
                title="Calibrated threshold stability")
    axes[1].legend(fontsize=7, ncol=2)
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    return _save(fig, destination, "08_cores_calibration_stability.png")


def generate_all_figures(results_directory: str | Path, output_directory: str | Path) -> list[Path]:
    """Load saved experiment tables and render every final figure."""
    results_directory = Path(results_directory)
    output_directory = Path(output_directory)
    tables = {
        path.stem: pd.read_csv(path)
        for path in results_directory.glob("*.csv")
    }
    required = {
        "fastdepth_training_history", "fastdepth_depth_metrics", "cores_scores",
        "cores_ood_metrics", "cores_component_ablation",
        "cores_training_effect_control", "cores_nyu_corruption_detection",
        "cores_threshold_calibration_comparison",
    }
    missing = sorted(required - tables.keys())
    if missing:
        raise FileNotFoundError(f"Missing result tables: {missing}")
    depth_metrics = tables["fastdepth_depth_metrics"].set_index(
        tables["fastdepth_depth_metrics"].columns[0]
    )
    ablation = tables["cores_component_ablation"].copy()
    ablation["configuration"] = (
        ablation["layer"].map(lambda x: DISPLAY_NAMES.get(x, x))
        + " — " + ablation["variant"].str.replace("_", " ")
    )
    ablation = ablation.set_index("configuration")[["auroc", "fpr95"]]
    paths = [
        plot_training(tables["fastdepth_training_history"], output_directory),
        plot_depth_metrics(depth_metrics, output_directory),
        plot_score_distributions(pd.read_csv(results_directory / "cores_scores.csv"), output_directory),
        plot_roc_curves(pd.read_csv(results_directory / "cores_scores.csv"), output_directory),
        _metric_bars(ablation, "CORES component ablation", output_directory, "05_cores_component_ablation.png"),
        plot_controls(tables, output_directory),
    ]
    if "cores_multilayer_aggregation" in tables:
        aggregation = tables["cores_multilayer_aggregation"].set_index("configuration")
        paths.append(_metric_bars(
            aggregation[["auroc", "fpr95"]], "CORES multi-layer aggregation",
            output_directory, "07_cores_multilayer_aggregation.png",
        ))
    if "cores_calibration_stability" in tables:
        paths.append(plot_stability(tables["cores_calibration_stability"], output_directory))
    return paths
