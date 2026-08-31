"""Generate the cross-architecture summary figure from verified results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
data = pd.read_csv(ROOT / "results" / "architecture_comparison.csv")
labels = ["MobileNetV2", "ResNet18"]
colors = ["#4472C4", "#ED7D31"]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.1))

axes[0].bar(labels, data["nyu_test_delta1"], color=colors)
axes[0].set_title("NYU official test depth")
axes[0].set_ylabel(r"$\delta_1$ (higher is better)")
axes[0].set_ylim(0, 1)

axes[1].bar(labels, data["middle_magnitude_auroc"], color=colors)
axes[1].set_title("CORES: NYU vs KITTI")
axes[1].set_ylabel("AUROC (higher is better)")
axes[1].set_ylim(0, 1.05)

axes[2].bar(labels, data["middle_magnitude_fpr95"], color=colors)
axes[2].set_title("CORES false positives")
axes[2].set_ylabel("FPR95 (lower is better)")
axes[2].set_ylim(0, 1)

for ax in axes:
    ax.grid(axis="y", alpha=0.25)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3)

fig.suptitle("Architecture ablation: depth quality does not predict OOD quality", weight="bold")
fig.tight_layout()
output = ROOT / "figures" / "09_architecture_comparison.png"
fig.savefig(output, dpi=200, bbox_inches="tight")
print(output)
