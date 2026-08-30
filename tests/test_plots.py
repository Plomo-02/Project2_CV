from pathlib import Path

import pandas as pd

from src.visualization.plots import plot_training


def test_training_plot_is_written(tmp_path: Path):
    history = pd.DataFrame({
        "epoch": [1, 2], "train_loss": [1.0, 0.8], "validation_loss": [1.1, 0.9],
        "rmse": [1.2, 1.0], "abs_rel": [0.3, 0.2], "delta1": [0.5, 0.7],
    })
    output = plot_training(history, tmp_path)
    assert output.is_file()
    assert output.stat().st_size > 0
