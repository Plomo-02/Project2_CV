"""CLI entry point for regenerating report figures from experiment CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization import generate_all_figures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, help="Directory containing experiment CSV files")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "figures")
    args = parser.parse_args()
    for path in generate_all_figures(args.results, args.output):
        print(path)


if __name__ == "__main__":
    main()
