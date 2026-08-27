"""NYU Depth v2 official-split data utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from torch.utils.data import Dataset


NYU_UINT16_MAX = float(2**16 - 1)
NYU_MAX_DEPTH_METERS = 10.0


def find_nyu_split_directory(root: str | Path, split: str) -> Path:
    root = Path(root)
    if split == "train":
        split_dir = root / "train"
    elif split in {"test", "val", "validation"}:
        split_dir = root / "test" / "official"
    else:
        raise ValueError("split must be 'train' or 'test'.")
    if not split_dir.is_dir():
        raise FileNotFoundError(f"NYU split directory not found: {split_dir}")
    return split_dir


def discover_nyu_pairs(root: str | Path, split: str) -> list[tuple[Path, Path]]:
    """Pair ``rgb_<id>.png`` with ``depth_<id>.png`` recursively."""
    split_dir = find_nyu_split_directory(root, split)
    rgb_paths = sorted(split_dir.rglob("rgb_*.png"))
    pairs = [
        (rgb_path, rgb_path.with_name(rgb_path.name.replace("rgb_", "depth_", 1)))
        for rgb_path in rgb_paths
    ]
    missing = [depth_path for _, depth_path in pairs if not depth_path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} NYU depth files; first: {missing[0]}"
        )
    if not pairs:
        raise FileNotFoundError(f"No NYU RGB/depth pairs found below {split_dir}.")
    return pairs


def _rgb_to_tensor(image: Image.Image, output_size: tuple[int, int]) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    tensor = functional.interpolate(
        tensor, size=output_size, mode="bilinear", align_corners=False
    )
    return tensor.squeeze(0)


class NYUDepthDataset(Dataset[dict[str, object]]):
    """Load paired RGB and dense depth images from the Kaggle official split."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        output_size: tuple[int, int] = (224, 304),
        limit: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.output_size = output_size
        pairs = discover_nyu_pairs(self.root, split)
        self.pairs = pairs[:limit] if limit is not None else pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_path, depth_path = self.pairs[index]
        with Image.open(image_path) as image_file:
            original_size = (image_file.height, image_file.width)
            image = _rgb_to_tensor(image_file, self.output_size)
        with Image.open(depth_path) as depth_file:
            encoded_depth = np.asarray(depth_file, dtype=np.float32)

        depth_array = encoded_depth / NYU_UINT16_MAX * NYU_MAX_DEPTH_METERS
        depth = torch.from_numpy(depth_array).unsqueeze(0)
        valid_mask = torch.isfinite(depth) & (depth > 0) & (depth <= NYU_MAX_DEPTH_METERS)
        return {
            "image": image,
            "depth": depth,
            "valid_mask": valid_mask,
            "original_size": original_size,
            "sample_id": image_path.stem.removeprefix("rgb_"),
        }

