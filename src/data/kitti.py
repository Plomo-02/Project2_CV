"""KITTI depth-prediction validation data utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from torch.utils.data import Dataset


KITTI_DEPTH_SCALE = 256.0


def _find_unique_directory(root: Path, directory_name: str) -> Path:
    candidates = sorted(
        path for path in root.rglob(directory_name) if path.is_dir()
    )
    if not candidates:
        raise FileNotFoundError(
            f"Could not find a '{directory_name}' directory below {root}."
        )
    if len(candidates) > 1:
        paths = "\n".join(f"  - {path}" for path in candidates)
        raise RuntimeError(
            f"Found multiple '{directory_name}' directories below {root}:\n{paths}"
        )
    return candidates[0]


def find_kitti_validation_directories(root: str | Path) -> tuple[Path, Path]:
    """Locate official ``val_selection_cropped`` RGB and ground-truth folders."""
    root = Path(root)
    selection_roots = sorted(
        path for path in root.rglob("val_selection_cropped") if path.is_dir()
    )
    if len(selection_roots) != 1:
        raise FileNotFoundError(
            "Expected exactly one val_selection_cropped directory below "
            f"{root}, found {len(selection_roots)}."
        )
    selection_root = selection_roots[0]
    image_dir = selection_root / "image"
    depth_dir = selection_root / "groundtruth_depth"
    if not image_dir.is_dir() or not depth_dir.is_dir():
        raise FileNotFoundError(
            f"Incomplete KITTI validation selection below {selection_root}."
        )
    return image_dir, depth_dir


def _rgb_to_tensor(image: Image.Image, output_size: tuple[int, int]) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    tensor = functional.interpolate(
        tensor, size=output_size, mode="bilinear", align_corners=False
    )
    return tensor.squeeze(0)


class KITTIDepthValidationDataset(Dataset[dict[str, object]]):
    """Load the 1000 paired samples from KITTI ``val_selection_cropped``.

    RGB is resized for network input. Ground truth remains at native resolution;
    model predictions must be resized back before computing official-style depth
    metrics. For OOD scoring only the resized RGB tensor is required.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        output_size: tuple[int, int] = (224, 304),
        limit: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.output_size = output_size
        image_dir, depth_dir = find_kitti_validation_directories(self.root)

        image_paths = sorted(image_dir.glob("*.png"))
        pairs = [(path, depth_dir / path.name) for path in image_paths]
        missing = [depth for _, depth in pairs if not depth.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} KITTI ground-truth files; first: {missing[0]}"
            )
        if not pairs:
            raise FileNotFoundError(f"No KITTI PNG pairs found below {image_dir}.")
        self.pairs = pairs[:limit] if limit is not None else pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_path, depth_path = self.pairs[index]
        with Image.open(image_path) as image_file:
            original_size = (image_file.height, image_file.width)
            image = _rgb_to_tensor(image_file, self.output_size)
        with Image.open(depth_path) as depth_file:
            depth_array = np.asarray(depth_file, dtype=np.float32) / KITTI_DEPTH_SCALE

        depth = torch.from_numpy(depth_array).unsqueeze(0)
        valid_mask = depth > 0
        return {
            "image": image,
            "depth": depth,
            "valid_mask": valid_mask,
            "original_size": original_size,
            "sample_id": image_path.stem,
        }

