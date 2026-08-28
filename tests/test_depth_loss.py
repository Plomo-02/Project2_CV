import pytest
import torch

from src.training.losses import masked_l1_depth_loss, resize_depth_target


def test_perfect_depth_has_zero_loss() -> None:
    target = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    assert masked_l1_depth_loss(target.clone(), target).item() == pytest.approx(0.0)


def test_invalid_depth_is_ignored() -> None:
    target = torch.tensor([[[[1.0, 0.0], [2.0, float("nan")]]]])
    prediction = torch.tensor([[[[2.0, 100.0], [3.0, 100.0]]]])
    assert masked_l1_depth_loss(prediction, target).item() == pytest.approx(1.0)


def test_resize_preserves_boolean_mask() -> None:
    target = torch.ones(1, 1, 4, 4)
    target[:, :, 0, 0] = 0.0
    resized_target, resized_valid = resize_depth_target(target, (2, 2))
    assert resized_target.shape == (1, 1, 2, 2)
    assert resized_valid.shape == (1, 1, 2, 2)
    assert resized_valid.dtype == torch.bool

