import pytest
import torch

from src.cores.score import single_layer_cores


def test_single_layer_components_match_hand_calculation() -> None:
    responses = torch.tensor(
        [[[[2.0, 0.0], [-2.0, 0.0]], [[3.0, 1.0], [-4.0, 0.0]]]]
    )
    result = single_layer_cores(
        responses, tau_positive=1.0, tau_negative=-1.0,
        lambda_magnitude=1.0, lambda_frequency=1.0,
    )
    assert result.response_magnitude_positive.item() == pytest.approx(1.5)
    assert result.response_magnitude_negative.item() == pytest.approx(2.0)
    assert result.response_frequency_positive.item() == pytest.approx(1.0)
    assert result.response_frequency_negative.item() == pytest.approx(1.0)
    assert result.log_score.exp().item() == pytest.approx(3.0)


def test_stronger_extremes_produce_higher_id_score() -> None:
    weak = torch.tensor([[[[1.1, -1.1]]]])
    strong = torch.tensor([[[[3.0, -3.0]]]])
    weak_score = single_layer_cores(
        weak, tau_positive=1.0, tau_negative=-1.0
    ).log_score
    strong_score = single_layer_cores(
        strong, tau_positive=1.0, tau_negative=-1.0
    ).log_score
    assert strong_score.item() > weak_score.item()


def test_rejects_post_relu_threshold_configuration() -> None:
    with pytest.raises(ValueError, match="tau_negative"):
        single_layer_cores(torch.ones(1, 2, 2, 2), tau_positive=1.0, tau_negative=1.0)
