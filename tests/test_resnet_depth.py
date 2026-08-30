import pytest

torch = pytest.importorskip("torch")

from src.models.resnet_depth import ResNet18Depth


def test_resnet_depth_shapes_and_cores_layers():
    model = ResNet18Depth(pretrained_encoder=False, max_depth=10.0).eval()
    inputs = torch.rand(1, 3, 224, 304)
    with torch.no_grad():
        features = model.forward_features(inputs)
        prediction = model(inputs)
    assert prediction.shape == (1, 1, 224, 304)
    assert set(features) == {
        "encoder_early", "encoder_middle", "encoder_late", "decoder_late"
    }
    assert set(model.cores_response_modules()) == set(features)
    assert prediction.min() >= 0
    assert prediction.max() <= 10
