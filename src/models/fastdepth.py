"""Modern PyTorch adaptation of the FastDepth encoder-decoder design."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_imagenet(images: torch.Tensor) -> torch.Tensor:
    """Normalize a [B, 3, H, W] float tensor in the [0, 1] range."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected [B, 3, H, W] RGB tensor, got {tuple(images.shape)}")
    mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (images - mean) / std


class DepthwiseDecoderBlock(nn.Module):
    """5x5 depthwise convolution followed by 1x1 channel projection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=5,
                padding=2,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(inputs))


class FastDepthMobileNetV2(nn.Module):
    """FastDepth-style model with MobileNetV2 and additive skip connections.

    The original FastDepth uses MobileNetV1. This maintained adaptation retains
    its efficient depthwise 5x5 decoder, nearest-neighbour upsampling and
    additive encoder-decoder skips while using torchvision's MobileNetV2.
    """

    def __init__(self, *, pretrained_encoder: bool = True, max_depth: float = 10.0) -> None:
        super().__init__()
        if max_depth <= 0:
            raise ValueError("max_depth must be positive.")
        weights = MobileNet_V2_Weights.DEFAULT if pretrained_encoder else None
        self.encoder = mobilenet_v2(weights=weights).features
        self.decoder4 = DepthwiseDecoderBlock(1280, 96)
        self.decoder3 = DepthwiseDecoderBlock(96, 32)
        self.decoder2 = DepthwiseDecoderBlock(32, 24)
        self.decoder1 = DepthwiseDecoderBlock(24, 16)
        self.decoder0 = DepthwiseDecoderBlock(16, 16)
        self.depth_head = nn.Conv2d(16, 1, kernel_size=1)
        self.max_depth = float(max_depth)

    @staticmethod
    def _upsample_add(inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = functional.interpolate(inputs, size=skip.shape[-2:], mode="nearest")
        return inputs + skip

    def forward_features(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        encoder_features: dict[int, torch.Tensor] = {}
        outputs = inputs
        for index, layer in enumerate(self.encoder):
            outputs = layer(outputs)
            if index in {1, 3, 6, 13}:
                encoder_features[index] = outputs

        late = outputs
        decoder4 = self._upsample_add(self.decoder4(late), encoder_features[13])
        decoder3 = self._upsample_add(self.decoder3(decoder4), encoder_features[6])
        decoder2 = self._upsample_add(self.decoder2(decoder3), encoder_features[3])
        decoder1 = self._upsample_add(self.decoder1(decoder2), encoder_features[1])
        decoder0 = self.decoder0(
            functional.interpolate(decoder1, size=inputs.shape[-2:], mode="nearest")
        )
        return {
            "encoder_early": encoder_features[3],
            "encoder_middle": encoder_features[6],
            "encoder_late": late,
            "decoder_late": decoder0,
        }

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(inputs)
        raw_depth = self.depth_head(features["decoder_late"])
        return torch.sigmoid(raw_depth) * self.max_depth

