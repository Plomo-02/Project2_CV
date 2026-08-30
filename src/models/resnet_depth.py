"""Lightweight ResNet18 encoder-decoder for architecture ablation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torchvision.models import ResNet18_Weights, resnet18

from .fastdepth import DepthwiseDecoderBlock


class ResNet18Depth(nn.Module):
    """Metric-depth network with a ResNet18 encoder and efficient decoder.

    The decoder intentionally follows the same depthwise, nearest-upsampling,
    additive-skip design used by the FastDepth-style baseline. This isolates the
    encoder architecture as the main controlled difference.
    """

    def __init__(self, *, pretrained_encoder: bool = True, max_depth: float = 10.0) -> None:
        super().__init__()
        if max_depth <= 0:
            raise ValueError("max_depth must be positive.")
        weights = ResNet18_Weights.DEFAULT if pretrained_encoder else None
        backbone = resnet18(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.decoder4 = DepthwiseDecoderBlock(512, 256)
        self.decoder3 = DepthwiseDecoderBlock(256, 128)
        self.decoder2 = DepthwiseDecoderBlock(128, 64)
        self.decoder1 = DepthwiseDecoderBlock(64, 64)
        self.decoder0 = DepthwiseDecoderBlock(64, 32)
        self.depth_head = nn.Conv2d(32, 1, kernel_size=1)
        self.max_depth = float(max_depth)

    @staticmethod
    def _upsample_add(inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = functional.interpolate(inputs, size=skip.shape[-2:], mode="nearest")
        return inputs + skip

    def cores_response_modules(self) -> dict[str, nn.Conv2d]:
        """Return signed convolution outputs at comparable network depths."""
        return {
            "encoder_early": self.stem[0],
            "encoder_middle": self.layer2[-1].conv2,
            "encoder_late": self.layer4[-1].conv2,
            "decoder_late": self.decoder0.pointwise[0],
        }

    def forward_features(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        stem = self.stem(inputs)
        layer1 = self.layer1(self.maxpool(stem))
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        decoder4 = self._upsample_add(self.decoder4(layer4), layer3)
        decoder3 = self._upsample_add(self.decoder3(decoder4), layer2)
        decoder2 = self._upsample_add(self.decoder2(decoder3), layer1)
        decoder1 = self._upsample_add(self.decoder1(decoder2), stem)
        decoder0 = self.decoder0(
            functional.interpolate(decoder1, size=inputs.shape[-2:], mode="nearest")
        )
        return {
            "encoder_early": stem,
            "encoder_middle": layer2,
            "encoder_late": layer4,
            "decoder_late": decoder0,
        }

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(inputs)
        raw_depth = self.depth_head(features["decoder_late"])
        return torch.sigmoid(raw_depth) * self.max_depth
