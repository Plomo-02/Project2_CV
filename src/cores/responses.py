"""Capture pre-activation convolutional responses with forward hooks."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn


class ConvolutionResponseExtractor:
    """Context manager capturing outputs from named convolution modules."""

    def __init__(self, modules: Mapping[str, nn.Module], *, detach: bool = True) -> None:
        if not modules:
            raise ValueError("At least one response module is required.")
        self.modules = dict(modules)
        self.detach = detach
        self._responses: dict[str, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _hook(self, name: str):
        def capture(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output):
            if not isinstance(output, torch.Tensor) or output.ndim != 4:
                raise ValueError(f"CORES layer {name!r} did not produce [B,C,H,W].")
            self._responses[name] = output.detach() if self.detach else output

        return capture

    def __enter__(self) -> "ConvolutionResponseExtractor":
        self._responses.clear()
        self._handles = [
            module.register_forward_hook(self._hook(name))
            for name, module in self.modules.items()
        ]
        return self

    def __exit__(self, *_exc_info) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @property
    def responses(self) -> dict[str, torch.Tensor]:
        missing = set(self.modules) - set(self._responses)
        if missing:
            raise RuntimeError(f"No responses captured for layers: {sorted(missing)}")
        return dict(self._responses)
