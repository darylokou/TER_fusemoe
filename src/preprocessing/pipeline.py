"""Modular preprocessing pipeline and reusable sample-level transforms."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

import torch

Sample = Dict[str, Any]


class PreprocessingPipeline:
    """Apply a list of preprocessing steps to a standardized sample dict."""

    def __init__(self, steps: Sequence[Callable[[Sample], Sample]]):
        self.steps = list(steps)

    def __call__(self, sample: Sample) -> Sample:
        for step in self.steps:
            sample = step(sample)
        return sample


class MaskGenerator:
    """Generate a binary modality-presence mask from ``sample['modalities']``."""

    def __init__(self, modality_keys: Optional[Iterable[str]] = None):
        self.modality_keys = list(modality_keys) if modality_keys is not None else None

    def __call__(self, sample: Sample) -> Sample:
        modalities = sample.get("modalities", {})
        keys = self.modality_keys if self.modality_keys is not None else modalities.keys()
        sample["mask"] = {key: int(modalities.get(key) is not None) for key in keys}
        return sample


class Normalize:
    """Normalize tensor modalities using modality-specific statistics."""
    def __init__(self, mean=None, std=None, eps=1e-8):
        self.mean = dict(mean or {})
        self.std = dict(std or {})
        self.eps = eps

    def __call__(self, sample: Sample) -> Sample:
        modalities = dict(sample.get("modalities", {}))

        for key, value in modalities.items():
            if value is None:
                continue

            if not isinstance(value, torch.Tensor):
                x = torch.tensor(value, dtype=torch.float32)
            else:
                x = value

            mu = torch.as_tensor(self.mean.get(key, 0.0), dtype=x.dtype, device=x.device)
            sigma = torch.as_tensor(self.std.get(key, 1.0), dtype=x.dtype, device=x.device)

            modalities[key] = (x - mu) / (sigma + self.eps)

        sample["modalities"] = modalities
        return sample


class ModalityTransform:
    """Apply arbitrary transform callables per modality key."""

    def __init__(self, transforms: Mapping[str, Callable[[Any], Any]]):
        self.transforms = dict(transforms)

    def __call__(self, sample: Sample) -> Sample:
        modalities = sample.get("modalities", {})
        for key, transform in self.transforms.items():
            value = modalities.get(key)
            if value is None:
                continue
            modalities[key] = transform(value)
        sample["modalities"] = modalities
        return sample


class ImageTransform(ModalityTransform):
    """Convenience modality transform for image-like modalities."""


class TabularTransform(ModalityTransform):
    """Convenience modality transform for tabular modalities."""
