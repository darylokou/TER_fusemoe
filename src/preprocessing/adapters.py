"""Output adapters for standardized multimodal samples."""

from __future__ import annotations

import torch
from typing import Any, Dict


class TERAdapter:
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        mod_keys = ["mod1", "mod2"]

        modalities = sample.get("modalities", {})
        mask_dict = sample.get("mask", {})

        mask_values = [mask_dict.get(k, 0) for k in mod_keys]

        return {
            "mod1": modalities.get("mod1"),
            "mod2": modalities.get("mod2"),
            "mask": torch.tensor(mask_values, dtype=torch.float32),
            "label": sample.get("target"),
        }


class ClinGenAdapter:
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "inputs": sample.get("modalities", {}),
            "mask": sample.get("mask", {}),
            "target": sample.get("target"),
            "metadata": sample.get("metadata", {}),
        }


class FuseMoEAdapter:
    def __call__(self, sample: Dict[str, Any]):
        mods = sample.get("modalities", {})

        x1 = mods.get("mod1")
        x2 = mods.get("mod2")

        if x1 is None or x2 is None:
            raise ValueError("FuseMoEAdapter requires both modalities")

        return x1, x2
