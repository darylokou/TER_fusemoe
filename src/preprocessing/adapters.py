"""Output adapters for standardized multimodal samples."""

from __future__ import annotations

from typing import Any, Dict

import torch


class TERAdapter:
    """Adapt standardized samples to TER-style training dictionaries."""

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        mask_values = list(sample.get("mask", {}).values())
        return {
            "mod1": sample["modalities"].get("mod1"),
            "mod2": sample["modalities"].get("mod2"),
            "mask": torch.tensor(mask_values, dtype=torch.float32),
            "label": sample.get("target"),
        }


class ClinGenAdapter:
    """Adapt standardized samples to ClinGen-style dictionaries."""

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "inputs": sample.get("modalities", {}),
            "mask": sample.get("mask", {}),
            "target": sample.get("target"),
            "metadata": sample.get("metadata", {}),
        }


class FuseMoEAdapter:
    """Adapt standardized samples to simple tuple outputs."""

    def __call__(self, sample: Dict[str, Any]):
        mods = sample.get("modalities", {})
        return mods.get("mod1"), mods.get("mod2")
