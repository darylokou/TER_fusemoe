"""Sparse router variants for FuseMoE.

All router variants in this file use sparse MoE dispatch only.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np


def _as_2d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError("Each modality feature must be [B, D] or [D]")
    return arr


def _mean_pool(outputs: Dict[str, np.ndarray], modality_order: Iterable[str]) -> np.ndarray:
    tensors = [outputs[m] for m in modality_order if m in outputs]
    if len(tensors) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    stacked = np.stack(tensors, axis=1)
    return stacked.mean(axis=1).astype(np.float32)


class JointSparseRouter:
    """Single sparse router over concatenated multi-modal representation."""

    def __init__(self, sparse_moe, modality_order: Optional[List[str]] = None, seed: int = 101):
        self.sparse_moe = sparse_moe
        self.modality_order = modality_order or ["vitals", "cxr", "text", "ecg"]
        self._proj = None
        self._rng = np.random.default_rng(seed)

    def _ensure_proj(self, in_dim: int) -> None:
        if self._proj is None or self._proj.shape[0] != in_dim:
            d_model = self.sparse_moe.cfg.model_dim
            self._proj = self._rng.standard_normal((in_dim, d_model), dtype=np.float32) / np.sqrt(
                max(1, in_dim)
            )

    def _concat_modalities(self, modality_dict: Dict[str, np.ndarray]) -> np.ndarray:
        feats = []
        batch_size = None
        for name in self.modality_order:
            if name not in modality_dict:
                continue
            arr = _as_2d(modality_dict[name])
            if batch_size is None:
                batch_size = arr.shape[0]
            elif arr.shape[0] != batch_size:
                raise ValueError("All modalities must share batch size")
            feats.append(arr)

        if len(feats) == 0:
            raise ValueError("No modalities available for joint routing")
        return np.concatenate(feats, axis=-1)

    def forward(self, modality_dict):
        x = self._concat_modalities(modality_dict)
        self._ensure_proj(x.shape[1])
        x_proj = x @ self._proj
        return self.sparse_moe.forward(x_proj)

    def forward_with_routing(self, modality_dict):
        x = self._concat_modalities(modality_dict)
        self._ensure_proj(x.shape[1])
        x_proj = x @ self._proj
        return self.sparse_moe.forward_with_routing(x_proj)


class ModalitySpecificSparseRouter:
    """One sparse router path per modality with shared sparse experts."""

    def __init__(self, shared_sparse_moe, modality_names):
        self.shared_sparse_moe = shared_sparse_moe
        self.modality_names = list(modality_names)
        self._proj = {}
        self._rng = np.random.default_rng(211)

    def _project(self, name: str, x: np.ndarray) -> np.ndarray:
        if name not in self._proj or self._proj[name].shape[0] != x.shape[1]:
            d_model = self.shared_sparse_moe.cfg.model_dim
            self._proj[name] = self._rng.standard_normal((x.shape[1], d_model), dtype=np.float32) / np.sqrt(
                max(1, x.shape[1])
            )
        return x @ self._proj[name]

    def forward(self, modality_dict):
        outputs = {}
        for m in self.modality_names:
            if m not in modality_dict:
                continue
            x_m = _as_2d(modality_dict[m])
            x_m = self._project(m, x_m)
            outputs[m] = self.shared_sparse_moe.forward(x_m)

        return _mean_pool(outputs, self.modality_names)

    def forward_with_routing(self, modality_dict):
        outputs = {}
        probs = {}
        for m in self.modality_names:
            if m not in modality_dict:
                continue
            x_m = _as_2d(modality_dict[m])
            x_m = self._project(m, x_m)
            y_m, probs_m = self.shared_sparse_moe.forward_with_routing(x_m)
            outputs[m] = y_m
            probs[m] = probs_m

        fused = _mean_pool(outputs, self.modality_names)
        return fused, probs


class DisjointSparseRouter:
    """Dedicated sparse expert pool per modality (disjoint experts)."""

    def __init__(self, modality_sparse_moe):
        # modality_sparse_moe: dict[str, SparseFuseMoE]
        self.modality_sparse_moe = modality_sparse_moe
        self.modality_names = list(modality_sparse_moe.keys())
        self._proj = {}
        self._rng = np.random.default_rng(307)

    def _project(self, name: str, x: np.ndarray) -> np.ndarray:
        if name not in self._proj or self._proj[name].shape[0] != x.shape[1]:
            d_model = self.modality_sparse_moe[name].cfg.model_dim
            self._proj[name] = self._rng.standard_normal((x.shape[1], d_model), dtype=np.float32) / np.sqrt(
                max(1, x.shape[1])
            )
        return x @ self._proj[name]

    def forward(self, modality_dict):
        outputs = {}
        for m, moe in self.modality_sparse_moe.items():
            if m not in modality_dict:
                continue
            x_m = _as_2d(modality_dict[m])
            x_m = self._project(m, x_m)
            outputs[m] = moe.forward(x_m)

        return _mean_pool(outputs, self.modality_names)

    def forward_with_routing(self, modality_dict):
        outputs = {}
        probs = {}
        for m, moe in self.modality_sparse_moe.items():
            if m not in modality_dict:
                continue
            x_m = _as_2d(modality_dict[m])
            x_m = self._project(m, x_m)
            y_m, probs_m = moe.forward_with_routing(x_m)
            outputs[m] = y_m
            probs[m] = probs_m

        fused = _mean_pool(outputs, self.modality_names)
        return fused, probs
