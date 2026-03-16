"""Sparse FuseMoE pseudocode.

This module intentionally implements only sparse MoE routing (no dense fallback):
- Laplace gating with negative L2 distances.
- Top-K sparse dispatch.
- 3 stacked sparse MoE layers.
"""

from dataclasses import dataclass

import numpy as np


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.clip(np.sum(exp_x, axis=axis, keepdims=True), 1e-8, None)


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


@dataclass
class SparseMoEConfig:
    n_experts: int = 16
    top_k: int = 4
    n_layers: int = 3
    model_dim: int = 128
    expert_hidden_dim: int = 512


class ExpertFFN:
    """Expert network Ei: FFN(128 -> 512 -> 128) with GeLU."""

    def __init__(self, cfg: SparseMoEConfig):
        self.cfg = cfg
        rng = np.random.default_rng()
        self.fc1_w = rng.standard_normal((cfg.model_dim, cfg.expert_hidden_dim), dtype=np.float32) / np.sqrt(
            cfg.model_dim
        )
        self.fc1_b = np.zeros((cfg.expert_hidden_dim,), dtype=np.float32)
        self.fc2_w = rng.standard_normal((cfg.expert_hidden_dim, cfg.model_dim), dtype=np.float32) / np.sqrt(
            cfg.expert_hidden_dim
        )
        self.fc2_b = np.zeros((cfg.model_dim,), dtype=np.float32)

    def forward(self, x):
        h = _gelu(x @ self.fc1_w + self.fc1_b)
        y = h @ self.fc2_w + self.fc2_b
        return y.astype(np.float32)


class LaplaceTopKGate:
    """Sparse Top-K gate: h_l(x) = TopK(-||W - x||_2)."""

    def __init__(self, cfg: SparseMoEConfig):
        self.cfg = cfg
        rng = np.random.default_rng(41)
        self.anchors = rng.standard_normal((cfg.n_experts, cfg.model_dim), dtype=np.float32) / np.sqrt(
            cfg.model_dim
        )

    def forward(self, x):
        """Return sparse routing weights and selected expert ids.

        x: [B, D]
        returns:
          weights: [B, top_k]
          expert_idx: [B, top_k]
        """
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim != 2:
            raise ValueError("x must be [B, D]")

        if x_arr.shape[1] != self.anchors.shape[1]:
            raise ValueError("x feature dimension must equal model_dim")

        diff = x_arr[:, None, :] - self.anchors[None, :, :]
        dist = np.linalg.norm(diff, ord=2, axis=-1)
        logits = -dist

        k = min(self.cfg.top_k, self.cfg.n_experts)
        part = np.argpartition(logits, kth=logits.shape[1] - k, axis=1)[:, -k:]
        part_vals = np.take_along_axis(logits, part, axis=1)
        order = np.argsort(-part_vals, axis=1)
        expert_idx = np.take_along_axis(part, order, axis=1)
        topk_vals = np.take_along_axis(logits, expert_idx, axis=1)

        weights = _softmax(topk_vals, axis=-1)
        return weights.astype(np.float32), expert_idx.astype(np.int64)

    def dense_probs(self, x) -> np.ndarray:
        """Return sparse probabilities expanded to [B, n_experts]."""
        w, idx = self.forward(x)
        probs = np.zeros((w.shape[0], self.cfg.n_experts), dtype=np.float32)
        np.put_along_axis(probs, idx, w, axis=1)
        return probs


class SparseMoELayer:
    """Single sparse MoE layer with Laplace Top-K routing only."""

    def __init__(self, cfg: SparseMoEConfig):
        self.cfg = cfg
        self.gate = LaplaceTopKGate(cfg)
        self.experts = [ExpertFFN(cfg) for _ in range(cfg.n_experts)]

    def forward(self, x):
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim != 2:
            raise ValueError("x must be [B, D]")

        weights, idx = self.gate.forward(x_arr)
        b, d = x_arr.shape
        y = np.zeros((b, d), dtype=np.float32)

        # Dispatch each selected route to its assigned expert.
        for slot in range(weights.shape[1]):
            e_ids = idx[:, slot]
            w_slot = weights[:, slot : slot + 1]

            slot_out = np.zeros_like(y)
            for e in range(self.cfg.n_experts):
                sel = np.where(e_ids == e)[0]
                if len(sel) == 0:
                    continue
                slot_out[sel] = self.experts[e].forward(x_arr[sel])

            y += w_slot * slot_out

        return y.astype(np.float32)

    def forward_with_routing(self, x):
        x_arr = np.asarray(x, dtype=np.float32)
        y = self.forward(x_arr)
        probs = self.gate.dense_probs(x_arr)
        return y, probs


class SparseFuseMoE:
    """3-layer sparse FuseMoE backbone (as used in sparse setting)."""

    def __init__(self, cfg: SparseMoEConfig):
        self.cfg = cfg
        self.layers = [SparseMoELayer(cfg) for _ in range(cfg.n_layers)]

    def forward(self, x):
        out = np.asarray(x, dtype=np.float32)
        for layer in self.layers:
            out = out + layer.forward(out)
        return out.astype(np.float32)

    def forward_with_routing(self, x):
        out = np.asarray(x, dtype=np.float32)
        probs_per_layer = []
        for layer in self.layers:
            y, probs = layer.forward_with_routing(out)
            out = out + y
            probs_per_layer.append(probs)
        return out.astype(np.float32), probs_per_layer
