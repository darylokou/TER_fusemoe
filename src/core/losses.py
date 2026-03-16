"""Losses for sparse FuseMoE including missingness-aware regularization."""

import numpy as np


class MissingnessEntropyRegularizer:
    """Learnable missing indicator embedding Z and sparse-entropy loss E(x)."""

    def __init__(self, n_modalities, model_dim, n_experts):
        # Z[m] is learned embedding added to missing modality representation.
        # self.Z = Parameter([n_modalities, model_dim])
        self.n_modalities = int(n_modalities)
        self.model_dim = int(model_dim)
        self.n_experts = n_experts

        rng = np.random.default_rng(401)
        self.Z = rng.standard_normal((self.n_modalities, self.model_dim), dtype=np.float32) * 0.02

    def apply_missing_embedding(self, modality_x, modality_present_mask, modality_id):
        # if modality_present_mask == 0:
        #   return modality_x + Z[modality_id]
        # return modality_x
        x = np.asarray(modality_x, dtype=np.float32)
        present = np.asarray(modality_present_mask, dtype=np.float32)
        m_id = int(modality_id)

        if m_id < 0 or m_id >= self.n_modalities:
            raise ValueError("modality_id out of range")
        if x.ndim != 2:
            raise ValueError("modality_x must be [B, D]")
        if x.shape[1] != self.model_dim:
            raise ValueError("modality_x feature dimension must equal model_dim")

        if present.ndim == 1:
            present = present[:, None]
        if present.shape[0] != x.shape[0]:
            raise ValueError("modality_present_mask batch size mismatch")

        missing = (present <= 0.0).astype(np.float32)
        return (x + missing * self.Z[m_id][None, :]).astype(np.float32)

    def entropy_loss(self, sparse_router_probs, missing_mask):
        """Compute E(x) for sparse router probabilities.

        sparse_router_probs: [B, M, E] where non-topk experts have zero probability.
        missing_mask: [B, M] 1 if modality missing.

        For missing modalities, maximize entropy to avoid collapse onto a few experts:
            E(x) = - mean_{missing} H(p)
        """
        probs = np.asarray(sparse_router_probs, dtype=np.float32)
        miss = np.asarray(missing_mask, dtype=np.float32)

        if probs.ndim != 3:
            raise ValueError("sparse_router_probs must be [B, M, E]")
        if miss.ndim != 2:
            raise ValueError("missing_mask must be [B, M]")
        if probs.shape[:2] != miss.shape:
            raise ValueError("missing_mask shape must match first two prob dims")

        p = probs / np.clip(probs.sum(axis=-1, keepdims=True), 1e-8, None)
        entropy = -np.sum(p * np.log(np.clip(p, 1e-8, None)), axis=-1)

        select = miss > 0.0
        if not np.any(select):
            return np.float32(0.0)

        # E(x) = -mean H(p_missing), matching the paper formulation.
        return np.float32(-entropy[select].mean())


def sparse_fusemoe_joint_loss(task_loss, entropy_reg, lambda_entropy=1.0):
    """Total objective for sparse FuseMoE: Task Loss + E(x)."""
    return task_loss + lambda_entropy * entropy_reg
