"""Losses for sparse FuseMoE including missingness-aware regularization."""


class MissingnessEntropyRegularizer:
    """Learnable missing indicator embedding Z and sparse-entropy loss E(x)."""

    def __init__(self, n_modalities, model_dim, n_experts):
        # Z[m] is learned embedding added to missing modality representation.
        # self.Z = Parameter([n_modalities, model_dim])
        self.n_experts = n_experts

    def apply_missing_embedding(self, modality_x, modality_present_mask, modality_id):
        # if modality_present_mask == 0:
        #   return modality_x + Z[modality_id]
        # return modality_x
        raise NotImplementedError

    def entropy_loss(self, sparse_router_probs, missing_mask):
        """Compute E(x) for sparse router probabilities.

        sparse_router_probs: [B, M, E] where non-topk experts have zero probability.
        missing_mask: [B, M] 1 if modality missing.

        For missing modalities, maximize entropy to avoid collapse onto a few experts:
            E(x) = - mean_{missing} H(p)
        """
        # p_miss = sparse_router_probs[missing_mask == 1]
        # entropy = -sum(p_miss * log(p_miss + 1e-8), dim=-1)
        # return -entropy.mean()
        raise NotImplementedError


def sparse_fusemoe_joint_loss(task_loss, entropy_reg, lambda_entropy=1.0):
    """Total objective for sparse FuseMoE: Task Loss + E(x)."""
    # return task_loss + lambda_entropy * entropy_reg
    raise NotImplementedError
