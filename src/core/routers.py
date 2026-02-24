"""Sparse router variants for FuseMoE.

All router variants in this file use sparse MoE dispatch only.
"""


class JointSparseRouter:
    """Single sparse router over concatenated multi-modal representation."""

    def __init__(self, sparse_moe):
        self.sparse_moe = sparse_moe

    def forward(self, modality_dict):
        # x = concat([modality_dict[m] for m in ['vitals', 'cxr', 'text', 'ecg']], dim=-1)
        # x = project_to_model_dim(x)
        # return self.sparse_moe(x)  # internally Top-K sparse routing
        raise NotImplementedError


class ModalitySpecificSparseRouter:
    """One sparse router path per modality with shared sparse experts."""

    def __init__(self, shared_sparse_moe, modality_names):
        self.shared_sparse_moe = shared_sparse_moe
        self.modality_names = modality_names

    def forward(self, modality_dict):
        # outputs = {}
        # for m in self.modality_names:
        #   x_m = project_to_model_dim(modality_dict[m])
        #   outputs[m] = self.shared_sparse_moe(x_m)
        # return fuse_modal_outputs(outputs)
        raise NotImplementedError


class DisjointSparseRouter:
    """Dedicated sparse expert pool per modality (disjoint experts)."""

    def __init__(self, modality_sparse_moe):
        # modality_sparse_moe: dict[str, SparseFuseMoE]
        self.modality_sparse_moe = modality_sparse_moe

    def forward(self, modality_dict):
        # outputs = {}
        # for m, moe in self.modality_sparse_moe.items():
        #   x_m = project_to_model_dim(modality_dict[m])
        #   outputs[m] = moe(x_m)
        # return fuse_modal_outputs(outputs)
        raise NotImplementedError
