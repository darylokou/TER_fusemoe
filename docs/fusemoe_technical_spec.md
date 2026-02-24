# FuseMoE Technical Specification (Sparse-Only, NeurIPS 2024-aligned)

## 1) Environment and Project Structure

> **Scope decision:** This repository follows the **sparse FuseMoE setting only** (Top-K sparse dispatch with Laplace gating), matching the public FuseMoE implementation direction. Dense/full-expert routing is intentionally excluded from this spec.


- Conda environment:
  - `name: MulEHR`
  - `python=3.8`
- Minimal Python dependencies:
  - `pytorch`, `torchvision`, `torchaudio`
  - `transformers`
  - `numpy`, `pandas`, `scikit-learn`

Proposed source layout:

```text
src/
  core/
    irregularity_encoder.py      # UTDE + mTAND + imputation + gating
    moe_fusion.py                # Laplace gate, experts, stacked MoE
    routers.py                   # Joint / Modality-Specific / Disjoint routers
    losses.py                    # missing-indicator Z + entropy regularization
  preprocessing/
    mimic_iv_pipeline.py         # MIMIC-IV multimodal feature extraction
```

## 2) Data Preprocessing Logic (MIMIC-IV)

### 2.1 Vitals/Labs (30 events)

- Select 30 predefined physiological/lab event types.
- Compute train-split statistics per event:
  - `mu_e = mean(train_values_e)`
  - `sigma_e = std(train_values_e)`
- Standardize each observed value:
  - `x'_e = (x_e - mu_e) / (sigma_e + eps)`
- Preserve irregular timestamps and a feature-wise observation mask.

### 2.2 CXR modality

- Use pre-trained DenseNet-121 as frozen/finetunable encoder.
- Remove classification head and extract penultimate pooled output.
- Output embedding dimension is `1024`.

### 2.3 Clinical text modality

- Use BioClinicalBERT tokenizer/model.
- Tokenize note text (`max_length=512`, truncation).
- Extract `768`-dim embedding (CLS token or masked mean pooling).

### 2.4 ECG modality

- Input per example: `4096 x 12` waveform.
- Train convolutional autoencoder with 6 encoder blocks:
  - Temporal Conv1D -> BatchNorm -> Dropout -> MaxPool
- Flatten and project to latent vector:
  - `z_ecg in R^256`
- Use encoder latent output as downstream ECG feature.

## 3) Irregularity Encoder: UTDE

UTDE unifies two discretizations into `gamma` regular bins.

- Let irregular observations be `(v_k, t_k)`.
- Let regular bin centers be `{tau_g}_{g=1..gamma}`.

### 3.1 mTAND branch

- Build `H` time embedding functions `phi_h(tau)` from periodic + linear basis.
  - Example: `sin(omega_h tau), cos(omega_h tau), alpha_h tau`
- mTAND attention:
  - Queries: regular bins `tau_g`
  - Keys: irregular times `t_k` through time embedding projection
  - Values: projected measurements `v_k`
- Output branch representation `E_mTAND in R^{B x gamma x d}` with `d=128`.

### 3.2 Imputation-based branch

For each feature and bin `tau_g`:
1. Find closest previous observed value at time `t_k <= tau_g`.
2. If none exists, impute with feature global mean from train split.

This yields `E_Imp in R^{B x gamma x d}` after projection to `d=128`.

### 3.3 Gated unification

- Concatenate branch outputs then compute gate:
  - `g = f(E_Imp ⊕ E_mTAND)` where `f` is MLP + sigmoid.
- Fuse point-wise:
  - `E_UTDE = g ⊙ E_Imp + (1-g) ⊙ E_mTAND`

## 4) Sparse MoE Fusion Layer with Laplace Gating

### 4.1 Experts

- Number of experts: `16`
- Each expert `E_i` is FFN:
  - `Linear(128, 512) -> GeLU -> Linear(512, 128)`

### 4.2 Laplace sparse gate (distance-based Top-K)

For an input token representation `x` and learnable expert anchors `W_i`:

- Compute gating logits via negative L2 distance:
  - `logit_i = -||W_i - x||_2`
- Select sparse experts with Top-K over logits:
  - `h_l(x) = TopK(logit_i)`
- Softmax only over selected experts for routing weights; non-selected experts receive zero probability.

### 4.3 Stacking

- MoE depth: `3` layers.
- Residual connection between layers recommended.

## 5) Sparse Router Architectures

### 5.1 Joint router

- Concatenate available modality embeddings into one vector.
- Project to model dimension (`128`) and route through one shared MoE.
- Top-K = `4`.

### 5.2 Modality-specific router (shared experts)

- Each modality has its own router/gating pathway.
- All routers dispatch into the same shared pool of 16 experts.
- Top-K = `4` per modality.

### 5.3 Disjoint router

- Dedicated expert pool per modality (no sharing).
- Route each modality independently into its own MoE.
- Top-K = `2`.

## 6) Missingness Handling

### 6.1 Learnable missing indicator embedding `Z`

- Maintain one learnable embedding vector `Z_m` per modality.
- If modality `m` is missing in sample `b`, replace/augment modality token:
  - `x_{b,m} <- x_{b,m} + Z_m`

### 6.2 Entropy regularization `E(x)` (sparse routing)

- Let `p_{b,m}` be sparse router probability distribution over experts (non-Top-K entries are zero).
- For missing modalities only, encourage high entropy to avoid overusing dominant experts:
  - `H(p_{b,m}) = -sum_i p_{b,m,i} log p_{b,m,i}`
  - `E(x) = - mean_{(b,m):missing} H(p_{b,m})`

Equivalent formulation: minimize KL divergence to uniform distribution.

### 6.3 Joint objective

- Final training objective:
  - `L_total = L_task + lambda_E * E(x)`

## 7) Hyperparameters from source constraints

- Experts: `16`
- Top-K: `4` (Joint + Modality-Specific), `2` (Disjoint)
- MoE layers: `3`
- Attention embedding dimension: `128`
- Expert FFN hidden size: `512`

## 8) Python-style Pseudocode (high level)

```python
# preprocess
batch = MIMICIVPipeline().build(raw_batch)
# batch: {
#   "vitals_irregular": (values, times, mask),
#   "cxr": cxr_1024,
#   "text": text_768,
#   "ecg": ecg_256,
#   "modality_present": presence_mask,
#   "labels": y
# }

# irregular vitals/labs -> regular token sequence
utde = UnifiedTemporalDiscretizationEmbedding(gamma_bins=G, attn_embed_dim=128, H=16)
vitals_token = utde(values, times, mask)          # [B, G, 128]
vitals_token = temporal_pool(vitals_token)         # [B, 128]

# align modalities to model dim
x = {
  "vitals": project(vitals_token, 128),
  "cxr": project(batch["cxr"], 128),
  "text": project(batch["text"], 128),
  "ecg": project(batch["ecg"], 128),
}

# add missing indicator embeddings
for m in ["vitals", "cxr", "text", "ecg"]:
    x[m] = apply_missing_embedding(x[m], batch["modality_present"][m], Z[m])

# choose router design
if router_type == "joint":
    fused, router_probs = joint_router(x, top_k=4)
elif router_type == "modality_specific":
    fused, router_probs = modality_specific_router(x, top_k=4)
else:  # disjoint
    fused, router_probs = disjoint_router(x, top_k=2)

# prediction head
logits = classifier(fused)
L_task = task_criterion(logits, batch["labels"])

# missingness-aware entropy regularization
E = entropy_regularization(router_probs, batch["modality_present"])
L_total = L_task + lambda_E * E

L_total.backward()
optimizer.step()
```
