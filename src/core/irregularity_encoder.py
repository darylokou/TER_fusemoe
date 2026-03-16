"""FuseMoE irregularity encoder pseudocode.

Implements Unified Temporal Discretization Embedding (UTDE):
1) mTAND embedding over irregular timestamps.
2) Imputation-based discretization over gamma temporal bins.
3) Learnable gate to unify both representations.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
  x_max = np.max(x, axis=axis, keepdims=True)
  e = np.exp(x - x_max)
  return e / np.clip(np.sum(e, axis=axis, keepdims=True), 1e-8, None)


def _sigmoid(x: np.ndarray) -> np.ndarray:
  return 1.0 / (1.0 + np.exp(-x))


@dataclass
class UTDEConfig:
    gamma_bins: int = 48
    mtand_heads: int = 8
    time_embed_functions: int = 16  # H
    attn_embed_dim: int = 128


class MTANDTimeEmbedding:
    """mTAND time encoder with periodic + linear features.

    phi_h(tau) for h in [1..H] may include sin/cos bases and linear projections.
    """

    def __init__(self, cfg: UTDEConfig):
      self.cfg = cfg
      self.H = max(2, cfg.time_embed_functions)
      self.periodic_dim = self.H // 2
      self.linear_dim = self.H - self.periodic_dim

      # Deterministic basis parameters.
      self.omega = np.linspace(0.1, 2.0, self.periodic_dim, dtype=np.float32)
      self.alpha = np.linspace(0.1, 1.0, self.linear_dim, dtype=np.float32)

      rng = np.random.default_rng(19)
      self.proj = rng.standard_normal((self.H, cfg.attn_embed_dim), dtype=np.float32) / np.sqrt(self.H)

    def _to_2d_time(self, tau) -> np.ndarray:
      tau_arr = np.asarray(tau, dtype=np.float32)
      if tau_arr.ndim == 1:
        tau_arr = tau_arr[None, :]
      if tau_arr.ndim != 2:
        raise ValueError("tau must have shape [N_obs] or [B, N_obs]")
      return tau_arr

    def phi(self, tau):
        """Return H-dimensional time features for scalar tau.

        Example basis:
          [sin(omega_i * tau), cos(omega_i * tau), alpha_j * tau]
        """
        tau_arr = self._to_2d_time(tau)

        periodic = np.sin(tau_arr[..., None] * self.omega[None, None, :])
        if self.linear_dim > 0:
          linear = tau_arr[..., None] * self.alpha[None, None, :]
          feats = np.concatenate([periodic, linear], axis=-1)
        else:
          feats = periodic

        return feats.astype(np.float32)

    def project(self, tau) -> np.ndarray:
        feats = self.phi(tau)
        return (feats @ self.proj).astype(np.float32)


class UnifiedTemporalDiscretizationEmbedding:
    """UTDE module.

    Inputs:
      values: [B, N_obs, D]
      times: [B, N_obs]
      mask: [B, N_obs, D]

    Outputs:
      E_utde: [B, gamma, attn_embed_dim]
    """

    def __init__(self, cfg: UTDEConfig):
      self.cfg = cfg
      self.mtand = MTANDTimeEmbedding(cfg)
      self.query_time = np.linspace(0.0, 1.0, cfg.gamma_bins, dtype=np.float32)
      self.query = self.mtand.project(self.query_time[None, :])[0]

      self.value_proj: Optional[np.ndarray] = None
      self.gate_w: Optional[np.ndarray] = None
      self.gate_b: Optional[np.ndarray] = None

      self.global_mean_: Optional[np.ndarray] = None

    def _ensure_params(self, input_dim: int) -> None:
      if self.value_proj is None or self.value_proj.shape[0] != input_dim:
        rng = np.random.default_rng(23)
        self.value_proj = rng.standard_normal(
          (input_dim, self.cfg.attn_embed_dim), dtype=np.float32
        ) / np.sqrt(max(1, input_dim))

      gate_in = 2 * self.cfg.attn_embed_dim
      if self.gate_w is None:
        rng = np.random.default_rng(29)
        self.gate_w = rng.standard_normal((gate_in, self.cfg.attn_embed_dim), dtype=np.float32) / np.sqrt(
          gate_in
        )
        self.gate_b = np.zeros((self.cfg.attn_embed_dim,), dtype=np.float32)

    def fit_global_mean(self, values, mask) -> np.ndarray:
      vals = np.asarray(values, dtype=np.float32)
      m = np.asarray(mask, dtype=np.float32)
      if vals.ndim != 3 or m.ndim != 3:
        raise ValueError("values and mask must be [B, N_obs, D]")
      denom = np.clip(m.sum(axis=(0, 1)), 1e-8, None)
      self.global_mean_ = (vals * m).sum(axis=(0, 1)) / denom
      return self.global_mean_.astype(np.float32)

    def _compute_bins(self, times: np.ndarray) -> np.ndarray:
      # Per-batch bins preserve each sample's observed time span.
      b, _ = times.shape
      bins = np.zeros((b, self.cfg.gamma_bins), dtype=np.float32)
      for i in range(b):
        t_i = times[i]
        valid = np.isfinite(t_i)
        if not np.any(valid):
          bins[i] = np.linspace(0.0, 1.0, self.cfg.gamma_bins, dtype=np.float32)
        else:
          t_min = float(np.min(t_i[valid]))
          t_max = float(np.max(t_i[valid]))
          if abs(t_max - t_min) < 1e-8:
            bins[i] = t_min
          else:
            bins[i] = np.linspace(t_min, t_max, self.cfg.gamma_bins, dtype=np.float32)
      return bins

    def imputation_discretization(self, values, times, mask):
        """Build E_Imp on regular bins with carry-forward + global-mean fallback.

        For each bin and feature:
          - choose closest previous observed value.
          - if no previous value exists, use global mean (computed on train split).
        """
        vals = np.asarray(values, dtype=np.float32)
        t = np.asarray(times, dtype=np.float32)
        m = np.asarray(mask, dtype=np.float32)

        if vals.ndim != 3 or t.ndim != 2 or m.ndim != 3:
          raise ValueError("values/times/mask shapes must be [B,N,D], [B,N], [B,N,D]")
        b, _, d = vals.shape
        self._ensure_params(d)

        if self.global_mean_ is None or self.global_mean_.shape[0] != d:
          self.fit_global_mean(vals, m)

        bins = self._compute_bins(t)
        e_imp = np.zeros((b, self.cfg.gamma_bins, d), dtype=np.float32)

        for bi in range(b):
          for di in range(d):
            obs_idx = np.where((m[bi, :, di] > 0.0) & np.isfinite(t[bi]))[0]
            if len(obs_idx) == 0:
              e_imp[bi, :, di] = self.global_mean_[di]
              continue

            obs_times = t[bi, obs_idx]
            obs_vals = vals[bi, obs_idx, di]
            order = np.argsort(obs_times)
            obs_times = obs_times[order]
            obs_vals = obs_vals[order]

            for gi, tau_g in enumerate(bins[bi]):
              pos = np.searchsorted(obs_times, tau_g, side="right") - 1
              if pos >= 0:
                e_imp[bi, gi, di] = obs_vals[pos]
              else:
                e_imp[bi, gi, di] = self.global_mean_[di]

        return (e_imp @ self.value_proj).astype(np.float32)

    def mtand_discretization(self, values, times, mask):
        """Build E_mTAND using attention from regular bins to irregular observations.

        Query = regular bins gamma; Keys/Values = irregular timestamps & measurements.
        """
        vals = np.asarray(values, dtype=np.float32)
        t = np.asarray(times, dtype=np.float32)
        m = np.asarray(mask, dtype=np.float32)

        if vals.ndim != 3 or t.ndim != 2 or m.ndim != 3:
          raise ValueError("values/times/mask shapes must be [B,N,D], [B,N], [B,N,D]")
        b, _, d = vals.shape
        self._ensure_params(d)

        k = self.mtand.project(t)
        v = vals @ self.value_proj

        obs_valid = (m.sum(axis=-1) > 0).astype(np.float32)
        obs_valid *= np.isfinite(t).astype(np.float32)

        q = np.broadcast_to(self.query[None, :, :], (b, self.cfg.gamma_bins, self.cfg.attn_embed_dim))

        logits = np.matmul(q, np.swapaxes(k, 1, 2)) / np.sqrt(self.cfg.attn_embed_dim)
        logits = np.where(obs_valid[:, None, :] > 0, logits, -1e9)

        attn = _softmax(logits, axis=-1)
        attn = attn * obs_valid[:, None, :]
        attn = attn / np.clip(attn.sum(axis=-1, keepdims=True), 1e-8, None)

        e_mtand = np.matmul(attn, v)
        return e_mtand.astype(np.float32)

    def forward(self, values, times, mask):
        """Fuse E_Imp and E_mTAND with learnable gate g.

        g = f(E_Imp ⊕ E_mTAND), where f is an MLP + sigmoid.
        E_UTDE = g ⊙ E_Imp + (1-g) ⊙ E_mTAND
        """
        e_imp = self.imputation_discretization(values, times, mask)
        e_mtand = self.mtand_discretization(values, times, mask)

        gate_in = np.concatenate([e_imp, e_mtand], axis=-1)
        g = _sigmoid(gate_in @ self.gate_w + self.gate_b)
        e_utde = g * e_imp + (1.0 - g) * e_mtand
        return e_utde.astype(np.float32)
