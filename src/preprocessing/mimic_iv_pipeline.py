"""MIMIC-IV preprocessing pseudocode for FuseMoE.

Modalities:
1) Vitals/Labs (30 events): standardize to mean=0, std=1.
2) CXR: 1024-d embeddings from pre-trained DenseNet-121.
3) Text: 768-d embeddings from BioClinicalBERT.
4) ECG: conv autoencoder (6 temporal blocks) -> 256-d latent.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


ArrayTriplet = Tuple[np.ndarray, np.ndarray, np.ndarray]


def _as_numpy(x: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def _stable_token_hash(token: str) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest, 16)


def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(denom, eps, None)


class VitalsLabsExtractor:
    """Extract and standardize 30 selected events."""

    def __init__(
        self,
        selected_events: Optional[Sequence[str]] = None,
        patient_id_col: str = "patient_id",
        episode_id_col: str = "episode_id",
        event_col: str = "event",
        time_col: str = "time",
        value_col: str = "value",
        eps: float = 1e-8,
        max_seq_len: Optional[int] = None,
    ):
        self.selected_events = list(selected_events) if selected_events is not None else None
        self.patient_id_col = patient_id_col
        self.episode_id_col = episode_id_col
        self.event_col = event_col
        self.time_col = time_col
        self.value_col = value_col
        self.eps = eps
        self.max_seq_len = max_seq_len

        self.events_: List[str] = []
        self.event_to_idx_: Dict[str, int] = {}
        self.mu_: Dict[str, float] = {}
        self.sigma_: Dict[str, float] = {}

    def _check_pandas(self):
        if pd is None:
            raise ImportError("pandas is required for VitalsLabsExtractor DataFrame mode")

    def _fit_long_format(self, train_df) -> None:
        if self.selected_events is None:
            events = train_df[self.event_col].dropna().astype(str).unique().tolist()
            self.events_ = sorted(events)[:30]
        else:
            self.events_ = [str(e) for e in self.selected_events]

        self.event_to_idx_ = {e: i for i, e in enumerate(self.events_)}
        for event in self.events_:
            vals = train_df.loc[train_df[self.event_col].astype(str) == event, self.value_col]
            vals = vals.astype(float)
            mu = float(vals.mean()) if len(vals) > 0 else 0.0
            sigma = float(vals.std(ddof=0)) if len(vals) > 0 else 1.0
            self.mu_[event] = mu
            self.sigma_[event] = sigma if sigma > self.eps else 1.0

    def _fit_wide_format(self, train_df) -> None:
        if self.selected_events is None:
            numeric_cols = [
                c
                for c in train_df.columns
                if c
                not in {
                    self.patient_id_col,
                    self.episode_id_col,
                    self.time_col,
                }
                and np.issubdtype(train_df[c].dtype, np.number)
            ]
            self.events_ = numeric_cols[:30]
        else:
            self.events_ = [e for e in self.selected_events if e in train_df.columns]

        self.event_to_idx_ = {e: i for i, e in enumerate(self.events_)}
        for event in self.events_:
            vals = train_df[event].astype(float)
            mu = float(vals.mean()) if len(vals) > 0 else 0.0
            sigma = float(vals.std(ddof=0)) if len(vals) > 0 else 1.0
            self.mu_[event] = mu
            self.sigma_[event] = sigma if sigma > self.eps else 1.0

    def fit(self, train_df):
        # self.mu[event], self.sigma[event] from train split only
        self._check_pandas()

        required_long = {
            self.patient_id_col,
            self.episode_id_col,
            self.event_col,
            self.time_col,
            self.value_col,
        }
        if required_long.issubset(set(train_df.columns)):
            self._fit_long_format(train_df)
        else:
            self._fit_wide_format(train_df)

        if not self.events_:
            raise ValueError("No event columns discovered during fit")
        return self

    def _ensure_fitted(self):
        if not self.events_:
            raise RuntimeError("VitalsLabsExtractor must be fitted before transform")

    def _standardize(self, value: float, event: str) -> float:
        mu = self.mu_.get(event, 0.0)
        sigma = self.sigma_.get(event, 1.0)
        return (float(value) - mu) / (sigma + self.eps)

    def _transform_long_format(self, df) -> Tuple[ArrayTriplet, List[Tuple[Any, Any]]]:
        groups = df.groupby([self.patient_id_col, self.episode_id_col], dropna=False)
        group_items: List[Tuple[Tuple[Any, Any], np.ndarray, np.ndarray, np.ndarray]] = []
        max_t = 0

        for group_key, gdf in groups:
            gdf = gdf[gdf[self.event_col].astype(str).isin(self.event_to_idx_)]
            if len(gdf) == 0:
                continue

            gdf = gdf.sort_values(self.time_col)
            unique_times = sorted(gdf[self.time_col].astype(float).unique().tolist())
            t_len = len(unique_times)
            max_t = max(max_t, t_len)

            time_to_idx = {t: i for i, t in enumerate(unique_times)}
            values = np.zeros((t_len, len(self.events_)), dtype=np.float32)
            mask = np.zeros((t_len, len(self.events_)), dtype=np.float32)

            for row in gdf.itertuples(index=False):
                event = str(getattr(row, self.event_col))
                if event not in self.event_to_idx_:
                    continue
                t_val = float(getattr(row, self.time_col))
                x_val = float(getattr(row, self.value_col))
                ti = time_to_idx[t_val]
                ei = self.event_to_idx_[event]
                values[ti, ei] = self._standardize(x_val, event)
                mask[ti, ei] = 1.0

            times = np.array(unique_times, dtype=np.float32)
            group_items.append((group_key, values, times, mask))

        if len(group_items) == 0:
            empty = (
                np.zeros((0, 0, len(self.events_)), dtype=np.float32),
                np.zeros((0, 0), dtype=np.float32),
                np.zeros((0, 0, len(self.events_)), dtype=np.float32),
            )
            return empty, []

        if self.max_seq_len is not None:
            max_t = min(max_t, self.max_seq_len)

        bsz = len(group_items)
        d = len(self.events_)
        out_values = np.zeros((bsz, max_t, d), dtype=np.float32)
        out_times = np.full((bsz, max_t), np.nan, dtype=np.float32)
        out_mask = np.zeros((bsz, max_t, d), dtype=np.float32)
        ids: List[Tuple[Any, Any]] = []

        for i, (group_key, values, times, mask) in enumerate(group_items):
            seq_len = min(values.shape[0], max_t)
            out_values[i, :seq_len, :] = values[:seq_len, :]
            out_times[i, :seq_len] = times[:seq_len]
            out_mask[i, :seq_len, :] = mask[:seq_len, :]
            ids.append(group_key)

        return (out_values, out_times, out_mask), ids

    def _transform_wide_format(self, df) -> Tuple[ArrayTriplet, List[Tuple[Any, Any]]]:
        req_cols = {self.patient_id_col, self.episode_id_col}
        has_ids = req_cols.issubset(df.columns)
        has_time = self.time_col in df.columns

        if not has_ids:
            df = df.copy()
            df[self.patient_id_col] = np.arange(len(df))
            df[self.episode_id_col] = 0
        if not has_time:
            df = df.copy()
            df[self.time_col] = 0.0

        groups = df.groupby([self.patient_id_col, self.episode_id_col], dropna=False)
        group_items: List[Tuple[Tuple[Any, Any], np.ndarray, np.ndarray, np.ndarray]] = []
        max_t = 0

        for group_key, gdf in groups:
            gdf = gdf.sort_values(self.time_col)
            t_len = len(gdf)
            max_t = max(max_t, t_len)

            values = np.zeros((t_len, len(self.events_)), dtype=np.float32)
            mask = np.zeros((t_len, len(self.events_)), dtype=np.float32)

            for e, ei in self.event_to_idx_.items():
                if e not in gdf.columns:
                    continue
                col = gdf[e].astype(float).to_numpy()
                obs = np.isfinite(col)
                if obs.any():
                    values[obs, ei] = (col[obs] - self.mu_[e]) / (self.sigma_[e] + self.eps)
                    mask[obs, ei] = 1.0

            times = gdf[self.time_col].astype(float).to_numpy(dtype=np.float32)
            group_items.append((group_key, values, times, mask))

        if len(group_items) == 0:
            empty = (
                np.zeros((0, 0, len(self.events_)), dtype=np.float32),
                np.zeros((0, 0), dtype=np.float32),
                np.zeros((0, 0, len(self.events_)), dtype=np.float32),
            )
            return empty, []

        if self.max_seq_len is not None:
            max_t = min(max_t, self.max_seq_len)

        bsz = len(group_items)
        d = len(self.events_)
        out_values = np.zeros((bsz, max_t, d), dtype=np.float32)
        out_times = np.full((bsz, max_t), np.nan, dtype=np.float32)
        out_mask = np.zeros((bsz, max_t, d), dtype=np.float32)
        ids: List[Tuple[Any, Any]] = []

        for i, (group_key, values, times, mask) in enumerate(group_items):
            seq_len = min(values.shape[0], max_t)
            out_values[i, :seq_len, :] = values[:seq_len, :]
            out_times[i, :seq_len] = times[:seq_len]
            out_mask[i, :seq_len, :] = mask[:seq_len, :]
            ids.append(group_key)

        return (out_values, out_times, out_mask), ids

    def transform(self, df):
        # x = (x - mu) / (sigma + 1e-8)
        # return irregular sequence: values, times, mask
        self._check_pandas()
        self._ensure_fitted()

        required_long = {
            self.patient_id_col,
            self.episode_id_col,
            self.event_col,
            self.time_col,
            self.value_col,
        }
        if required_long.issubset(set(df.columns)):
            return self._transform_long_format(df)
        return self._transform_wide_format(df)


class CXRExtractor:
    """DenseNet-121 image encoder producing 1024-d vectors."""

    def __init__(self, out_dim: int = 1024, downsample_size: int = 32, seed: int = 7):
        # model = torchvision.models.densenet121(pretrained=True)
        # remove classifier head, global pooling output -> 1024
        self.out_dim = out_dim
        self.downsample_size = downsample_size
        rng = np.random.default_rng(seed)
        in_dim = 3 * downsample_size * downsample_size
        self.proj = rng.standard_normal((in_dim, out_dim), dtype=np.float32) / np.sqrt(in_dim)

    def _to_chw(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        if img.ndim != 3:
            raise ValueError("Each CXR image must be HxW or HxWxC")

        if img.shape[-1] in (1, 3):
            img = np.transpose(img, (2, 0, 1))
        elif img.shape[0] not in (1, 3):
            raise ValueError("Invalid CXR image shape")

        if img.shape[0] == 1:
            img = np.repeat(img, 3, axis=0)
        return img.astype(np.float32)

    def _center_crop_square(self, chw: np.ndarray) -> np.ndarray:
        _, h, w = chw.shape
        side = min(h, w)
        h0 = (h - side) // 2
        w0 = (w - side) // 2
        return chw[:, h0 : h0 + side, w0 : w0 + side]

    def _resize_nearest(self, chw: np.ndarray, size: int) -> np.ndarray:
        _, h, w = chw.shape
        y_idx = np.clip(np.round(np.linspace(0, h - 1, size)).astype(int), 0, h - 1)
        x_idx = np.clip(np.round(np.linspace(0, w - 1, size)).astype(int), 0, w - 1)
        return chw[:, y_idx][:, :, x_idx]

    def _preprocess_one(self, image: np.ndarray) -> np.ndarray:
        chw = self._to_chw(image)
        chw = self._center_crop_square(chw)
        chw = self._resize_nearest(chw, self.downsample_size)

        if chw.max() > 1.0:
            chw = chw / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        chw = (chw - mean) / std
        return chw

    def transform(self, cxr_images):
        # preprocess: resize, normalize, center crop
        # return embeddings [N, 1024]
        images = _as_numpy(cxr_images)
        if images.size == 0:
            return np.zeros((0, self.out_dim), dtype=np.float32)

        if images.ndim == 3:
            images = images[None, ...]

        vectors = []
        for i in range(images.shape[0]):
            chw = self._preprocess_one(images[i])
            flat = chw.reshape(-1)
            vec = flat @ self.proj
            vectors.append(vec.astype(np.float32))

        out = np.stack(vectors, axis=0)
        return _l2_normalize(out).astype(np.float32)


class TextExtractor:
    """BioClinicalBERT text encoder producing 768-d vectors."""

    def __init__(
        self,
        out_dim: int = 768,
        max_length: int = 512,
        use_transformers: bool = False,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    ):
        # tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        # model = AutoModel.from_pretrained(...)
        self.out_dim = out_dim
        self.max_length = max_length
        self.use_transformers = use_transformers
        self.model_name = model_name
        self._token_pattern = re.compile(r"[A-Za-z0-9']+")

        self.tokenizer = None
        self.model = None
        if use_transformers:
            try:
                from transformers import AutoModel, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.model.eval()
            except Exception:
                self.tokenizer = None
                self.model = None

    def _tokenize(self, text: str) -> List[str]:
        return self._token_pattern.findall(text.lower())[: self.max_length]

    def _hashing_embed(self, text: str) -> np.ndarray:
        vec = np.zeros((self.out_dim,), dtype=np.float32)
        tokens = self._tokenize(text)
        if not tokens:
            return vec

        for tok in tokens:
            h = _stable_token_hash(tok)
            idx = h % self.out_dim
            sign = 1.0 if ((h >> 1) & 1) == 0 else -1.0
            vec[idx] += sign

        vec /= np.sqrt(len(tokens))
        return vec

    def _transformers_embed(self, note_texts: Sequence[str]) -> Optional[np.ndarray]:
        if self.tokenizer is None or self.model is None:
            return None

        try:
            import torch
        except Exception:
            return None

        with torch.no_grad():
            tokens = self.tokenizer(
                list(note_texts),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            hidden = self.model(**tokens).last_hidden_state
            attn = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            summed = (hidden * attn).sum(dim=1)
            denom = attn.sum(dim=1).clamp_min(1.0)
            pooled = summed / denom
            out = pooled.cpu().numpy().astype(np.float32)

        if out.shape[1] != self.out_dim:
            if out.shape[1] > self.out_dim:
                out = out[:, : self.out_dim]
            else:
                pad = np.zeros((out.shape[0], self.out_dim - out.shape[1]), dtype=np.float32)
                out = np.concatenate([out, pad], axis=1)
        return out

    def transform(self, note_texts):
        # tokens = tokenizer(..., truncation=True, max_length=512)
        # hidden = model(**tokens).last_hidden_state
        # return cls_pool_or_mean_pool(hidden)  # [N, 768]
        texts = [str(t) for t in note_texts]
        if len(texts) == 0:
            return np.zeros((0, self.out_dim), dtype=np.float32)

        if self.use_transformers:
            embedded = self._transformers_embed(texts)
            if embedded is not None:
                return _l2_normalize(embedded).astype(np.float32)

        out = np.stack([self._hashing_embed(t) for t in texts], axis=0)
        return _l2_normalize(out).astype(np.float32)


class ECGAutoencoder:
    """Temporal conv autoencoder from 4096x12 to 256-d latent."""

    def __init__(self, latent_dim: int = 256, downsample_len: int = 128, seed: int = 11):
        # Encoder: 6 blocks [Conv1D -> BatchNorm -> Dropout -> MaxPool]
        # Flatten + projection to latent_dim=256
        # Decoder mirrors encoder for reconstruction training
        self.latent_dim = latent_dim
        self.downsample_len = downsample_len
        self._proj: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(seed)

    def _ensure_projection(self, in_dim: int) -> None:
        if self._proj is None or self._proj.shape[0] != in_dim:
            self._proj = self._rng.standard_normal((in_dim, self.latent_dim), dtype=np.float32) / np.sqrt(
                in_dim
            )

    def _downsample(self, x: np.ndarray) -> np.ndarray:
        # x: [N, T, C]
        n, t, c = x.shape
        if t == self.downsample_len:
            return x

        if t < self.downsample_len:
            pad_len = self.downsample_len - t
            pad = np.repeat(x[:, -1:, :], pad_len, axis=1)
            return np.concatenate([x, pad], axis=1)

        window = int(np.floor(t / self.downsample_len))
        trimmed = x[:, : window * self.downsample_len, :]
        reshaped = trimmed.reshape(n, self.downsample_len, window, c)
        return reshaped.mean(axis=2)

    def encode(self, ecg_batch):
        # ecg_batch: [N, 4096, 12]
        # return z: [N, 256]
        x = _as_numpy(ecg_batch, dtype=np.float32)
        if x.size == 0:
            return np.zeros((0, self.latent_dim), dtype=np.float32)
        if x.ndim != 3:
            raise ValueError("ecg_batch must have shape [N, T, C]")

        x = np.nan_to_num(x)
        mu = x.mean(axis=1, keepdims=True)
        sigma = x.std(axis=1, keepdims=True) + 1e-8
        x = (x - mu) / sigma

        x_ds = self._downsample(x)
        flat = x_ds.reshape(x_ds.shape[0], -1)
        self._ensure_projection(flat.shape[1])
        z = flat @ self._proj
        return _l2_normalize(z).astype(np.float32)


class MIMICIVPipeline:
    """End-to-end preprocessing and modality alignment."""

    MODALITIES = ("vitals", "cxr", "text", "ecg")

    def __init__(
        self,
        vitals_extractor: Optional[VitalsLabsExtractor] = None,
        cxr_extractor: Optional[CXRExtractor] = None,
        text_extractor: Optional[TextExtractor] = None,
        ecg_encoder: Optional[ECGAutoencoder] = None,
    ):
        self.vl = vitals_extractor if vitals_extractor is not None else VitalsLabsExtractor()
        self.cxr = cxr_extractor if cxr_extractor is not None else CXRExtractor()
        self.text = text_extractor if text_extractor is not None else TextExtractor()
        self.ecg = ecg_encoder if ecg_encoder is not None else ECGAutoencoder()

    def _coerce_id_list(self, ids: Optional[Iterable[Any]], n: int) -> Optional[List[Any]]:
        if ids is None:
            return None
        out = list(ids)
        if len(out) != n:
            raise ValueError("Provided IDs length does not match modality batch size")
        return out

    def _align_dense_features(
        self,
        features: np.ndarray,
        target_ids: List[Any],
        source_ids: Optional[List[Any]],
        feature_dim: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        out = np.zeros((len(target_ids), feature_dim), dtype=np.float32)
        present = np.zeros((len(target_ids),), dtype=np.float32)

        if len(features) == 0:
            return out, present

        if source_ids is None:
            count = min(len(target_ids), features.shape[0])
            out[:count] = features[:count]
            present[:count] = 1.0
            return out, present

        id_to_idx = {k: i for i, k in enumerate(source_ids)}
        for i, tid in enumerate(target_ids):
            j = id_to_idx.get(tid)
            if j is not None:
                out[i] = features[j]
                present[i] = 1.0
        return out, present

    def _align_vitals(
        self,
        vitals_triplet: ArrayTriplet,
        target_ids: List[Any],
        source_ids: Optional[List[Any]],
    ) -> Tuple[ArrayTriplet, np.ndarray]:
        values, times, mask = vitals_triplet
        if values.ndim != 3 or times.ndim != 2 or mask.ndim != 3:
            raise ValueError("Vitals arrays must be [B,T,D], [B,T], [B,T,D]")

        b, t, d = values.shape
        out_values = np.zeros((len(target_ids), t, d), dtype=np.float32)
        out_times = np.full((len(target_ids), t), np.nan, dtype=np.float32)
        out_mask = np.zeros((len(target_ids), t, d), dtype=np.float32)
        present = np.zeros((len(target_ids),), dtype=np.float32)

        if b == 0:
            return (out_values, out_times, out_mask), present

        if source_ids is None:
            count = min(len(target_ids), b)
            out_values[:count] = values[:count]
            out_times[:count] = times[:count]
            out_mask[:count] = mask[:count]
            present[:count] = (mask[:count].sum(axis=(1, 2)) > 0).astype(np.float32)
            return (out_values, out_times, out_mask), present

        id_to_idx = {k: i for i, k in enumerate(source_ids)}
        for i, tid in enumerate(target_ids):
            j = id_to_idx.get(tid)
            if j is not None:
                out_values[i] = values[j]
                out_times[i] = times[j]
                out_mask[i] = mask[j]
                present[i] = 1.0 if mask[j].sum() > 0 else 0.0

        return (out_values, out_times, out_mask), present

    def _determine_target_ids(
        self,
        size_hints: Dict[str, int],
        id_lists: Dict[str, Optional[List[Any]]],
    ) -> List[Any]:
        for modality in self.MODALITIES:
            ids = id_lists.get(modality)
            if ids is not None and len(ids) > 0:
                return list(ids)

        n = max(size_hints.values()) if size_hints else 0
        return list(range(n))

    def _validate_vitals_triplet(self, triplet: Any) -> ArrayTriplet:
        if not isinstance(triplet, (tuple, list)) or len(triplet) != 3:
            raise ValueError("vitals_irregular must be a tuple/list (values, times, mask)")
        values = _as_numpy(triplet[0], dtype=np.float32)
        times = _as_numpy(triplet[1], dtype=np.float32)
        mask = _as_numpy(triplet[2], dtype=np.float32)
        return values, times, mask

    def build(self, batch):
        # 1) extract each modality features
        # 2) align by patient-episode + timestamps
        # 3) create modality presence mask for missingness handling
        # 4) return dict for UTDE + MoE routing
        if not isinstance(batch, dict):
            raise TypeError("batch must be a dictionary")

        vitals_ids: Optional[List[Any]] = None
        if "vitals_irregular" in batch:
            vitals_triplet = self._validate_vitals_triplet(batch["vitals_irregular"])
            vitals_ids = self._coerce_id_list(batch.get("vitals_ids"), vitals_triplet[0].shape[0])
        elif "vitals_df" in batch:
            vitals_triplet, vitals_ids = self.vl.transform(batch["vitals_df"])
        else:
            vitals_triplet = (
                np.zeros((0, 0, 0), dtype=np.float32),
                np.zeros((0, 0), dtype=np.float32),
                np.zeros((0, 0, 0), dtype=np.float32),
            )

        cxr_images = batch.get("cxr_images", np.zeros((0, 224, 224), dtype=np.float32))
        cxr_features = self.cxr.transform(cxr_images)
        cxr_ids = self._coerce_id_list(batch.get("cxr_ids"), cxr_features.shape[0])

        notes = batch.get("note_texts", [])
        text_features = self.text.transform(notes)
        text_ids = self._coerce_id_list(batch.get("text_ids"), text_features.shape[0])

        ecg_wave = batch.get("ecg_waveforms", np.zeros((0, 4096, 12), dtype=np.float32))
        ecg_features = self.ecg.encode(ecg_wave)
        ecg_ids = self._coerce_id_list(batch.get("ecg_ids"), ecg_features.shape[0])

        id_lists: Dict[str, Optional[List[Any]]] = {
            "vitals": vitals_ids,
            "cxr": cxr_ids,
            "text": text_ids,
            "ecg": ecg_ids,
        }
        size_hints = {
            "vitals": vitals_triplet[0].shape[0],
            "cxr": cxr_features.shape[0],
            "text": text_features.shape[0],
            "ecg": ecg_features.shape[0],
        }
        target_ids = self._determine_target_ids(size_hints, id_lists)

        aligned_vitals, present_vitals = self._align_vitals(vitals_triplet, target_ids, vitals_ids)
        aligned_cxr, present_cxr = self._align_dense_features(
            cxr_features, target_ids, cxr_ids, self.cxr.out_dim
        )
        aligned_text, present_text = self._align_dense_features(
            text_features, target_ids, text_ids, self.text.out_dim
        )
        aligned_ecg, present_ecg = self._align_dense_features(
            ecg_features, target_ids, ecg_ids, self.ecg.latent_dim
        )

        modality_present = np.stack(
            [present_vitals, present_cxr, present_text, present_ecg], axis=1
        ).astype(np.float32)

        labels = batch.get("labels")
        if labels is not None:
            labels = _as_numpy(labels)
            if labels.ndim == 0:
                labels = labels.reshape(1)

            if labels.shape[0] != len(target_ids):
                if batch.get("label_ids") is not None:
                    label_ids = self._coerce_id_list(batch.get("label_ids"), labels.shape[0])
                    label_map = {k: labels[i] for i, k in enumerate(label_ids)}
                    aligned_labels = np.zeros((len(target_ids),) + labels.shape[1:], dtype=labels.dtype)
                    for i, tid in enumerate(target_ids):
                        if tid in label_map:
                            aligned_labels[i] = label_map[tid]
                    labels = aligned_labels
                else:
                    raise ValueError("labels length must match aligned batch size or provide label_ids")

        return {
            "ids": target_ids,
            "vitals_irregular": aligned_vitals,
            "cxr": aligned_cxr,
            "text": aligned_text,
            "ecg": aligned_ecg,
            "modality_present": modality_present,
            "labels": labels,
            "modality_names": list(self.MODALITIES),
        }
