"""MIMIC-IV preprocessing pseudocode for FuseMoE.

Modalities:
1) Vitals/Labs (30 events): standardize to mean=0, std=1.
2) CXR: 1024-d embeddings from pre-trained DenseNet-121.
3) Text: 768-d embeddings from BioClinicalBERT.
4) ECG: conv autoencoder (6 temporal blocks) -> 256-d latent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.preprocessing.pipeline import PreprocessingPipeline, MaskGenerator, Normalize
from src.preprocessing.adapters import ClinGenAdapter

ArrayTriplet = Tuple[np.ndarray, np.ndarray, np.ndarray]


def _as_numpy(x: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(denom, eps, None)


class MIMICIVPipeline:
    """End-to-end preprocessing with modular ClinGen-style pipeline support."""

    MODALITIES = ("vitals", "cxr", "text", "ecg")

    def __init__(
        self,
        vitals_extractor=None,
        cxr_extractor=None,
        text_extractor=None,
        ecg_encoder=None,
        pipeline: Optional[PreprocessingPipeline] = None,
        adapter: Optional[Any] = None,
    ):
        self.vl = vitals_extractor
        self.cxr = cxr_extractor
        self.text = text_extractor
        self.ecg = ecg_encoder

        self.pipeline = pipeline
        self.adapter = adapter if adapter is not None else ClinGenAdapter()

    def _coerce_id_list(self, ids: Optional[Sequence[Any]], n: int) -> Optional[List[Any]]:
        if ids is None:
            return None
        out = list(ids)
        if len(out) != n:
            raise ValueError("Provided IDs length does not match modality batch size")
        return out

    def build(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(batch, dict):
            raise TypeError("batch must be a dictionary")

        # --- Extract / encode modalities ---
        vitals_triplet = batch.get("vitals_irregular", None)
        vitals_ids = self._coerce_id_list(batch.get("vitals_ids"), vitals_triplet[0].shape[0]) if vitals_triplet is not None else None

        cxr_features = self.cxr.transform(batch.get("cxr_images", np.zeros((0, 224, 224), dtype=np.float32)))
        cxr_ids = self._coerce_id_list(batch.get("cxr_ids"), cxr_features.shape[0])

        text_features = self.text.transform(batch.get("note_texts", []))
        text_ids = self._coerce_id_list(batch.get("text_ids"), text_features.shape[0])

        ecg_features = self.ecg.encode(batch.get("ecg_waveforms", np.zeros((0, 4096, 12), dtype=np.float32)))
        ecg_ids = self._coerce_id_list(batch.get("ecg_ids"), ecg_features.shape[0])

        # --- Determine unified target IDs ---
        id_lists = {
            "vitals": vitals_ids,
            "cxr": cxr_ids,
            "text": text_ids,
            "ecg": ecg_ids,
        }
        size_hints = {
            "vitals": vitals_triplet[0].shape[0] if vitals_triplet is not None else 0,
            "cxr": cxr_features.shape[0],
            "text": text_features.shape[0],
            "ecg": ecg_features.shape[0],
        }
        target_ids = next((ids for ids in id_lists.values() if ids is not None), list(range(max(size_hints.values(), default=0))))

        # --- Align features and compute modality presence ---
        def align_dense(features, src_ids, dim):
            n = len(target_ids)
            out = np.zeros((n, dim), dtype=np.float32)
            present = np.zeros(n, dtype=np.float32)
            if features is None or len(features) == 0:
                return out, present
            if src_ids is None:
                count = min(n, features.shape[0])
                out[:count] = features[:count]
                present[:count] = 1.0
            else:
                id_map = {k: i for i, k in enumerate(src_ids)}
                for i, tid in enumerate(target_ids):
                    j = id_map.get(tid)
                    if j is not None:
                        out[i] = features[j]
                        present[i] = 1.0
            return out, present

        aligned_cxr, present_cxr = align_dense(cxr_features, cxr_ids, cxr_features.shape[1] if len(cxr_features) > 0 else 0)
        aligned_text, present_text = align_dense(text_features, text_ids, text_features.shape[1] if len(text_features) > 0 else 0)
        aligned_ecg, present_ecg = align_dense(ecg_features, ecg_ids, ecg_features.shape[1] if len(ecg_features) > 0 else 0)

        # Vitals irregular sequence handled separately
        aligned_vitals = vitals_triplet if vitals_triplet is not None else (np.zeros((0, 0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0, 0), dtype=np.float32))
        present_vitals = np.ones(len(target_ids), dtype=np.float32) if vitals_triplet is not None else np.zeros(len(target_ids), dtype=np.float32)

        modality_present = np.stack([present_vitals, present_cxr, present_text, present_ecg], axis=1).astype(np.float32)

        labels = batch.get("labels")
        if labels is not None:
            labels = _as_numpy(labels)
            if labels.ndim == 0:
                labels = labels.reshape(1)

        # --- Create standardized sample ---
        sample = {
            "modalities": {
                "vitals": aligned_vitals,
                "cxr": aligned_cxr,
                "text": aligned_text,
                "ecg": aligned_ecg,
            },
            "mask": modality_present,
            "target": labels,
            "metadata": {"ids": target_ids},
        }

        # --- Apply preprocessing pipeline ---
        if self.pipeline is not None:
            sample = self.pipeline(sample)

        # --- Adapt to model-specific format ---
        if self.adapter is not None:
            return self.adapter(sample)
        return sample
