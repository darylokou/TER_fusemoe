"""Utilities to adapt the MIMIC-IV demo dataset to the FuseMoE preprocessing pipeline.

This module builds a vitals/labs long-format DataFrame with columns expected by
`VitalsLabsExtractor` and can directly produce a batch ready for `MIMICIVPipeline`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from src.preprocessing.mimic_iv_pipeline import MIMICIVPipeline, VitalsLabsExtractor

PathLike = Union[str, Path]


def _ensure_demo_root(demo_root: PathLike) -> Path:
    root = Path(demo_root)
    if not root.exists():
        raise FileNotFoundError(f"Demo root does not exist: {root}")
    if not (root / "hosp" / "labevents.csv").exists() and not (
        root / "icu" / "chartevents.csv"
    ).exists():
        raise FileNotFoundError(
            "Could not find expected MIMIC-IV demo files under hosp/ or icu/"
        )
    return root


def _load_labevents(root: Path) -> pd.DataFrame:
    labs = pd.read_csv(
        root / "hosp" / "labevents.csv",
        usecols=["subject_id", "hadm_id", "itemid", "charttime", "valuenum"],
    )
    labs = labs.dropna(subset=["hadm_id", "charttime", "valuenum"])

    d_lab = pd.read_csv(root / "hosp" / "d_labitems.csv", usecols=["itemid", "label"])
    labs = labs.merge(d_lab, on="itemid", how="left")
    labs = labs.rename(columns={"label": "event", "valuenum": "value"})
    labs["time"] = pd.to_datetime(labs["charttime"], errors="coerce")

    return labs[["subject_id", "hadm_id", "event", "time", "value"]]


def _load_chartevents(root: Path) -> pd.DataFrame:
    chart = pd.read_csv(
        root / "icu" / "chartevents.csv",
        usecols=["subject_id", "hadm_id", "itemid", "charttime", "valuenum"],
    )
    chart = chart.dropna(subset=["hadm_id", "charttime", "valuenum"])

    d_items = pd.read_csv(root / "icu" / "d_items.csv", usecols=["itemid", "label"])
    chart = chart.merge(d_items, on="itemid", how="left")
    chart = chart.rename(columns={"label": "event", "valuenum": "value"})
    chart["time"] = pd.to_datetime(chart["charttime"], errors="coerce")

    return chart[["subject_id", "hadm_id", "event", "time", "value"]]


def _add_relative_hours(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    adm = pd.read_csv(root / "hosp" / "admissions.csv", usecols=["subject_id", "hadm_id", "admittime"])
    adm["admittime"] = pd.to_datetime(adm["admittime"], errors="coerce")

    merged = df.merge(adm, on=["subject_id", "hadm_id"], how="left")
    merged["time"] = (
        (merged["time"] - pd.to_datetime(merged["admittime"], errors="coerce"))
        .dt.total_seconds()
        .div(3600.0)
    )
    merged = merged.dropna(subset=["time"])
    return merged


def load_demo_vitals_df(
    demo_root: PathLike,
    top_n_events: int = 30,
    max_events_per_episode: Optional[int] = 200,
    include_labs: bool = True,
    include_chartevents: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Load MIMIC-IV demo events and return long-format vitals DataFrame.

    Returns:
        A tuple `(vitals_df, selected_events)` where:
        - `vitals_df` columns are: patient_id, episode_id, event, time, value
        - `selected_events` is the event list kept in the returned DataFrame.
    """
    root = _ensure_demo_root(demo_root)

    frames: List[pd.DataFrame] = []
    if include_labs:
        frames.append(_load_labevents(root))
    if include_chartevents:
        frames.append(_load_chartevents(root))
    if not frames:
        raise ValueError("At least one event source must be enabled")

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["subject_id", "hadm_id", "event", "time", "value"])
    df = _add_relative_hours(df, root)

    selected_events = df["event"].astype(str).value_counts().head(top_n_events).index.tolist()
    df = df[df["event"].isin(selected_events)].copy()

    df = df.rename(columns={"subject_id": "patient_id", "hadm_id": "episode_id"})
    df = df[["patient_id", "episode_id", "event", "time", "value"]]
    df = df.sort_values(["patient_id", "episode_id", "time"])

    if max_events_per_episode is not None:
        df = (
            df.groupby(["patient_id", "episode_id"], group_keys=False)
            .head(max_events_per_episode)
            .reset_index(drop=True)
        )

    return df.reset_index(drop=True), selected_events


def load_demo_labels(demo_root: PathLike) -> pd.DataFrame:
    """Load admission-level label table from MIMIC-IV demo admissions.

    Returns columns: patient_id, episode_id, label.
    """
    root = _ensure_demo_root(demo_root)
    adm = pd.read_csv(
        root / "hosp" / "admissions.csv",
        usecols=["subject_id", "hadm_id", "hospital_expire_flag"],
    )
    adm = adm.rename(
        columns={
            "subject_id": "patient_id",
            "hadm_id": "episode_id",
            "hospital_expire_flag": "label",
        }
    )
    return adm.dropna(subset=["patient_id", "episode_id", "label"])


def build_demo_pipeline_output(
    demo_root: PathLike,
    top_n_events: int = 30,
    max_seq_len: Optional[int] = 128,
    max_events_per_episode: Optional[int] = 200,
    with_labels: bool = True,
) -> Dict[str, Any]:
    """Build a FuseMoE-ready batch dict from the MIMIC-IV demo dataset.

    The resulting dictionary is the direct output of `MIMICIVPipeline.build`.
    """
    vitals_df, selected_events = load_demo_vitals_df(
        demo_root=demo_root,
        top_n_events=top_n_events,
        max_events_per_episode=max_events_per_episode,
        include_labs=True,
        include_chartevents=True,
    )

    vitals = VitalsLabsExtractor(selected_events=selected_events, max_seq_len=max_seq_len)
    vitals.fit(vitals_df)

    pipeline = MIMICIVPipeline(vitals_extractor=vitals)

    raw_batch: Dict[str, Any] = {"vitals_df": vitals_df}

    if with_labels:
        labels_df = load_demo_labels(demo_root)
        labels_df["id_key"] = list(zip(labels_df["patient_id"], labels_df["episode_id"]))
        raw_batch["label_ids"] = labels_df["id_key"].tolist()
        raw_batch["labels"] = labels_df["label"].astype("float32").to_numpy()

    return pipeline.build(raw_batch)
