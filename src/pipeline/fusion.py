from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from config.project_config import FEATURE_DIM_TOTAL, PRICE_FEATURE_COLUMNS

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None


@dataclass(frozen=True)
class FusionReport:
    rows: int
    feature_dim: int
    target_positive_rate: float
    source_price_path: str
    source_news_path: str
    source_crowd_path: str
    sequence_rows: int = 0
    sequence_len: int = 0
    source_persona_path: str = ""


def _require_pandas() -> Any:
    if pd is None:
        raise ImportError("pandas is required for fusion operations.")
    return pd


def load_price_frame(price_path: Path):
    pandas = _require_pandas()
    if price_path.suffix.lower() == ".parquet":
        frame = pandas.read_parquet(price_path)
    else:
        frame = pandas.read_csv(price_path, index_col=0, parse_dates=True)
    return frame


def extract_price_block(frame) -> np.ndarray:
    missing = [column for column in PRICE_FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required price columns: {', '.join(missing)}")
    return frame[PRICE_FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)


def extract_target_vector(frame, target_column: str = "target_direction") -> np.ndarray:
    if target_column not in frame.columns:
        raise ValueError(f"Missing target column: {target_column}")
    return frame[target_column].to_numpy(dtype=np.float32, copy=True)


def normalize_binary_targets(values: np.ndarray) -> np.ndarray:
    return (values > 0).astype(np.float32)


def align_row_count(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    if not arrays:
        return ()
    row_count = min(len(array) for array in arrays)
    return tuple(np.asarray(array[:row_count], dtype=np.float32) for array in arrays)


def build_fused_feature_matrix(price_block: np.ndarray, news_block: np.ndarray, crowd_block: np.ndarray) -> np.ndarray:
    price_block, news_block, crowd_block = align_row_count(price_block, news_block, crowd_block)
    fused = np.concatenate([price_block, news_block, crowd_block], axis=1)
    if fused.shape[1] != FEATURE_DIM_TOTAL:
        raise ValueError(f"Expected fused width {FEATURE_DIM_TOTAL}, got {fused.shape[1]}")
    return fused.astype(np.float32, copy=False)


def build_sequence_tensor(feature_matrix: np.ndarray, target_vector: np.ndarray, sequence_len: int) -> tuple[np.ndarray, np.ndarray]:
    if sequence_len <= 0:
        raise ValueError("sequence_len must be positive")
    if len(feature_matrix) != len(target_vector):
        raise ValueError("Feature matrix and target vector must have the same row count")
    usable = len(feature_matrix) - sequence_len + 1
    if usable <= 0:
        raise ValueError("Not enough rows to build sequence tensor")

    tensor = np.stack([feature_matrix[index : index + sequence_len] for index in range(usable)], axis=0).astype(np.float32, copy=False)
    seq_targets = np.asarray(target_vector[sequence_len - 1 :], dtype=np.float32)
    return tensor, seq_targets


def save_numpy_artifact(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def save_fusion_report(path: Path, report: FusionReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.__dict__, indent=2), encoding="utf-8")
