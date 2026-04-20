"""Memmap-backed rolling-window dataset for diffusion model training.

Loads the 405K-row fused feature matrix and creates overlapping
windows of length `sequence_len`. Supports train/val/test split
by year via row_slice (DatasetSlice).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object


@dataclass(frozen=True)
class DatasetSlice:
    start: int
    stop: int

    def __len__(self) -> int:
        return max(0, self.stop - self.start)


class DiffusionDataset(Dataset):
    """Rolling-window dataset backed by memmap arrays for diffusion training.

    Each sample returns (window, context) where:
      - window: (C, L) tensor — the feature sequence
      - context: (C,) tensor — the last timestep as conditioning vector

    Args:
        feature_path: Path to .npy fused feature matrix (N, C).
        sequence_len: Rolling window length L.
        row_slice: Optional DatasetSlice for train/val/test split.
        timestamp_path: Optional path to timestamp array for year-based splitting.
    """

    def __init__(
        self,
        feature_path: Path,
        sequence_len: int = 120,
        row_slice: Optional[DatasetSlice] = None,
        timestamp_path: Optional[Path] = None,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch required for DiffusionDataset")

        self.features = np.load(feature_path, mmap_mode="r")
        self.sequence_len = sequence_len
        self.timestamps = None
        if timestamp_path is not None and timestamp_path.exists():
            self.timestamps = np.load(timestamp_path, mmap_mode="r")

        usable = len(self.features) - sequence_len
        if usable <= 0:
            raise ValueError(f"Not enough rows ({len(self.features)}) for sequence_len={sequence_len}")

        if row_slice is None:
            self.row_slice = DatasetSlice(0, usable)
        else:
            self.row_slice = DatasetSlice(
                max(0, row_slice.start),
                min(usable, row_slice.stop),
            )
        if len(self.row_slice) <= 0:
            raise ValueError("Empty dataset slice after bounds checking")

    def __len__(self) -> int:
        return len(self.row_slice)

    def __getitem__(self, idx: int):
        base = self.row_slice.start + idx
        window = np.asarray(self.features[base:base + self.sequence_len], dtype=np.float32).copy()
        context = np.asarray(self.features[base + self.sequence_len - 1], dtype=np.float32).copy()
        window_t = torch.from_numpy(window).permute(1, 0)
        context_t = torch.from_numpy(context)
        return window_t, context_t


def split_by_year(total_rows: int, sequence_len: int, timestamps: Optional[np.ndarray] = None,
                  train_end_year: int = 2021, val_end_year: int = 2024) -> tuple[DatasetSlice, DatasetSlice, DatasetSlice]:
    """Split dataset into train/val/test by year boundaries.

    Falls back to proportional split if timestamps unavailable.
    """
    usable = total_rows - sequence_len
    if usable <= 0:
        raise ValueError("Not enough rows for sequence windows")

    if timestamps is not None and len(timestamps) >= total_rows:
        import pandas as pd
        ts = pd.to_datetime(timestamps[:total_rows])
        years = ts.year.values
        train_end_idx = 0
        val_end_idx = 0
        for i in range(usable):
            if years[i + sequence_len] < train_end_year and train_end_idx == 0 or train_end_idx == 0:
                if years[i + sequence_len] >= train_end_year:
                    train_end_idx = i
            if years[i + sequence_len] < val_end_year and val_end_idx == 0:
                if years[i + sequence_len] >= val_end_year and train_end_idx > 0:
                    val_end_idx = i
        if train_end_idx == 0:
            train_end_idx = int(usable * 0.70)
        if val_end_idx == 0:
            val_end_idx = int(usable * 0.85)
        return (
            DatasetSlice(0, train_end_idx),
            DatasetSlice(train_end_idx, val_end_idx),
            DatasetSlice(val_end_idx, usable),
        )

    train_end = int(usable * 0.70)
    val_end = int(usable * 0.85)
    return (
        DatasetSlice(0, train_end),
        DatasetSlice(train_end, val_end),
        DatasetSlice(val_end, usable),
    )
