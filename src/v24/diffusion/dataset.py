"""RAM-backed rolling-window dataset for diffusion model training.

Phase 0.5: Loads entire fused matrix into RAM (3.5 GB) for fast random access.
Returns (window, past_context) where past_context provides temporal depth
for the GRU temporal encoder.

Supports max_samples for subsampling — random-shuffles indices each epoch
so the model sees different windows every epoch without needing all 4M+.
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
    """Rolling-window dataset backed by RAM-loaded arrays for diffusion training.

    Each sample returns (window, past_context) where:
    - window: (C, L) tensor — the feature sequence to be diffused
    - past_context: (context_len, C) tensor — the preceding bars for temporal encoding
    If context_len=0, returns a zero-context vector of shape (C,) for backward compat.

    Args:
        feature_path: Path to .npy fused feature matrix (N, C).
        sequence_len: Rolling window length L.
        row_slice: Optional DatasetSlice for train/val/test split.
        timestamp_path: Optional path to timestamp array for year-based splitting.
        context_len: Number of past bars for temporal context (0 = legacy mode).
        max_samples: If set, subsample to this many samples per epoch (random each epoch).
        load_to_ram: If True, load entire .npy into RAM (3.5 GB, much faster than mmap).
    """

    def __init__(
        self,
        feature_path: Path,
        sequence_len: int = 120,
        row_slice: Optional[DatasetSlice] = None,
        timestamp_path: Optional[Path] = None,
        context_len: int = 0,
        max_samples: int = 0,
        load_to_ram: bool = True,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch required for DiffusionDataset")

        self.sequence_len = sequence_len
        self.context_len = context_len
        self.max_samples = max_samples
        self.timestamps = None

        if load_to_ram:
            print(f"  [Dataset] Loading {feature_path} into RAM ...")
            import time
            t0 = time.time()
            self.features = np.load(str(feature_path)).astype(np.float32)
            print(f"  [Dataset] RAM load: {time.time()-t0:.1f}s, shape={self.features.shape}, "
                  f"{self.features.nbytes/1e9:.2f} GB")
        else:
            self.features = np.load(str(feature_path), mmap_mode="r")

        if timestamp_path is not None and timestamp_path.exists():
            self.timestamps = np.load(str(timestamp_path), mmap_mode="r")

        context_offset = context_len if context_len > 0 else 0
        usable = len(self.features) - sequence_len - context_offset
        if usable <= 0:
            raise ValueError(f"Not enough rows ({len(self.features)}) for sequence_len={sequence_len} + context_len={context_len}")

        if row_slice is None:
            self.row_slice = DatasetSlice(0, usable)
        else:
            self.row_slice = DatasetSlice(
                max(0, row_slice.start),
                min(usable, row_slice.stop),
            )
        if len(self.row_slice) <= 0:
            raise ValueError("Empty dataset slice after bounds checking")

        self._full_length = len(self.row_slice)
        self._epoch_indices = np.arange(self._full_length)
        self._shuffle_indices()

    def _shuffle_indices(self) -> None:
        np.random.shuffle(self._epoch_indices)
        if self.max_samples > 0 and self.max_samples < self._full_length:
            self._epoch_indices = self._epoch_indices[:self.max_samples]

    def __len__(self) -> int:
        return len(self._epoch_indices)

    def __getitem__(self, idx: int):
        base = self.row_slice.start + self._epoch_indices[idx]
        window_start = base + self.context_len
        window = self.features[window_start:window_start + self.sequence_len].copy()
        window_t = torch.from_numpy(window).permute(1, 0)

        if self.context_len > 0:
            ctx_start = base
            past_ctx = self.features[ctx_start:ctx_start + self.context_len].copy()
            past_ctx_t = torch.from_numpy(past_ctx)
        else:
            context = self.features[window_start + self.sequence_len - 1].copy()
            past_ctx_t = torch.from_numpy(context)

        return window_t, past_ctx_t

    def new_epoch(self) -> None:
        """Reshuffle indices for a new epoch. Call before each epoch's dataloader iteration."""
        self._epoch_indices = np.arange(self._full_length)
        self._shuffle_indices()


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
