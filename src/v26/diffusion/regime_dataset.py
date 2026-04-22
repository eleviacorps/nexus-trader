"""Regime-aware dataset for V26 Phase 1 - Agent 3.

Extends DiffusionDataset to include regime labels for each training window.
Pre-computes regime labels using v6/regime_detection.py for efficiency.

Returns tuple: (window, past_context, regime_probs)
- window: (C, L) tensor — the feature sequence to be diffused
- past_context: (context_len, C) tensor — preceding bars for temporal encoding
- regime_probs: (num_regimes,) tensor — regime probabilities for conditioning
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.v24.diffusion.dataset import DatasetSlice
from src.v6.regime_detection import REGIME_LABELS, detect_regime


class RegimeDiffusionDataset(Dataset):
    """Rolling-window dataset with pre-computed regime labels for V26.

    Each sample returns (window, past_context, regime_probs) where:
    - window: (C, L) tensor — the feature sequence to be diffused
    - past_context: (context_len, C) tensor — preceding bars for temporal encoding
    - regime_probs: (num_regimes,) tensor — regime probabilities for conditioning

    Regime labels are pre-computed at initialization and stored alongside features
    for efficiency (regime detection on-the-fly is too slow).

    Args:
        feature_path: Path to .npy fused feature matrix (N, C).
        sequence_len: Rolling window length L.
        row_slice: Optional DatasetSlice for train/val/test split.
        timestamp_path: Optional path to timestamp array for year-based splitting.
        context_len: Number of past bars for temporal context.
        max_samples: If set, subsample to this many samples per epoch.
        load_to_ram: If True, load entire .npy into RAM.
        regime_cache_path: Optional path to pre-computed regime labels .npy file.
            If not provided or doesn't exist, regime labels will be computed and cached.
    """

    def __init__(
        self,
        feature_path: Path,
        sequence_len: int = 120,
        row_slice: Optional[DatasetSlice] = None,
        timestamp_path: Optional[Path] = None,
        context_len: int = 256,
        max_samples: int = 0,
        load_to_ram: bool = True,
        regime_cache_path: Optional[Path] = None,
    ) -> None:
        self.sequence_len = sequence_len
        self.context_len = context_len
        self.max_samples = max_samples
        self.timestamps = None
        self.num_regimes = len(REGIME_LABELS)

        # Load features
        if load_to_ram:
            print(f" [RegimeDataset] Loading {feature_path} into RAM ...")
            import time
            t0 = time.time()
            self.features = np.load(str(feature_path)).astype(np.float32)
            print(f" [RegimeDataset] RAM load: {time.time()-t0:.1f}s, shape={self.features.shape}, "
                  f"{self.features.nbytes/1e9:.2f} GB")
        else:
            self.features = np.load(str(feature_path), mmap_mode="r")

        if timestamp_path is not None and timestamp_path.exists():
            self.timestamps = np.load(str(timestamp_path), mmap_mode="r")

        # Calculate usable range
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

        # Load or compute regime labels
        self.regime_probs = self._load_or_compute_regimes(regime_cache_path)

    def _load_or_compute_regimes(self, regime_cache_path: Optional[Path]) -> np.ndarray:
        """Load pre-computed regime labels or compute and cache them."""
        if regime_cache_path is not None and regime_cache_path.exists():
            print(f" [RegimeDataset] Loading pre-computed regimes from {regime_cache_path}")
            regimes = np.load(str(regime_cache_path)).astype(np.float32)
            # Verify shape matches
            expected_shape = (len(self.features), self.num_regimes)
            if regimes.shape != expected_shape:
                print(f" [RegimeDataset] Warning: Cached regimes shape {regimes.shape} != expected {expected_shape}")
                print(f" [RegimeDataset] Recomputing regime labels...")
                regimes = self._compute_regimes()
                if regime_cache_path is not None:
                    np.save(str(regime_cache_path), regimes)
                    print(f" [RegimeDataset] Saved regime cache to {regime_cache_path}")
            return regimes
        else:
            print(f" [RegimeDataset] Computing regime labels for {len(self.features)} samples...")
            regimes = self._compute_regimes()
            if regime_cache_path is not None:
                regime_cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(regime_cache_path), regimes)
                print(f" [RegimeDataset] Saved regime cache to {regime_cache_path}")
            return regimes

    def _compute_regimes(self) -> np.ndarray:
        """Compute regime probabilities for all feature rows.

        Returns:
            Array of shape (N, num_regimes) with regime probabilities.
        """
        num_samples = len(self.features)
        regime_probs = np.zeros((num_samples, self.num_regimes), dtype=np.float32)

        # Get feature column names (assuming standard fused feature format)
        # This is a heuristic - adjust based on actual feature structure
        for i in range(num_samples):
            # Convert feature row to dict for regime detection
            row = self.features[i]
            row_dict = self._features_to_dict(row)

            # Detect regime
            result = detect_regime(row_dict)

            # Store probabilities in consistent order
            for j, label in enumerate(REGIME_LABELS):
                regime_probs[i, j] = result.probabilities.get(label, 0.0)

            if (i + 1) % 10000 == 0:
                print(f" [RegimeDataset] Computed {i+1}/{num_samples} regimes")

        return regime_probs

    def _features_to_dict(self, row: np.ndarray) -> dict[str, float]:
        """Convert feature row array to dict for regime detection.

        Uses common feature naming conventions. Adjust indices based on
        actual feature structure in fused matrix.
        """
        # Default feature indices (common in fused matrices)
        # These should be adjusted based on actual feature structure
        return {
            # Core features (assuming first few columns)
            "quant_trend_score": float(row[0]) if len(row) > 0 else 0.0,
            "ema_cross": float(row[1]) if len(row) > 1 else 0.0,
            "macd_hist": float(row[2]) if len(row) > 2 else 0.0,
            "analog_bias": float(row[3]) if len(row) > 3 else 0.0,
            "quant_transition_risk": float(row[4]) if len(row) > 4 else 0.0,
            "quant_regime_strength": float(row[5]) if len(row) > 5 else 0.0,
            "quant_vol_forecast": float(row[6]) if len(row) > 6 else 0.0,
            "atr_pct": float(row[7]) if len(row) > 7 else 0.001,
            "quant_tail_risk": float(row[8]) if len(row) > 8 else 0.0,
            "news_shock": float(row[9]) if len(row) > 9 else 0.0,
            "llm_event_severity": float(row[10]) if len(row) > 10 else 0.0,
            "crowd_panic_index": float(row[11]) if len(row) > 11 else 0.0,
            "crowd_bias": float(row[12]) if len(row) > 12 else 0.0,
            "crowd_extreme": float(row[13]) if len(row) > 13 else 0.0,
            "macro_bias": float(row[14]) if len(row) > 14 else 0.0,
            "quant_route_prob_up": float(row[15]) if len(row) > 15 else 0.25,
            "quant_route_prob_down": float(row[16]) if len(row) > 16 else 0.25,
            "quant_route_prob_range": float(row[17]) if len(row) > 17 else 0.25,
            "quant_route_prob_chop": float(row[18]) if len(row) > 18 else 0.25,
        }

    def _shuffle_indices(self) -> None:
        np.random.shuffle(self._epoch_indices)
        if self.max_samples > 0 and self.max_samples < self._full_length:
            self._epoch_indices = self._epoch_indices[:self.max_samples]

    def __len__(self) -> int:
        return len(self._epoch_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single sample with regime labels.

        Returns:
            window: (C, L) tensor — feature sequence to diffuse
            past_context: (context_len, C) tensor — preceding bars for temporal encoding
            regime_probs: (num_regimes,) tensor — regime probabilities
        """
        base = self.row_slice.start + self._epoch_indices[idx]
        window_start = base + self.context_len

        # Get window features
        window = self.features[window_start:window_start + self.sequence_len].copy()
        window_t = torch.from_numpy(window).permute(1, 0)  # (C, L)

        # Get past context
        if self.context_len > 0:
            ctx_start = base
            past_ctx = self.features[ctx_start:ctx_start + self.context_len].copy()
            past_ctx_t = torch.from_numpy(past_ctx)  # (context_len, C)
        else:
            context = self.features[window_start + self.sequence_len - 1].copy()
            past_ctx_t = torch.from_numpy(context)

        # Get regime label (use the regime at the end of past context / start of window)
        regime_idx = window_start
        regime_t = torch.from_numpy(self.regime_probs[regime_idx].copy())  # (num_regimes,)

        return window_t, past_ctx_t, regime_t

    def new_epoch(self) -> None:
        """Reshuffle indices for a new epoch."""
        self._epoch_indices = np.arange(self._full_length)
        self._shuffle_indices()

    def get_regime_distribution(self) -> dict[str, float]:
        """Get the distribution of dominant regimes in the dataset."""
        dominant_regimes = np.argmax(self.regime_probs, axis=1)
        counts = np.bincount(dominant_regimes, minlength=self.num_regimes)
        total = len(dominant_regimes)
        return {
            REGIME_LABELS[i]: float(counts[i]) / total
            for i in range(self.num_regimes)
        }
