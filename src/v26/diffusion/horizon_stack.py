"""Horizon Stack: Manages multi-horizon path concatenation and overlap handling.

Ensures smooth transitions between short/medium/long horizons
and provides utilities for extracting specific time ranges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from src.v26.diffusion.multi_horizon_generator import GeneratedHorizon, MultiHorizonResult


@dataclass
class StackedPath:
    """A single stacked path across all horizons."""
    short_idx: int
    medium_idx: int
    long_idx: int
    full_path: np.ndarray  # (C, total_len)
    short_segment: np.ndarray  # (C, 120)
    medium_segment: np.ndarray  # (C, 120)
    long_segment: np.ndarray  # (C, 120)
    regime: str
    consistency_score: float


class HorizonStack:
    """Manages stacking of multi-horizon paths.

    Handles:
    - Path concatenation with optional blending at boundaries
    - Extraction of specific time ranges
    - Consistency validation between horizons
    """

    SHORT_LEN = 120
    MEDIUM_LEN = 120
    LONG_LEN = 120
    OVERLAP = 10  # Bars to blend at boundaries

    def __init__(self, result: MultiHorizonResult):
        """Initialize from MultiHorizonResult.

        Args:
            result: Output from MultiHorizonGenerator
        """
        self.result = result
        self.short = result.short
        self.medium = result.medium
        self.long = result.long

        # Convert to numpy for easier manipulation
        self.short_np = result.short.paths.cpu().numpy()
        self.medium_np = result.medium.paths.cpu().numpy()
        self.long_np = result.long.paths.cpu().numpy()

    def stack_paths(
        self,
        blend: bool = True,
        blend_window: int = 10,
    ) -> List[StackedPath]:
        """Create all valid horizon combinations.

        Args:
            blend: Whether to blend overlapping regions
            blend_window: Number of bars to blend

        Returns:
            List of StackedPath objects
        """
        stacked = []

        # Create all combinations (Cartesian product)
        for s_idx in range(self.short_np.shape[0]):
            for m_idx in range(self.medium_np.shape[0]):
                for l_idx in range(self.long_np.shape[0]):
                    path = self._create_single_stack(s_idx, m_idx, l_idx, blend, blend_window)
                    stacked.append(path)

        return stacked

    def _create_single_stack(
        self,
        short_idx: int,
        medium_idx: int,
        long_idx: int,
        blend: bool,
        blend_window: int,
    ) -> StackedPath:
        """Create a single stacked path."""
        short_seg = self.short_np[short_idx].copy()
        med_seg = self.medium_np[medium_idx].copy()
        long_seg = self.long_np[long_idx].copy()

        if blend:
            # Blend short-medium boundary
            short_seg, med_seg = self._blend_boundary(
                short_seg, med_seg, blend_window
            )
            # Blend medium-long boundary
            med_seg, long_seg = self._blend_boundary(
                med_seg, long_seg, blend_window
            )

        # Concatenate
        full_path = np.concatenate([short_seg, med_seg, long_seg], axis=1)

        # Determine dominant regime
        short_regime = self.short.regime_probs[short_idx].argmax().item()
        medium_regime = self.medium.regime_probs[medium_idx].argmax().item()
        long_regime = self.long.regime_probs[long_idx].argmax().item()

        # Use short horizon regime as primary
        regime_labels = [
            "trend_up_strong", "trend_up_weak", "range",
            "mean_reversion", "breakout", "panic_news_shock",
            "trend_down_weak", "trend_down_strong", "low_volatility"
        ]
        primary_regime = regime_labels[short_regime]

        # Compute local consistency
        consistency = self._compute_local_consistency(short_seg, med_seg, long_seg)

        return StackedPath(
            short_idx=short_idx,
            medium_idx=medium_idx,
            long_idx=long_idx,
            full_path=full_path,
            short_segment=short_seg,
            medium_segment=med_seg,
            long_segment=long_seg,
            regime=primary_regime,
            consistency_score=consistency,
        )

    def _blend_boundary(
        self, left: np.ndarray, right: np.ndarray, window: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Blend two segments at their boundary.

        Args:
            left: (C, L) left segment
            right: (C, L) right segment
            window: number of bars to blend

        Returns:
            Tuple of blended (left, right)
        """
        if window <= 0:
            return left, right

        left_blended = left.copy()
        right_blended = right.copy()

        # Create linear blend weights
        weights = np.linspace(0, 1, window)

        # Blend right end of left with left end of right
        for i in range(window):
            w = weights[i]
            left_idx = left.shape[1] - window + i
            right_idx = i
            blended = (1 - w) * left[:, left_idx] + w * right[:, right_idx]
            left_blended[:, left_idx] = blended
            right_blended[:, right_idx] = blended

        return left_blended, right_blended

    def _compute_local_consistency(
        self, short: np.ndarray, medium: np.ndarray, long: np.ndarray
    ) -> float:
        """Compute consistency score for a specific path combination."""
        # End of short vs start of medium
        short_end = short[:, -10:].mean(axis=1)
        med_start = medium[:, :10].mean(axis=1)
        sm_sim = np.dot(short_end, med_start) / (
            np.linalg.norm(short_end) * np.linalg.norm(med_start) + 1e-8
        )

        # End of medium vs start of long
        med_end = medium[:, -10:].mean(axis=1)
        long_start = long[:, :10].mean(axis=1)
        ml_sim = np.dot(med_end, long_start) / (
            np.linalg.norm(med_end) * np.linalg.norm(long_start) + 1e-8
        )

        return float((sm_sim + ml_sim) / 2)

    def extract_time_range(
        self, start_bar: int, end_bar: int, stacked_paths: Optional[List[StackedPath]] = None
    ) -> np.ndarray:
        """Extract a specific time range from stacked paths.

        Args:
            start_bar: Starting bar (0-indexed)
            end_bar: Ending bar (exclusive)
            stacked_paths: Optional pre-computed stacks

        Returns:
            (num_paths, C, end_bar - start_bar) array
        """
        if stacked_paths is None:
            stacked_paths = self.stack_paths()

        extracts = []
        for sp in stacked_paths:
            if end_bar <= sp.full_path.shape[1]:
                extracts.append(sp.full_path[:, start_bar:end_bar])

        return np.stack(extracts) if extracts else np.array([])

    def get_horizon_boundaries(self) -> Dict[str, Tuple[int, int]]:
        """Get bar index boundaries for each horizon.

        Returns:
            Dict mapping horizon name to (start, end) tuple
        """
        return {
            "short": (0, self.SHORT_LEN),
            "medium": (self.SHORT_LEN, self.SHORT_LEN + self.MEDIUM_LEN),
            "long": (
                self.SHORT_LEN + self.MEDIUM_LEN,
                self.SHORT_LEN + self.MEDIUM_LEN + self.LONG_LEN,
            ),
        }

    def compute_horizon_statistics(self) -> Dict[str, Dict[str, float]]:
        """Compute statistics for each horizon."""
        stats = {}

        # Short horizon
        short_vol = np.std(self.short_np[:, 0, :], axis=1).mean()  # Return feature
        short_acf = self._compute_autocorr(self.short_np[:, 0, :])
        stats["short"] = {
            "volatility": float(short_vol),
            "acf_lag1": float(short_acf),
            "num_paths": self.short_np.shape[0],
        }

        # Medium horizon
        med_vol = np.std(self.medium_np[:, 0, :], axis=1).mean()
        med_acf = self._compute_autocorr(self.medium_np[:, 0, :])
        stats["medium"] = {
            "volatility": float(med_vol),
            "acf_lag1": float(med_acf),
            "num_paths": self.medium_np.shape[0],
        }

        # Long horizon
        long_vol = np.std(self.long_np[:, 0, :], axis=1).mean()
        long_acf = self._compute_autocorr(self.long_np[:, 0, :])
        stats["long"] = {
            "volatility": float(long_vol),
            "acf_lag1": float(long_acf),
            "num_paths": self.long_np.shape[0],
        }

        return stats

    def _compute_autocorr(self, x: np.ndarray, lag: int = 1) -> float:
        """Compute mean autocorrelation across paths."""
        if x.shape[1] <= lag:
            return 0.0

        x_centered = x - x.mean(axis=1, keepdims=True)
        var = np.var(x, axis=1).mean()
        if var < 1e-8:
            return 0.0

        cov = (x_centered[:, lag:] * x_centered[:, :-lag]).mean(axis=1).mean()
        return float(cov / var)

    def filter_by_consistency(self, min_score: float = 0.75) -> List[StackedPath]:
        """Filter paths by consistency score.

        Args:
            min_score: Minimum consistency score to keep

        Returns:
            Filtered list of StackedPath
        """
        all_paths = self.stack_paths()
        return [p for p in all_paths if p.consistency_score >= min_score]

    def to_branch_candidates(self, top_k: int = 20) -> List[Dict]:
        """Convert stacked paths to CABR branch candidates.

        Args:
            top_k: Number of top paths by consistency to return

        Returns:
            List of branch candidates for CABR ranking
        """
        all_paths = self.stack_paths()
        # Sort by consistency score
        sorted_paths = sorted(all_paths, key=lambda x: x.consistency_score, reverse=True)

        candidates = []
        for sp in sorted_paths[:top_k]:
            candidate = {
                "path_data": sp.full_path,
                "regime": sp.regime,
                "consistency_score": sp.consistency_score,
                "short_idx": sp.short_idx,
                "medium_idx": sp.medium_idx,
                "long_idx": sp.long_idx,
                "horizon_stack": True,
            }
            candidates.append(candidate)

        return candidates


def create_horizon_stack(
    multi_horizon_result: MultiHorizonResult,
    blend: bool = True,
    blend_window: int = 10,
) -> HorizonStack:
    """Factory function to create HorizonStack from MultiHorizonResult.

    Args:
        multi_horizon_result: Output from MultiHorizonGenerator
        blend: Whether to blend boundaries
        blend_window: Bars to blend

    Returns:
        Initialized HorizonStack
    """
    return HorizonStack(multi_horizon_result)