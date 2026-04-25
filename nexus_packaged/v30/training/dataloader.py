"""V30 DataLoader - creates training samples from aligned data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Generator


class DiffusionEvaluatorDataset(Dataset):
    """Dataset for training the path evaluator.
    
    For each sample:
    - Input: context window (features) + generated paths
    - Target: actual future outcome (for computing rewards)
    
    Supports two modes:
    1. Precomputed: Uses cached paths from disk (FAST)
    2. Generator: Generates paths on-the-fly (SLOW, for prototyping)
    """

    def __init__(
        self,
        features: np.ndarray,
        ohlcv: pd.DataFrame,
        timestamps: np.ndarray | None = None,
        precomputed_paths: np.ndarray | None = None,
        generator: callable | None = None,
        lookback: int = 128,
        horizon: int = 20,
        num_paths: int = 64,
        start_idx: int | None = None,
        end_idx: int | None = None,
    ):
        """
        Args:
            features: [N, lookback, feature_dim] - pre-computed features
            ohlcv: DataFrame with OHLCV data
            timestamps: [N] - timestamps for alignment
            precomputed_paths: [N_samples, num_paths, horizon] - OPTIONAL, for fast training
            generator: Function to generate diffusion paths (if precomputed not available)
            lookback: Context length
            horizon: Prediction horizon
            num_paths: Number of paths to generate
            start_idx, end_idx: Slice of data to use
        """
        self.features = features
        self.ohlcv = ohlcv
        self.timestamps = timestamps
        self.precomputed_paths = precomputed_paths
        self.generator = generator
        self.lookback = lookback
        self.horizon = horizon
        self.num_paths = num_paths
        
        # Determine valid range
        self.start_idx = start_idx if start_idx is not None else lookback
        self.end_idx = end_idx if end_idx is not None else len(features) - horizon - 1
        
        # Pre-compute close prices for actual outcomes
        self.close_prices = ohlcv["close"].values
        
    def __len__(self) -> int:
        return max(0, self.end_idx - self.start_idx)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get one training sample.
        
        Precomputed mode: idx maps directly to precomputed paths array
        The paths array has samples at 15-min intervals, so we sample
        features and OHLCV at 15-min intervals as well.
        
        Generator mode: idx maps to features array in 1-min units
        
        Returns:
            {
                "context": [lookback, feature_dim],
                "paths": [num_paths, horizon],
                "actual": [horizon],  # actual price trajectory
                "actual_return": float,
            }
        """
        if self.precomputed_paths is not None:
            # Precomputed mode: idx is directly into path array
            path_idx = idx
            
            # Get paths
            paths = torch.from_numpy(self.precomputed_paths[path_idx]).float()
            
            # For context: sample features at 15-min intervals
            # Need to ensure we have exactly 'lookback' samples
            # Start from path_idx - lookback, going backwards in 15-min steps
            ctx_start = path_idx - self.lookback
            if ctx_start < 0:
                # Not enough history, need to pad
                # Get available history
                available = path_idx
                if available > 0:
                    # Get available context
                    ctx_indices = [15 * i for i in range(available)]
                    ctx = self.features[ctx_indices]
                    # Pad to lookback
                    pad_size = self.lookback - available
                    pad = np.zeros((pad_size, self.features.shape[1]), dtype=self.features.dtype)
                    context = np.vstack([pad, ctx])
                else:
                    # No history available, use zeros
                    context = np.zeros((self.lookback, self.features.shape[1]), dtype=self.features.dtype)
            else:
                # Full history available
                ctx_indices = [15 * i for i in range(ctx_start, path_idx)]
                context = self.features[ctx_indices]
            
            context = torch.from_numpy(context).float()
            
            # For actual: get OHLCV at 15-min intervals
            actual_indices = [15 * path_idx + i * 15 for i in range(self.horizon)]
            actual = self.close_prices[actual_indices]
            actual = torch.from_numpy(actual).float()
            
        else:
            # Generator mode
            real_idx = self.start_idx + idx
            
            # Context from features
            context = torch.from_numpy(self.features[real_idx - self.lookback:real_idx]).float()
            
            # Get paths
            if self.generator is not None:
                paths = self.generator(context.numpy())
                paths = torch.from_numpy(paths).float()
            else:
                base_price = self.close_prices[real_idx]
                paths = torch.randn(self.num_paths, self.horizon) * 0.01 * base_price + base_price
            
            # Actual future
            actual = self.close_prices[real_idx:real_idx + self.horizon]
            actual = torch.from_numpy(actual).float()
        
        # Actual return
        actual_return = (actual[-1] - actual[0]) / (actual[0] + 1e-8)
        
        return {
            "context": context,  # [lookback, feature_dim]
            "paths": paths,       # [num_paths, horizon]
            "actual": actual,     # [horizon]
            "actual_return": actual_return,
        }


def create_dataloaders(
    features: np.ndarray,
    ohlcv: pd.DataFrame,
    generator: callable,
    config: dict,
    train_end_date: str = "2020-12-31",
    val_start_date: str = "2021-01-01",
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders with temporal split.
    
    Args:
        features: Feature array
        ohlcv: OHLCV DataFrame  
        generator: Diffusion model for path generation
        config: Model config
        train_end_date: End of training period
        val_start_date: Start of validation period
    
    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader
    
    lookback = config.get("lookback", 128)
    horizon = config.get("horizon", 20)
    num_paths = config.get("num_paths", 64)
    
    # Get timestamps for splitting
    if "timestamp" in ohlcv.columns:
        timestamps = pd.to_datetime(ohlcv.index)
    else:
        timestamps = pd.to_datetime(ohlcv.index)
    
    # Find split indices
    train_end_ts = pd.Timestamp(train_end_date)
    val_start_ts = pd.Timestamp(val_start_date)
    
    train_end_idx = timestamps.searchsorted(train_end_ts)
    val_start_idx = timestamps.searchsorted(val_start_ts)
    
    # Create datasets
    train_dataset = DiffusionEvaluatorDataset(
        features=features,
        ohlcv=ohlcv,
        generator=generator,
        lookback=lookback,
        horizon=horizon,
        num_paths=num_paths,
        start_idx=lookback,
        end_idx=train_end_idx,
    )
    
    val_dataset = DiffusionEvaluatorDataset(
        features=features,
        ohlcv=ohlcv,
        generator=generator,
        lookback=lookback,
        horizon=horizon,
        num_paths=num_paths,
        start_idx=val_start_idx,
        end_idx=len(features) - horizon - 1,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


def load_alignment_data(config: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """Load aligned features and OHLCV data.
    
    Args:
        config: Dict with data paths
    
    Returns:
        features, ohlcv
    """
    features_path = config.get("features_path")
    ohlcv_path = config.get("ohlcv_path")
    
    # Handle relative paths - resolve from project root
    import os
    project_root = Path(__file__).resolve().parents[3]
    
    if not Path(features_path).is_absolute():
        features_path = project_root / features_path
    if not Path(ohlcv_path).is_absolute():
        ohlcv_path = project_root / ohlcv_path
    
    features = np.load(features_path)
    ohlcv = pd.read_parquet(ohlcv_path)
    
    return features, ohlcv