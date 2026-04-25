"""Diagnostic: understand why path return aggregation fails.

Key diagnostic questions:
1. What is the distribution of path returns?
2. What does uniform weighting produce as expected return?
3. What does the generator produce given context?
4. Does attention actually learn to weight the "correct" paths?
"""

import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path

script_path = Path(__file__).resolve()
repo_root = script_path.parents[2]
sys.path.insert(0, str(script_path.parents[1]))

features = np.load(repo_root / "data/features/diffusion_fused_6m.npy", mmap_mode="r")[:10000]
ohlcv_df = pd.read_parquet(repo_root / "nexus_packaged/data/ohlcv.parquet")
close_prices = ohlcv_df["close"].values.astype(np.float32)

paths_file = repo_root / "nexus_packaged/v30/data/processed/v30_1m_paths.npy"
paths_fp = np.memmap(paths_file, dtype=np.float32, mode="r", shape=(599744, 64, 20))
paths = np.array(paths_fp[:10000]).astype(np.float32)

actual_starts = np.arange(0, 10000)
actual_ends = np.minimum(actual_starts + 21, len(close_prices))
actuals = []
for s, e in zip(actual_starts, actual_ends):
    a = close_prices[s:e]
    if len(a) < 21:
        pad = np.zeros(21, dtype=np.float32)
        pad[: len(a)] = a
        a = pad
    actuals.append(a)
actuals = np.array(actuals)

print("=" * 60)
print("DIAGNOSTIC: Understanding path return distribution")
print("=" * 60)

path_returns = (paths[:, :, -1] - paths[:, :, 0]) / (paths[:, :, 0] + 1e-8)
actual_returns = (actuals[:, -1] - actuals[:, 0]) / (actuals[:, 0] + 1e-8)

print(f"\nPath returns (64 paths per sample):")
print(f"  Shape: {path_returns.shape}")
print(f"  Mean across paths: {path_returns.mean():.6f}")
print(f"  Std across paths: {path_returns.std():.6f}")
print(f"  Min: {path_returns.min():.4f}, Max: {path_returns.max():.4f}")
print(f"  Median: {np.median(path_returns):.6f}")

print(f"\nActual returns:")
print(f"  Mean: {actual_returns.mean():.6f}")
print(f"  Std: {actual_returns.std():.6f}")
print(f"  Min: {actual_returns.min():.4f}, Max: {actual_returns.max():.4f}")
print(f"  % positive: {(actual_returns > 0).mean():.4f}")

uniform_expected = path_returns.mean(axis=1)
print(f"\nUniform weight aggregation (mean of 64 paths):")
print(f"  Corr with actual: {np.corrcoef(uniform_expected, actual_returns)[0,1]:.4f}")
print(f"  Std of prediction: {uniform_expected.std():.6f}")
print(f"  Mean of prediction: {uniform_expected.mean():.6f}")

median_expected = np.median(path_returns, axis=1)
print(f"\nMedian aggregation:")
print(f"  Corr with actual: {np.corrcoef(median_expected, actual_returns)[0,1]:.4f}")

best_path_idx = np.argmax(np.abs(path_returns), axis=1)
best_path_returns = np.array([path_returns[i, best_path_idx[i]] for i in range(len(path_returns))])
print(f"\nBest path by absolute return (cheating):")
print(f"  Corr with actual: {np.corrcoef(best_path_returns, actual_returns)[0,1]:.4f}")

diffs = np.abs(path_returns - actual_returns[:, np.newaxis])
closest_idx = np.argmin(diffs, axis=1)
closest_returns = np.array([path_returns[i, closest_idx[i]] for i in range(len(path_returns))])
print(f"\nClosest path by L2 distance:")
print(f"  Corr with actual: {np.corrcoef(closest_returns, actual_returns)[0,1]:.4f}")

print(f"\nMean return by regime (path_returns mean across paths):")
pos_regime = path_returns.mean(axis=1) > 0
neg_regime = path_returns.mean(axis=1) < 0
print(f"  Positive regime: {pos_regime.mean():.4f}, actual return mean: {actual_returns[pos_regime].mean():.4f}")
print(f"  Negative regime: {neg_regime.mean():.4f}, actual return mean: {actual_returns[neg_regime].mean():.4f}")

print(f"\nStd of path returns per sample:")
print(f"  Mean std: {path_returns.std(axis=1).mean():.4f}")
print(f"  This means paths are very spread out!")

print(f"\nKey insight: if paths span {-path_returns.max():.2f} to {path_returns.max():.2f},")
print(f"then mean aggregation will have near-zero correlation with actual return")
print(f"unless the model learns to concentrate on the RIGHT subset of paths.")