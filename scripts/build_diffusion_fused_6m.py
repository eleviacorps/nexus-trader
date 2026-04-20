"""Build the 6M-row, 144-feature fused matrix for Phase 0.5 diffusion training.

Sources (all 6,024,602 rows at 1-min resolution, perfectly time-aligned):
  - data_store/processed/XAUUSD_1m_features.parquet  → 36 price features
  - data/embeddings/news_embeddings.npy               → 32 news embeddings
  - data/embeddings/crowd_embeddings.npy              → 32 crowd embeddings
  - data/processed/market_dynamics_labels.parquet     → 23 numeric features (excl OHLCV + label)
  - data/processed/quant_features.parquet             → 21 quant features

Total: 36 + 32 + 32 + 23 + 21 = 144 features.

No aggregation needed — all sources join directly by row index (same timestamps).

Usage:
    python scripts/build_diffusion_fused_6m.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import (
    EMBEDDINGS_DIR,
    FEATURES_DIR,
    LEGACY_PROCESSED_DIR,
    PRICE_FEATURE_COLUMNS,
    PROCESSED_DATA_DIR,
)

PRICE_PATH = LEGACY_PROCESSED_DIR / "XAUUSD_1m_features.parquet"
NEWS_EMB_NPY_PATH = EMBEDDINGS_DIR / "news_embeddings.npy"
CROWD_EMB_NPY_PATH = EMBEDDINGS_DIR / "crowd_embeddings.npy"
MARKET_DYNAMICS_PATH = PROCESSED_DATA_DIR / "market_dynamics_labels.parquet"
QUANT_FEATURES_PATH = PROCESSED_DATA_DIR / "quant_features.parquet"

OUT_FUSED_PATH = FEATURES_DIR / "diffusion_fused_6m.npy"
OUT_TIMESTAMPS_PATH = FEATURES_DIR / "diffusion_timestamps_6m.npy"
OUT_NORM_STATS_PATH = PROJECT_ROOT / "config" / "diffusion_norm_stats_6m.json"

MARKET_DYNAMICS_DROP = {"open", "high", "low", "close", "market_dynamics_label"}

DIM_PRICE = 36
DIM_NEWS = 32
DIM_CROWD = 32
DIM_MARKET_DYNAMICS = 23
DIM_QUANT = 21
DIM_TOTAL = DIM_PRICE + DIM_NEWS + DIM_CROWD + DIM_MARKET_DYNAMICS + DIM_QUANT


def load_price_features():
    print(f"[1/5] Loading price features from {PRICE_PATH} ...")
    t0 = time.time()
    df = pd.read_parquet(PRICE_PATH, columns=PRICE_FEATURE_COLUMNS)
    price_matrix = df.values.astype(np.float32)
    timestamps = df.index
    print(f"  Shape: {price_matrix.shape}, elapsed: {time.time()-t0:.1f}s")
    return price_matrix, timestamps


def load_embeddings(npy_path, label):
    print(f"  Loading {label} embeddings from {npy_path} ...")
    t0 = time.time()
    emb = np.load(npy_path, mmap_mode="r")
    arr = np.asarray(emb, dtype=np.float32)
    print(f"  {label} shape: {arr.shape}, elapsed: {time.time()-t0:.1f}s")
    return arr


def load_market_dynamics():
    print(f"[3/5] Loading market dynamics from {MARKET_DYNAMICS_PATH} ...")
    t0 = time.time()
    df = pd.read_parquet(MARKET_DYNAMICS_PATH)
    numeric_cols = [c for c in df.columns if c not in MARKET_DYNAMICS_DROP and df[c].dtype in ("float32", "float64", "int32", "int64")]
    numeric_cols.sort()
    mat = df[numeric_cols].values.astype(np.float32)
    print(f"  Shape: {mat.shape}, columns: {len(numeric_cols)}, elapsed: {time.time()-t0:.1f}s")
    print(f"  Columns: {numeric_cols}")
    return mat, numeric_cols


def load_quant_features():
    print(f"[4/5] Loading quant features from {QUANT_FEATURES_PATH} ...")
    t0 = time.time()
    df = pd.read_parquet(QUANT_FEATURES_PATH)
    numeric_cols = [c for c in df.columns if df[c].dtype in ("float32", "float64", "int32", "int64")]
    numeric_cols.sort()
    mat = df[numeric_cols].values.astype(np.float32)
    print(f"  Shape: {mat.shape}, columns: {len(numeric_cols)}, elapsed: {time.time()-t0:.1f}s")
    print(f"  Columns: {numeric_cols}")
    return mat, numeric_cols


def fuse_all(price, news, crowd, market_dyn, quant):
    print(f"[5/5] Fusing all sources into {DIM_TOTAL}-feature matrix ...")
    t0 = time.time()
    n = len(price)
    assert len(news) == n, f"News rows {len(news)} != price rows {n}"
    assert len(crowd) == n, f"Crowd rows {len(crowd)} != price rows {n}"
    assert len(market_dyn) == n, f"Market dynamics rows {len(market_dyn)} != price rows {n}"
    assert len(quant) == n, f"Quant rows {len(quant)} != price rows {n}"

    fused = np.empty((n, DIM_TOTAL), dtype=np.float32)
    c = 0
    fused[:, c:c+DIM_PRICE] = price; c += DIM_PRICE
    fused[:, c:c+DIM_NEWS] = news; c += DIM_NEWS
    fused[:, c:c+DIM_CROWD] = crowd; c += DIM_CROWD
    fused[:, c:c+DIM_MARKET_DYNAMICS] = market_dyn; c += DIM_MARKET_DYNAMICS
    fused[:, c:c+DIM_QUANT] = quant; c += DIM_QUANT
    assert c == DIM_TOTAL

    nan_count = np.isnan(fused).sum()
    inf_count = np.isinf(fused).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"  WARNING: {nan_count} NaN, {inf_count} Inf — replacing with 0")
        fused = np.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Fused shape: {fused.shape}, elapsed: {time.time()-t0:.1f}s")
    return fused


def normalize_and_save(fused, timestamps):
    print("Normalizing and saving ...")
    t0 = time.time()

    means = np.mean(fused, axis=0).astype(np.float64)
    stds = np.std(fused, axis=0).astype(np.float64)
    stds[stds < 1e-8] = 1.0

    fused_norm = ((fused - means) / stds).astype(np.float32)

    nan_count = np.isnan(fused_norm).sum()
    inf_count = np.isinf(fused_norm).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"  WARNING: {nan_count} NaN, {inf_count} Inf after normalization — clipping")
        fused_norm = np.nan_to_num(fused_norm, nan=0.0, posinf=3.0, neginf=-3.0)

    fused_norm = np.clip(fused_norm, -5.0, 5.0)

    norm_stats = {
        "means": means.tolist(),
        "stds": stds.tolist(),
        "feature_groups": {
            "price": {"start": 0, "dim": DIM_PRICE},
            "news": {"start": DIM_PRICE, "dim": DIM_NEWS},
            "crowd": {"start": DIM_PRICE + DIM_NEWS, "dim": DIM_CROWD},
            "market_dynamics": {"start": DIM_PRICE + DIM_NEWS + DIM_CROWD, "dim": DIM_MARKET_DYNAMICS},
            "quant": {"start": DIM_PRICE + DIM_NEWS + DIM_CROWD + DIM_MARKET_DYNAMICS, "dim": DIM_QUANT},
        },
        "n_rows": int(fused_norm.shape[0]),
        "n_features": DIM_TOTAL,
    }

    print(f"  Saving fused matrix ({fused_norm.nbytes / 1e9:.2f} GB) to {OUT_FUSED_PATH} ...")
    OUT_FUSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_FUSED_PATH, fused_norm)

    ts_strings = np.array([str(t) for t in timestamps])
    print(f"  Saving timestamps to {OUT_TIMESTAMPS_PATH} ...")
    np.save(OUT_TIMESTAMPS_PATH, ts_strings)

    OUT_NORM_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_NORM_STATS_PATH, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"  Saved norm stats to {OUT_NORM_STATS_PATH}")

    print(f"  Normalization elapsed: {time.time()-t0:.1f}s")
    return fused_norm


def validate(fused_norm):
    print("Validating output ...")
    print(f"  Shape: {fused_norm.shape}")
    print(f"  Dtype: {fused_norm.dtype}")
    print(f"  Mean per feature (first 5): {fused_norm.mean(axis=0)[:5]}")
    print(f"  Std per feature (first 5): {fused_norm.std(axis=0)[:5]}")
    print(f"  Min: {fused_norm.min():.4f}, Max: {fused_norm.max():.4f}")

    reloaded = np.load(OUT_FUSED_PATH, mmap_mode="r")
    assert reloaded.shape == fused_norm.shape, f"Shape mismatch: {reloaded.shape} vs {fused_norm.shape}"
    print("  Reload verification: PASS")
    print("\nDone! 6M fused feature matrix ready for Phase 0.5 diffusion training.")


def main():
    t_start = time.time()

    price, timestamps = load_price_features()

    print("[2/5] Loading embeddings ...")
    news = load_embeddings(NEWS_EMB_NPY_PATH, "news")
    crowd = load_embeddings(CROWD_EMB_NPY_PATH, "crowd")

    market_dyn, md_cols = load_market_dynamics()
    print(f"  Market dynamics actual dim: {len(md_cols)} (expected {DIM_MARKET_DYNAMICS})")

    quant, q_cols = load_quant_features()
    print(f"  Quant actual dim: {len(q_cols)} (expected {DIM_QUANT})")

    dim_md_actual = market_dyn.shape[1]
    dim_q_actual = quant.shape[1]
    dim_total_actual = DIM_PRICE + DIM_NEWS + DIM_CROWD + dim_md_actual + dim_q_actual

    global DIM_MARKET_DYNAMICS, DIM_QUANT, DIM_TOTAL
    DIM_MARKET_DYNAMICS = dim_md_actual
    DIM_QUANT = dim_q_actual
    DIM_TOTAL = dim_total_actual
    print(f"  Adjusted total features: {DIM_TOTAL}")

    fused = fuse_all(price, news, crowd, market_dyn, quant)
    fused_norm = normalize_and_save(fused, timestamps)
    validate(fused_norm)

    print(f"\nTotal elapsed: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
