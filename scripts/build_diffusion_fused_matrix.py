"""Build the 405K-row fused feature matrix for diffusion model training.

Aligns v21_features.parquet (405K x 36 price columns at 15-min bars)
with mean-pooled news/crowd embeddings (6M x 32 each at 1-min bars,
aggregated to 15-min windows) to produce a (405434, 100) float32 array.

Usage:
    python scripts/build_diffusion_fused_matrix.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import (
    EMBEDDINGS_DIR,
    FEATURES_DIR,
    PRICE_FEATURE_COLUMNS,
)

V21_FEATURES_PATH = FEATURES_DIR / "v21_features.parquet"
NEWS_EMB_NPY_PATH = EMBEDDINGS_DIR / "news_embeddings.npy"
CROWD_EMB_NPY_PATH = EMBEDDINGS_DIR / "crowd_embeddings.npy"
NEWS_INDEX_PATH = EMBEDDINGS_DIR / "news_emb_index.parquet"
CROWD_INDEX_PATH = EMBEDDINGS_DIR / "crowd_emb_index.parquet"

OUT_FUSED_PATH = FEATURES_DIR / "diffusion_fused_405k.npy"
OUT_TARGETS_PATH = FEATURES_DIR / "diffusion_targets_405k.npy"
OUT_TIMESTAMPS_PATH = FEATURES_DIR / "diffusion_timestamps_405k.npy"
OUT_NORM_STATS_PATH = PROJECT_ROOT / "config" / "diffusion_norm_stats.json"

FEATURE_DIM_PRICE = len(PRICE_FEATURE_COLUMNS)
FEATURE_DIM_NEWS = 32
FEATURE_DIM_CROWD = 32
FEATURE_DIM_TOTAL = FEATURE_DIM_PRICE + FEATURE_DIM_NEWS + FEATURE_DIM_CROWD


def _floor_to_15min(ts_array):
    import pandas as pd
    series = pd.DatetimeIndex(pd.to_datetime(ts_array, utc=True))
    floored = series.floor("15min")
    return floored


def load_price_features():
    print(f"[1/5] Loading price features from {V21_FEATURES_PATH} ...")
    table = pq.read_table(V21_FEATURES_PATH)
    col_names = table.schema.names

    dt_col = "datetime" if "datetime" in col_names else col_names[-1]
    datetime_arr = table.column(dt_col).to_pylist()
    print(f"  Datetime column: '{dt_col}', first={datetime_arr[0]}")

    price_table = table.select([c for c in PRICE_FEATURE_COLUMNS if c in col_names])
    price_matrix = np.stack([price_table.column(c).to_numpy().astype(np.float32) for c in price_table.schema.names], axis=1)

    missing = [c for c in PRICE_FEATURE_COLUMNS if c not in col_names]
    if missing:
        print(f"  WARNING: Missing price columns (will zero-fill): {missing}")
        full = np.zeros((len(price_matrix), FEATURE_DIM_PRICE), dtype=np.float32)
        present = [c for c in PRICE_FEATURE_COLUMNS if c in col_names]
        for i, c in enumerate(PRICE_FEATURE_COLUMNS):
            if c in col_names:
                j = present.index(c)
                full[:, i] = price_matrix[:, j]
        price_matrix = full

    print(f"  Price matrix shape: {price_matrix.shape}")
    return price_matrix, datetime_arr


def load_and_aggregate_embeddings(npy_path, index_path, label):
    print(f"[2/5] Loading {label} embeddings from {npy_path} ...")
    embeddings = np.load(npy_path, mmap_mode="r")
    print(f"  Raw shape: {embeddings.shape}")

    print(f"  Loading timestamp index from {index_path} ...")
    idx_table = pq.read_table(index_path, columns=["timestamp"])
    raw_ts = idx_table.column("timestamp").to_pylist()

    print(f"  Flooring timestamps to 15-min windows ...")
    floored = _floor_to_15min(raw_ts)

    import pandas as pd
    emb_df = pd.DataFrame({"window": floored})
    emb_df["row_idx"] = np.arange(len(emb_df))

    print(f"  Grouping by 15-min window and mean-pooling ...")
    grouped = emb_df.groupby("window")["row_idx"].apply(list)

    n_windows = len(grouped)
    dim = embeddings.shape[1]
    aggregated = np.zeros((n_windows, dim), dtype=np.float32)
    timestamps = []

    for i, (window_ts, row_indices) in enumerate(grouped.items()):
        indices = np.array(row_indices, dtype=np.int64)
        indices = indices[indices < len(embeddings)]
        if len(indices) > 0:
            aggregated[i] = embeddings[indices].mean(axis=0)
        timestamps.append(window_ts)

    print(f"  Aggregated {label}: {aggregated.shape}, windows: {len(timestamps)}")
    return aggregated, timestamps


def align_and_fuse(price_matrix, price_timestamps, news_agg, news_ts, crowd_agg, crowd_ts):
    print("[3/5] Aligning price features with aggregated embeddings ...")

    import pandas as pd

    price_df = pd.DataFrame({"dt": pd.to_datetime(price_timestamps, utc=True)})
    price_df["price_idx"] = np.arange(len(price_df))

    news_df = pd.DataFrame({"dt": pd.to_datetime(news_ts, utc=True), "news_idx": np.arange(len(news_agg))})
    crowd_df = pd.DataFrame({"dt": pd.to_datetime(crowd_ts, utc=True), "crowd_idx": np.arange(len(crowd_agg))})

    merged = price_df.merge(news_df, on="dt", how="left").merge(crowd_df, on="dt", how="left")

    n_price = len(price_matrix)
    fused = np.zeros((n_price, FEATURE_DIM_TOTAL), dtype=np.float32)
    fused[:, :FEATURE_DIM_PRICE] = price_matrix

    news_filled = 0
    crowd_filled = 0
    for i in range(n_price):
        row = merged.iloc[i]
        if not np.isnan(row.get("news_idx", np.nan)):
            idx = int(row["news_idx"])
            if idx < len(news_agg):
                fused[i, FEATURE_DIM_PRICE:FEATURE_DIM_PRICE + FEATURE_DIM_NEWS] = news_agg[idx]
                news_filled += 1
        if not np.isnan(row.get("crowd_idx", np.nan)):
            idx = int(row["crowd_idx"])
            if idx < len(crowd_agg):
                fused[i, FEATURE_DIM_PRICE + FEATURE_DIM_NEWS:] = crowd_agg[idx]
                crowd_filled += 1

    print(f"  Fused matrix shape: {fused.shape}")
    print(f"  News alignment: {news_filled}/{n_price} rows")
    print(f"  Crowd alignment: {crowd_filled}/{n_price} rows")

    nan_count = np.isnan(fused).sum()
    inf_count = np.isinf(fused).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"  WARNING: {nan_count} NaN, {inf_count} Inf values — replacing with 0")
        fused = np.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)

    return fused


def compute_targets(price_matrix):
    return_1 = price_matrix[:, PRICE_FEATURE_COLUMNS.index("return_1")]
    targets = return_1.astype(np.float32)
    return targets


def normalize_and_save(fused, targets, price_timestamps):
    print("[4/5] Normalizing and saving ...")

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
        "feature_dim_price": FEATURE_DIM_PRICE,
        "feature_dim_news": FEATURE_DIM_NEWS,
        "feature_dim_crowd": FEATURE_DIM_CROWD,
        "feature_dim_total": FEATURE_DIM_TOTAL,
        "n_rows": int(fused_norm.shape[0]),
    }

    print(f"  Saving fused matrix to {OUT_FUSED_PATH} ...")
    np.save(OUT_FUSED_PATH, fused_norm)

    print(f"  Saving targets to {OUT_TARGETS_PATH} ...")
    np.save(OUT_TARGETS_PATH, targets)

    ts_strings = [str(t) for t in price_timestamps]
    print(f"  Saving timestamps to {OUT_TIMESTAMPS_PATH} ...")
    np.save(OUT_TIMESTAMPS_PATH, np.array(ts_strings))

    OUT_NORM_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_NORM_STATS_PATH, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"  Saved norm stats to {OUT_NORM_STATS_PATH}")

    return fused_norm


def validate(fused_norm):
    print("[5/5] Validating output ...")
    print(f"  Shape: {fused_norm.shape}")
    print(f"  Dtype: {fused_norm.dtype}")
    print(f"  Mean per feature (should be ~0): {fused_norm.mean(axis=0)[:5]}")
    print(f"  Std per feature (should be ~1): {fused_norm.std(axis=0)[:5]}")
    print(f"  Min: {fused_norm.min():.4f}, Max: {fused_norm.max():.4f}")

    reloaded = np.load(OUT_FUSED_PATH, mmap_mode="r")
    assert reloaded.shape == fused_norm.shape, f"Shape mismatch: {reloaded.shape} vs {fused_norm.shape}"
    print(f"  Reload verification: PASS")

    targets = np.load(OUT_TARGETS_PATH, mmap_mode="r")
    print(f"  Targets shape: {targets.shape}, mean: {targets.mean():.6f}, std: {targets.std():.6f}")

    print("\nDone! Fused feature matrix ready for diffusion training.")


def main():
    price_matrix, price_timestamps = load_price_features()
    news_agg, news_ts = load_and_aggregate_embeddings(NEWS_EMB_NPY_PATH, NEWS_INDEX_PATH, "news")
    crowd_agg, crowd_ts = load_and_aggregate_embeddings(CROWD_EMB_NPY_PATH, CROWD_INDEX_PATH, "crowd")

    fused = align_and_fuse(price_matrix, price_timestamps, news_agg, news_ts, crowd_agg, crowd_ts)
    targets = compute_targets(price_matrix)
    fused_norm = normalize_and_save(fused, targets, price_timestamps)
    validate(fused_norm)


if __name__ == "__main__":
    main()
