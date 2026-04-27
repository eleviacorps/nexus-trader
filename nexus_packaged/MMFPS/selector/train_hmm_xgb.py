"""Train HMM regime model + XGBoost path scorer from selector dataset chunks.

Expected chunk format (.npz):
- contexts: (B, T, C)
- paths:    (B, N, H)
- futures:  (B, H)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import xgboost as xgb
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import mean_squared_error


def _resolve_data_dir(user_dir: str | None) -> Path:
    if user_dir:
        data_dir = Path(user_dir).expanduser().resolve()
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        return data_dir

    root = Path(__file__).resolve().parents[3]
    preferred = root / "nexus_packaged" / "MMFPS" / "selector" / "dataset"
    fallback = root / "nexus_packaged" / "outputs" / "MMFPS" / "selector_data"

    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        "No dataset directory found. Checked:\n"
        f"- {preferred}\n"
        f"- {fallback}"
    )


def _iter_chunks(data_dir: Path, max_chunks: int | None) -> Iterable[Path]:
    chunks = sorted(data_dir.glob("chunk_*.npz"))
    if not chunks:
        chunks = sorted(data_dir.glob("*.npz"))
    if not chunks:
        raise FileNotFoundError(f"No .npz chunks found in {data_dir}")
    if max_chunks is not None and max_chunks > 0:
        chunks = chunks[: max_chunks]
    return chunks


def _compute_path_features(paths: np.ndarray) -> np.ndarray:
    """Compute per-path features.

    Input:
    - paths: (B, N, H)

    Output:
    - features: (B*N, 6) in order:
      [mean_return, volatility, max_drawdown, trend_strength, final_return, range]
    """
    bsz, n_paths, _ = paths.shape
    flat = paths.reshape(bsz * n_paths, -1).astype(np.float32)

    prev = flat[:, :-1]
    nxt = flat[:, 1:]
    ret = (nxt - prev) / (np.abs(prev) + 1e-6)

    mean_return = ret.mean(axis=1)
    volatility = ret.std(axis=1)

    cumulative = np.cumsum(ret, axis=1)
    running_max = np.maximum.accumulate(cumulative, axis=1)
    drawdown = cumulative - running_max
    max_drawdown = drawdown.min(axis=1)

    trend_strength = np.mean(np.abs(np.diff(flat, axis=1)), axis=1)
    final_return = (flat[:, -1] - flat[:, 0]) / (np.abs(flat[:, 0]) + 1e-6)
    prange = flat.max(axis=1) - flat.min(axis=1)

    feats = np.stack(
        [mean_return, volatility, max_drawdown, trend_strength, final_return, prange],
        axis=1,
    ).astype(np.float32)
    return feats


def _prepare_hmm_training_array(
    chunks: list[Path],
) -> tuple[np.ndarray, list[int], np.ndarray]:
    series_list: list[np.ndarray] = []
    lengths: list[int] = []

    for chunk_path in chunks:
        arr = np.load(chunk_path)
        contexts = arr["contexts"].astype(np.float32)
        # Per requirement: extract from channel-0 context sequence.
        seq = contexts[:, :, 0]
        series_list.append(seq)
        lengths.extend([seq.shape[1]] * seq.shape[0])

    all_series = np.concatenate(series_list, axis=0)
    flat = all_series.reshape(-1, 1).astype(np.float64)
    return flat, lengths, all_series


def _prepare_xgb_training_arrays(
    chunks: list[Path],
    max_paths_per_sample: int | None,
    max_path_samples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    feat_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    total_rows = 0

    for chunk_path in chunks:
        arr = np.load(chunk_path)
        paths = arr["paths"].astype(np.float32)    # (B, N, H)
        futures = arr["futures"].astype(np.float32)  # (B, H)

        if max_paths_per_sample is not None and max_paths_per_sample > 0 and paths.shape[1] > max_paths_per_sample:
            idx = rng.choice(paths.shape[1], size=max_paths_per_sample, replace=False)
            paths = paths[:, idx, :]

        features = _compute_path_features(paths)

        distance = ((paths - futures[:, None, :]) ** 2).mean(axis=-1)
        score = (-distance).reshape(-1).astype(np.float32)

        feat_parts.append(features)
        target_parts.append(score)
        total_rows += features.shape[0]
        print(f"  {chunk_path.name}: +{features.shape[0]:,} rows (total={total_rows:,})")

    X = np.concatenate(feat_parts, axis=0).astype(np.float32)
    y = np.concatenate(target_parts, axis=0).astype(np.float32)

    if max_path_samples is not None and max_path_samples > 0 and X.shape[0] > max_path_samples:
        keep = rng.choice(X.shape[0], size=max_path_samples, replace=False)
        X = X[keep]
        y = y[keep]
        print(f"Subsampled to {X.shape[0]:,} path rows for XGBoost training")

    return X, y


def main() -> int:
    parser = argparse.ArgumentParser(description="Train HMM + XGBoost models for selector features")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing selector .npz chunks")
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Output model directory (default: nexus_packaged/MMFPS/models)",
    )
    parser.add_argument("--max-chunks", type=int, default=0, help="Limit number of chunks (0 = all)")
    parser.add_argument("--max-paths-per-sample", type=int, default=128, help="Cap paths per sample for XGB training")
    parser.add_argument("--max-path-samples", type=int, default=0, help="Cap total path rows for XGB training (0 = all)")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--hmm-components", type=int, default=4)
    parser.add_argument("--hmm-iters", type=int, default=100)

    parser.add_argument("--xgb-n-estimators", type=int, default=200)
    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--xgb-random-state", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    data_dir = _resolve_data_dir(args.data_dir)
    root = Path(__file__).resolve().parents[3]
    models_dir = (
        Path(args.models_dir).expanduser().resolve()
        if args.models_dir
        else (root / "nexus_packaged" / "MMFPS" / "models")
    )
    models_dir.mkdir(parents=True, exist_ok=True)

    hmm_out = models_dir / "hmm_regime.pkl"
    xgb_out = models_dir / "xgb_path_scorer.json"

    max_chunks = args.max_chunks if args.max_chunks > 0 else None
    chunks = list(_iter_chunks(data_dir, max_chunks))

    print("=== Training HMM + XGBoost for selector ===")
    print(f"data_dir={data_dir}")
    print(f"models_dir={models_dir}")
    print(f"chunks={len(chunks)}")

    # --- HMM ---
    print("\n[1/2] Training HMM regime model...")
    hmm_X, hmm_lengths, hmm_series = _prepare_hmm_training_array(chunks)
    print(f"HMM samples={hmm_X.shape[0]:,}, sequences={len(hmm_lengths):,}, seq_len={hmm_series.shape[1]}")

    hmm_model = GaussianHMM(
        n_components=args.hmm_components,
        covariance_type="diag",
        n_iter=args.hmm_iters,
        random_state=args.seed,
    )
    hmm_model.fit(hmm_X, lengths=hmm_lengths)
    joblib.dump(hmm_model, hmm_out)

    hmm_states = hmm_model.predict(hmm_X, lengths=hmm_lengths)
    state_hist = np.bincount(hmm_states, minlength=args.hmm_components).astype(np.float64)
    state_hist = state_hist / max(state_hist.sum(), 1.0)
    print(f"Saved HMM model -> {hmm_out}")
    print(f"HMM state distribution={np.round(state_hist, 4).tolist()}")

    # --- XGBoost ---
    print("\n[2/2] Training XGBoost path scorer...")
    max_path_samples = args.max_path_samples if args.max_path_samples > 0 else None
    max_paths_per_sample = args.max_paths_per_sample if args.max_paths_per_sample > 0 else None
    X, y = _prepare_xgb_training_arrays(
        chunks=chunks,
        max_paths_per_sample=max_paths_per_sample,
        max_path_samples=max_path_samples,
        seed=args.seed,
    )
    print(f"XGB train rows={X.shape[0]:,}, features={X.shape[1]}")

    xgb_model = xgb.XGBRegressor(
        n_estimators=args.xgb_n_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample_bytree,
        objective="reg:squarederror",
        random_state=args.xgb_random_state,
        n_jobs=-1,
    )
    xgb_model.fit(X, y)
    xgb_model.save_model(xgb_out)

    pred = xgb_model.predict(X[: min(200_000, X.shape[0])])
    y_ref = y[: pred.shape[0]]
    rmse = float(np.sqrt(mean_squared_error(y_ref, pred)))
    fi = xgb_model.feature_importances_.astype(np.float64)
    fi = fi / max(fi.sum(), 1e-12)
    print(f"Saved XGBoost model -> {xgb_out}")
    print(f"XGB train RMSE(sample)={rmse:.6f}")
    print(f"XGB feature importance (normalized)={np.round(fi, 4).tolist()}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
