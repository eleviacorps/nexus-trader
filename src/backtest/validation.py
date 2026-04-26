from __future__ import annotations

from typing import Any

import numpy as np

from config.project_config import FEATURE_DIM_CROWD, FEATURE_DIM_NEWS, FEATURE_DIM_PRICE, PRICE_FEATURE_COLUMNS


def infer_feature_names(feature_dim: int) -> list[str]:
    price_names = list(PRICE_FEATURE_COLUMNS[: min(FEATURE_DIM_PRICE, feature_dim)])
    remaining = feature_dim - len(price_names)
    names = price_names
    if remaining > 0:
        news_count = min(FEATURE_DIM_NEWS, remaining)
        names.extend([f"news_{idx}" for idx in range(news_count)])
        remaining -= news_count
    if remaining > 0:
        crowd_count = min(FEATURE_DIM_CROWD, remaining)
        names.extend([f"crowd_{idx}" for idx in range(crowd_count)])
        remaining -= crowd_count
    if remaining > 0:
        names.extend([f"feature_{idx}" for idx in range(len(names), feature_dim)])
    return names[:feature_dim]


def analyze_timestamp_monotonicity(timestamps: np.ndarray) -> dict[str, Any]:
    values = np.asarray(timestamps).astype("datetime64[ns]")
    if values.size == 0:
        return {"sample_count": 0, "strictly_increasing": True, "duplicate_count": 0, "backward_jump_count": 0}
    diffs = np.diff(values.astype("int64"))
    duplicate_count = int((diffs == 0).sum())
    backward_jump_count = int((diffs < 0).sum())
    return {
        "sample_count": int(values.size),
        "strictly_increasing": bool(duplicate_count == 0 and backward_jump_count == 0),
        "duplicate_count": duplicate_count,
        "backward_jump_count": backward_jump_count,
        "min_step_ns": int(diffs.min()) if diffs.size else 0,
        "max_step_ns": int(diffs.max()) if diffs.size else 0,
    }


def analyze_feature_target_correlations(
    fused_features: np.ndarray,
    target_values: np.ndarray,
    *,
    feature_names: list[str] | None = None,
    suspicious_threshold: float = 0.4,
    critical_threshold: float = 0.98,
) -> dict[str, Any]:
    features = np.asarray(fused_features, dtype=np.float32)
    targets = np.asarray(target_values, dtype=np.float32).reshape(-1)
    if features.ndim == 3:
        features = features[:, -1, :]
    if features.ndim != 2:
        raise ValueError("Expected a 2D or 3D feature tensor.")
    if len(features) != len(targets):
        limit = min(len(features), len(targets))
        features = features[:limit]
        targets = targets[:limit]
    names = feature_names or infer_feature_names(features.shape[1])
    results = []
    centered_targets = targets - targets.mean() if len(targets) else targets
    target_scale = float(np.sqrt(np.mean(centered_targets ** 2))) if len(targets) else 0.0
    for idx in range(features.shape[1]):
        values = features[:, idx]
        finite_mask = np.isfinite(values) & np.isfinite(targets)
        if finite_mask.sum() < 8:
            corr = 0.0
        else:
            x = values[finite_mask] - values[finite_mask].mean()
            y = targets[finite_mask] - targets[finite_mask].mean()
            x_scale = float(np.sqrt(np.mean(x ** 2)))
            y_scale = float(np.sqrt(np.mean(y ** 2)))
            corr = float(np.mean(x * y) / max(1e-8, x_scale * y_scale)) if x_scale > 0 and y_scale > 0 else 0.0
        results.append({"feature": names[idx] if idx < len(names) else f"feature_{idx}", "correlation": corr})
    suspicious = [item for item in results if abs(item["correlation"]) >= suspicious_threshold]
    critical = [item for item in results if abs(item["correlation"]) >= critical_threshold]
    suspicious.sort(key=lambda item: abs(item["correlation"]), reverse=True)
    critical.sort(key=lambda item: abs(item["correlation"]), reverse=True)
    return {
        "feature_count": int(features.shape[1]),
        "sample_count": int(len(targets)),
        "target_scale": target_scale,
        "suspicious_threshold": float(suspicious_threshold),
        "critical_threshold": float(critical_threshold),
        "top_correlations": suspicious[:20],
        "critical_correlations": critical,
    }


def analyze_recursive_window_consistency(fused_tensor: np.ndarray, *, tolerance: float = 1e-6) -> dict[str, Any]:
    values = np.asarray(fused_tensor, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("Expected fused tensor with shape (samples, sequence_len, features).")
    if values.shape[0] < 2 or values.shape[1] < 2:
        return {
            "sample_count": int(values.shape[0]),
            "sequence_len": int(values.shape[1]) if values.ndim >= 2 else 0,
            "feature_dim": int(values.shape[2]) if values.ndim == 3 else 0,
            "max_abs_mismatch": 0.0,
            "mean_abs_mismatch": 0.0,
            "mismatch_rate": 0.0,
            "consistent": True,
        }
    overlap_left = values[:-1, 1:, :]
    overlap_right = values[1:, :-1, :]
    mismatch = np.abs(overlap_left - overlap_right)
    max_abs = float(np.nanmax(mismatch))
    mean_abs = float(np.nanmean(mismatch))
    mismatch_rate = float((mismatch > tolerance).mean())
    return {
        "sample_count": int(values.shape[0]),
        "sequence_len": int(values.shape[1]),
        "feature_dim": int(values.shape[2]),
        "max_abs_mismatch": max_abs,
        "mean_abs_mismatch": mean_abs,
        "mismatch_rate": mismatch_rate,
        "consistent": bool(max_abs <= tolerance),
        "tolerance": float(tolerance),
    }
