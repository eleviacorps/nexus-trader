from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.project_config import (
    FUSED_FEATURE_MATRIX_PATH,
    TFT_CHECKPOINT_PATH,
    V10_BRANCH_ARCHIVE_PATH,
    V10_BRANCH_FEATURES_PATH,
    V11_PCOP_STAGE10_MODEL_PATH,
    V11_PCOP_STAGE5_MODEL_PATH,
    V11_RESEARCH_BACKTEST_PATH,
    V11_SETL_MODEL_PATH,
    V12_FEATURE_CONSISTENCY_REPORT_PATH,
    V9_MEMORY_BANK_ENCODER_PATH,
)
from src.v12.bar_consistent_features import (
    DEFAULT_ARCHIVE_FEATURE_PATH,
    DEFAULT_RAW_BAR_PATH,
    PRICE_FEATURE_COLUMNS,
    align_feature_frames,
    compute_bar_consistent_features,
    load_default_archive_features,
    load_default_raw_bars,
    _to_utc_timestamp,
)

try:
    from scipy.stats import ks_2samp  # type: ignore
except Exception:  # pragma: no cover
    ks_2samp = None


REQUIRED_V12_ARTIFACTS: tuple[Path, ...] = (
    V10_BRANCH_ARCHIVE_PATH.with_name("branch_archive_v10_full.parquet"),
    V10_BRANCH_FEATURES_PATH.with_name("branch_features_v10_full.parquet"),
    V11_RESEARCH_BACKTEST_PATH.with_name("research_backtest_full.json"),
    V11_SETL_MODEL_PATH,
    V11_PCOP_STAGE5_MODEL_PATH,
    V11_PCOP_STAGE10_MODEL_PATH,
    V9_MEMORY_BANK_ENCODER_PATH,
    TFT_CHECKPOINT_PATH,
    FUSED_FEATURE_MATRIX_PATH,
)


def verify_v12_artifacts(paths: tuple[Path, ...] = REQUIRED_V12_ARTIFACTS) -> dict[str, Any]:
    artifacts = []
    missing = []
    for path in paths:
        exists = path.exists()
        size = int(path.stat().st_size) if exists else 0
        record = {"path": str(path), "exists": exists, "size_bytes": size}
        artifacts.append(record)
        if not exists:
            missing.append(str(path))
    status = {
        "all_present": not missing,
        "missing": missing,
        "artifacts": artifacts,
    }
    return status


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if np.allclose(left, left[0]) and np.allclose(right, right[0]):
        return 1.0 if np.allclose(left, right) else 0.0
    left_std = float(np.std(left))
    right_std = float(np.std(right))
    if left_std <= 1e-12 or right_std <= 1e-12:
        return 1.0 if np.allclose(left, right) else 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _ks_statistic(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    if ks_2samp is not None:
        return float(ks_2samp(left, right).statistic)
    left_sorted = np.sort(np.asarray(left, dtype=np.float64))
    right_sorted = np.sort(np.asarray(right, dtype=np.float64))
    values = np.sort(np.unique(np.concatenate([left_sorted, right_sorted])))
    if values.size == 0:
        return 0.0
    left_cdf = np.searchsorted(left_sorted, values, side="right") / max(len(left_sorted), 1)
    right_cdf = np.searchsorted(right_sorted, values, side="right") / max(len(right_sorted), 1)
    return float(np.max(np.abs(left_cdf - right_cdf)))


def _variance_ratio(left: np.ndarray, right: np.ndarray) -> float:
    left_var = float(np.var(left))
    right_var = float(np.var(right))
    return left_var / max(right_var, 1e-10)


def _recommend_fix(feature: str, correlation: float) -> str:
    if feature == "volume_ratio":
        return "Legacy archive uses zero-volume bars heavily; replace with causal fallback or drop from V12 model inputs."
    if feature.startswith("ema_") or feature in {"macd", "macd_sig", "macd_hist"}:
        return "Recompute from BCFE rolling state with consistent warmup and no archive-side backfill."
    if feature.startswith("rsi") or feature.startswith("stoch"):
        return "Use the BCFE causal oscillator path and verify early-window seeding."
    if feature in {"atr_pct", "bb_width", "bb_pct", "dist_to_high", "dist_to_low", "hh", "ll"}:
        return "Use BCFE rolling extrema/ATR path; current archive/live mismatch is likely window warmup drift."
    if feature.startswith("session_") or feature.endswith("_sin") or feature.endswith("_cos"):
        return "Timestamp-derived feature should already be stable; inspect timezone alignment if it drifts."
    if correlation < 0.95:
        return "Block this feature from V12 model training until BCFE/live replay agreement is restored."
    return "No fix required."


def compare_feature_frames(
    archive_features: pd.DataFrame,
    live_features: pd.DataFrame,
    *,
    pass_threshold: float = 0.95,
) -> dict[str, Any]:
    archive_aligned, live_aligned = align_feature_frames(archive_features, live_features)
    feature_results: dict[str, Any] = {}
    pass_features: list[str] = []
    fail_features: list[str] = []

    for feature in archive_aligned.columns:
        archive_col = archive_aligned[feature].to_numpy(dtype=np.float64)
        live_col = live_aligned[feature].to_numpy(dtype=np.float64)
        pearson = _pearson(archive_col, live_col)
        result = {
            "pearson_correlation": round(pearson, 6),
            "ks_statistic": round(_ks_statistic(archive_col, live_col), 6),
            "mean_absolute_difference": round(float(np.mean(np.abs(archive_col - live_col))), 6),
            "variance_ratio": round(_variance_ratio(archive_col, live_col), 6),
            "passes": bool(pearson >= pass_threshold),
            "recommended_fix": _recommend_fix(feature, pearson),
        }
        feature_results[feature] = result
        if result["passes"]:
            pass_features.append(feature)
        else:
            fail_features.append(feature)

    return {
        "row_count": int(len(archive_aligned)),
        "feature_count": int(len(feature_results)),
        "pass_threshold": float(pass_threshold),
        "pass_features": pass_features,
        "fail_features": fail_features,
        "features": feature_results,
    }


def _default_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    end = pd.Timestamp("2024-01-01 00:00:00+00:00")
    start = end - pd.Timedelta(days=90)
    return start, end


def run_feature_consistency_audit(
    *,
    raw_bars: pd.DataFrame | None = None,
    archive_features: pd.DataFrame | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    warmup_bars: int = 200,
    pass_threshold: float = 0.95,
) -> dict[str, Any]:
    window_start, window_end = _default_window() if start is None and end is None else (
        _to_utc_timestamp(start) if start is not None else None,
        _to_utc_timestamp(end) if end is not None else None,
    )

    raw = raw_bars if raw_bars is not None else load_default_raw_bars(start=window_start, end=window_end)
    archive = archive_features if archive_features is not None else load_default_archive_features(start=window_start, end=window_end)
    bcfe_archive = compute_bar_consistent_features(raw)[list(PRICE_FEATURE_COLUMNS)]
    start_at = max(int(warmup_bars) - 1, 0)
    live = bcfe_archive.iloc[start_at:].copy()
    legacy_report = compare_feature_frames(archive, live, pass_threshold=pass_threshold)

    bcfe_archive, bcfe_live = align_feature_frames(bcfe_archive, live)
    bcfe_report = compare_feature_frames(bcfe_archive, bcfe_live, pass_threshold=pass_threshold)

    return {
        "window": {
            "start": str(raw.index.min()) if len(raw) else str(window_start),
            "end": str(raw.index.max()) if len(raw) else str(window_end),
            "warmup_bars": int(warmup_bars),
            "raw_bar_count": int(len(raw)),
            "legacy_archive_path": str(DEFAULT_ARCHIVE_FEATURE_PATH),
            "raw_bar_path": str(DEFAULT_RAW_BAR_PATH),
        },
        "legacy_archive_vs_live": legacy_report,
        "bcfe_self_check": bcfe_report,
    }


def render_feature_consistency_summary(report: dict[str, Any]) -> str:
    lines = ["Feature".ljust(18) + " Corr    MAD      Pass"]
    lines.append("-" * 44)
    features = report.get("legacy_archive_vs_live", {}).get("features", {})
    for name in PRICE_FEATURE_COLUMNS:
        payload = features.get(name)
        if payload is None:
            continue
        lines.append(
            f"{name[:18].ljust(18)} {payload['pearson_correlation']:<7} {payload['mean_absolute_difference']:<8} {str(payload['passes'])}"
        )
    return "\n".join(lines)


def write_feature_consistency_report(report: dict[str, Any], path: Path = V12_FEATURE_CONSISTENCY_REPORT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
