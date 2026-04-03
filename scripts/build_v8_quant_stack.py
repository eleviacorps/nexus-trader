from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None

from config.project_config import (  # noqa: E402
    LEGACY_PRICE_FEATURES_CSV,
    LEGACY_PRICE_FEATURES_PARQUET,
    MARKET_DYNAMICS_LABELS_PATH,
    PRICE_FEATURES_CSV_FALLBACK,
    PRICE_FEATURES_PATH,
    QUANT_FEATURES_CSV_FALLBACK,
    QUANT_FEATURES_PATH,
    V8_ANALOG_CACHE_META_PATH,
    V8_ANALOG_CACHE_PATH,
    V8_FAIR_VALUE_FRAME_PATH,
    V8_FAIR_VALUE_MODEL_PATH,
    V8_GARCH_FRAME_PATH,
    V8_GARCH_MODEL_PATH,
    V8_HMM_FRAME_PATH,
    V8_HMM_MODEL_PATH,
    V8_QUANT_STACK_REPORT_PATH,
)
from src.pipeline.fusion import load_price_frame, merge_market_dynamics_features  # noqa: E402
from src.quant.hybrid import merge_quant_features  # noqa: E402
from src.v8.analog_retrieval import build_analog_cache  # noqa: E402
from src.v8.fair_value import build_fair_value_frame, summarize_fair_value_frame  # noqa: E402
from src.v8.garch_volatility import build_garch_like_frame, summarize_volatility_frame  # noqa: E402
from src.v8.hmm_regime import build_hmm_regime_frame, summarize_hmm_frame  # noqa: E402


def _resolve_first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    joined = ", ".join(str(path) for path in paths)
    raise FileNotFoundError(f"No artifact found in: {joined}")


def _read_frame(path: Path):
    if pd is None:
        raise ImportError("pandas is required to build the v8 quant stack.")
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
    if "timestamp" in frame.columns:
        frame = frame.set_index(pd.to_datetime(frame["timestamp"], errors="coerce")).drop(columns=["timestamp"])
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    return frame


def _write_frame(frame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path)
        return path
    except Exception:
        fallback = path.with_suffix(".csv")
        frame.to_csv(fallback)
        return fallback


def _load_base_frame():
    price_path = _resolve_first_existing(
        [PRICE_FEATURES_PATH, PRICE_FEATURES_CSV_FALLBACK, LEGACY_PRICE_FEATURES_PARQUET, LEGACY_PRICE_FEATURES_CSV]
    )
    frame = load_price_frame(price_path)
    if QUANT_FEATURES_PATH.exists():
        frame = merge_quant_features(frame, _read_frame(QUANT_FEATURES_PATH))
    elif QUANT_FEATURES_CSV_FALLBACK.exists():
        frame = merge_quant_features(frame, _read_frame(QUANT_FEATURES_CSV_FALLBACK))
    if MARKET_DYNAMICS_LABELS_PATH.exists():
        frame = merge_market_dynamics_features(frame, _read_frame(MARKET_DYNAMICS_LABELS_PATH))
    else:
        dynamics_csv = MARKET_DYNAMICS_LABELS_PATH.with_suffix(".csv")
        if dynamics_csv.exists():
            frame = merge_market_dynamics_features(frame, _read_frame(dynamics_csv))
    return frame


def main() -> int:
    frame = _load_base_frame()

    hmm_frame = build_hmm_regime_frame(frame)
    garch_frame = build_garch_like_frame(frame)
    fair_value_frame = build_fair_value_frame(frame)
    analog_stride = 120 if len(frame) > 1_000_000 else 30
    analog_cache = build_analog_cache(sample_stride=analog_stride)

    saved_hmm = _write_frame(hmm_frame, V8_HMM_FRAME_PATH)
    saved_garch = _write_frame(garch_frame, V8_GARCH_FRAME_PATH)
    saved_fair_value = _write_frame(fair_value_frame, V8_FAIR_VALUE_FRAME_PATH)

    V8_HMM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with V8_HMM_MODEL_PATH.open("wb") as handle:
        pickle.dump(
            {
                "provider": str(hmm_frame.attrs.get("provider", "unknown")),
                "state_mapping": list(hmm_frame.attrs.get("state_mapping", [])),
                "feature_path": str(saved_hmm),
            },
            handle,
        )
    with V8_GARCH_MODEL_PATH.open("wb") as handle:
        pickle.dump(
            {
                "provider": str(garch_frame.attrs.get("provider", "unknown")),
                "feature_path": str(saved_garch),
            },
            handle,
        )
    with V8_FAIR_VALUE_MODEL_PATH.open("wb") as handle:
        pickle.dump(
            {
                "feature_path": str(saved_fair_value),
            },
            handle,
        )

    V8_ANALOG_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        V8_ANALOG_CACHE_PATH,
        features=np.asarray(analog_cache.features, dtype=np.float32),
        future_paths=np.asarray(analog_cache.future_paths, dtype=np.float32),
        timestamps=np.asarray(analog_cache.timestamps).astype("datetime64[ns]").astype(np.int64),
        feature_mean=np.asarray(analog_cache.feature_mean, dtype=np.float32),
        feature_std=np.asarray(analog_cache.feature_std, dtype=np.float32),
        window_size=np.asarray([analog_cache.window_size], dtype=np.int32),
        sample_stride=np.asarray([analog_cache.sample_stride], dtype=np.int32),
    )

    report = {
        "rows": int(len(frame)),
        "hmm": summarize_hmm_frame(hmm_frame).__dict__,
        "volatility": summarize_volatility_frame(garch_frame).__dict__,
        "fair_value": summarize_fair_value_frame(fair_value_frame).__dict__,
        "analog_cache": {
            "rows": int(len(analog_cache.features)),
            "window_size": int(analog_cache.window_size),
            "sample_stride": int(analog_cache.sample_stride),
            "feature_dim": int(analog_cache.features.shape[1]) if analog_cache.features.ndim == 2 else 0,
            "future_dim": int(analog_cache.future_paths.shape[1]) if analog_cache.future_paths.ndim == 2 else 0,
        },
        "artifacts": {
            "hmm_frame_path": str(saved_hmm),
            "hmm_model_path": str(V8_HMM_MODEL_PATH),
            "garch_frame_path": str(saved_garch),
            "garch_model_path": str(V8_GARCH_MODEL_PATH),
            "fair_value_frame_path": str(saved_fair_value),
            "fair_value_model_path": str(V8_FAIR_VALUE_MODEL_PATH),
            "analog_cache_path": str(V8_ANALOG_CACHE_PATH),
        },
    }
    V8_ANALOG_CACHE_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    V8_ANALOG_CACHE_META_PATH.write_text(
        json.dumps(report["analog_cache"] | {"artifact_path": str(V8_ANALOG_CACHE_PATH)}, indent=2),
        encoding="utf-8",
    )
    V8_QUANT_STACK_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    V8_QUANT_STACK_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
