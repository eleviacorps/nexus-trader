from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import V20_DENOISED_OHLCV_PATH, V20_FEATURES_METADATA_PATH, V20_FEATURES_PATH, V20_HMM_MODEL_PATH
from src.v12.bar_consistent_features import load_default_raw_bars
from src.v20.feature_builder import build_v20_feature_frame, save_feature_metadata
from src.v20.regime_detector import RegimeDetector
from src.v20.wavelet_denoiser import WaveletDenoiser


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the V20 feature frame.")
    parser.add_argument("--interval", default="15min")
    parser.add_argument("--start")
    parser.add_argument("--end")
    args = parser.parse_args()

    raw = load_default_raw_bars(start=args.start, end=args.end)
    if str(args.interval).lower() not in {"1min", "1m"}:
        raw = raw.resample(str(args.interval)).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    denoised = WaveletDenoiser().fit_transform(raw)
    V20_DENOISED_OHLCV_PATH.parent.mkdir(parents=True, exist_ok=True)
    denoised.to_parquet(V20_DENOISED_OHLCV_PATH)
    detector = RegimeDetector.load(V20_HMM_MODEL_PATH) if V20_HMM_MODEL_PATH.exists() else None
    features, metadata = build_v20_feature_frame(raw, hmm_detector=detector)
    V20_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(V20_FEATURES_PATH)
    save_feature_metadata(V20_FEATURES_METADATA_PATH, metadata)
    print(f"saved={V20_FEATURES_PATH}")
    print(f"feature_count={metadata['feature_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
