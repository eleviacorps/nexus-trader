from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import V20_HMM_MODEL_PATH, V20_REGIME_LABELS_PATH
from src.v12.bar_consistent_features import load_default_raw_bars
from src.v20.macro_features import compute_macro_features
from src.v20.regime_detector import train_hmm


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the V20 6-state HMM regime detector.")
    parser.add_argument("--interval", default="15min")
    parser.add_argument("--start")
    parser.add_argument("--end")
    args = parser.parse_args()

    raw = load_default_raw_bars(start=args.start, end=args.end)
    if str(args.interval).lower() not in {"1min", "1m"}:
        raw = raw.resample(str(args.interval)).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    close = pd.to_numeric(raw["close"], errors="coerce").ffill().bfill()
    volume = pd.to_numeric(raw["volume"], errors="coerce").ffill().bfill()
    macro = compute_macro_features(raw)
    frame = pd.DataFrame(index=raw.index)
    frame["log_return"] = np.log(close).diff().fillna(0.0)
    frame["realized_vol_20"] = pd.to_numeric(macro["macro_realized_vol_20"], errors="coerce").fillna(0.0)
    frame["volume_zscore"] = ((volume - volume.rolling(96, min_periods=12).mean()) / volume.rolling(96, min_periods=12).std(ddof=0).replace(0.0, pd.NA)).fillna(0.0)
    frame["macro_vol_regime_class"] = pd.to_numeric(macro["macro_vol_regime_class"], errors="coerce").fillna(0.0)
    frame["macro_jump_flag"] = pd.to_numeric(macro["macro_jump_flag"], errors="coerce").fillna(0.0)
    detector, _, _ = train_hmm(frame)
    labels = detector.transform(frame)
    detector.save(V20_HMM_MODEL_PATH)
    V20_REGIME_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(V20_REGIME_LABELS_PATH)
    print(f"saved_model={V20_HMM_MODEL_PATH}")
    print(f"saved_labels={V20_REGIME_LABELS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
