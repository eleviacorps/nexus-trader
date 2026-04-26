from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import V20_MACRO_FEATURES_PATH
from src.v12.bar_consistent_features import load_default_raw_bars
from src.v20.macro_features import compute_macro_features


def main() -> int:
    parser = argparse.ArgumentParser(description="Build V20 macro features from raw OHLCV or local proxies.")
    parser.add_argument("--interval", default="15min")
    parser.add_argument("--start")
    parser.add_argument("--end")
    args = parser.parse_args()

    raw = load_default_raw_bars(start=args.start, end=args.end)
    if str(args.interval).lower() not in {"1min", "1m"}:
        raw = raw.resample(str(args.interval)).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    macro = compute_macro_features(raw)
    V20_MACRO_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    macro.to_parquet(V20_MACRO_FEATURES_PATH)
    print(f"saved={V20_MACRO_FEATURES_PATH}")
    print(f"rows={len(macro)} cols={len(macro.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
