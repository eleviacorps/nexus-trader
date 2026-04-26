from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config.project_config import LEGACY_PRICE_FEATURES_PARQUET, OUTPUTS_V17_DIR, V14_BRANCH_FEATURES_PATH, V17_MMM_FEATURES_PATH
from src.v17.mmm import MultifractalMarketMemory


def main() -> int:
    archive = pd.read_parquet(V14_BRANCH_FEATURES_PATH, columns=["timestamp"])
    candidate_timestamps = (
        pd.to_datetime(archive["timestamp"], utc=True, errors="coerce")
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    feature_frame = pd.read_parquet(LEGACY_PRICE_FEATURES_PARQUET, columns=["return_1", "atr_pct"])
    feature_frame.index = pd.to_datetime(feature_frame.index, utc=False, errors="coerce")
    feature_frame = feature_frame.sort_index().loc[~feature_frame.index.isna()].copy()
    builder = MultifractalMarketMemory(window=252)
    rows: list[dict[str, object]] = []
    for timestamp in candidate_timestamps:
        key = pd.Timestamp(timestamp).tz_convert(None)
        window = feature_frame.loc[:key].tail(builder.window)
        if len(window) < 30:
            result = {
                "hurst_overall": 0.5,
                "hurst_positive": 0.5,
                "hurst_negative": 0.5,
                "hurst_asymmetry": 0.0,
                "market_memory_regime": "random_walk",
            }
        else:
            result = builder.compute_all(
                window["return_1"].fillna(0.0).to_numpy(dtype=float),
                window["atr_pct"].fillna(0.0).to_numpy(dtype=float),
            )
        rows.append({"timestamp": pd.Timestamp(timestamp).isoformat()} | result)
    enriched = pd.DataFrame(rows)
    OUTPUTS_V17_DIR.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(V17_MMM_FEATURES_PATH, index=False)

    summary = {
        "rows": int(len(enriched)),
        "timestamp_min": str(enriched["timestamp"].min()) if len(enriched) else None,
        "timestamp_max": str(enriched["timestamp"].max()) if len(enriched) else None,
        "hurst_overall_mean": round(float(enriched["hurst_overall"].mean()) if len(enriched) else 0.5, 6),
        "hurst_positive_mean": round(float(enriched["hurst_positive"].mean()) if len(enriched) else 0.5, 6),
        "hurst_negative_mean": round(float(enriched["hurst_negative"].mean()) if len(enriched) else 0.5, 6),
        "hurst_asymmetry_mean": round(float(enriched["hurst_asymmetry"].mean()) if len(enriched) else 0.0, 6),
    }
    summary_path = OUTPUTS_V17_DIR / "mmm_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(str(V17_MMM_FEATURES_PATH), flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
