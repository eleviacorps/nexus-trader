from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import V15_CPM_LABELS_PATH, V15_CPM_SUMMARY_PATH
from src.v12.bar_consistent_features import compute_bar_consistent_features, load_default_archive_features, load_default_raw_bars
from src.v15.cpm import ConditionalPredictabilityMapper


def _load_feature_frame():
    try:
        frame = load_default_archive_features()
        if {"return_1", "rsi_14", "ema_cross", "macd_hist", "bb_pct", "volume_ratio"} <= set(frame.columns):
            return frame
    except Exception:
        pass
    raw_bars = load_default_raw_bars()
    return compute_bar_consistent_features(raw_bars)


def main() -> int:
    mapper = ConditionalPredictabilityMapper()
    feature_frame = _load_feature_frame()
    labeled = mapper.label_archive(feature_frame)
    summary = mapper.summarize_distribution(labeled)

    V15_CPM_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(V15_CPM_LABELS_PATH)
    V15_CPM_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(str(V15_CPM_LABELS_PATH), flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
