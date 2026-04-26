from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import OUTPUTS_V20_DIR, V20_FEATURES_PATH, V20_SJD_DATASET_PATH
from src.v20.macro_features import MACRO_FEATURE_COLS
from src.v20.sjd_v20 import rule_based_sjd_labels


def _balanced_resample(frame: pd.DataFrame, target_rows: int, seed: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    rng = np.random.default_rng(seed)
    working = frame.reset_index(drop=False).reset_index(drop=True)
    per_regime_target = max(target_rows // 6, 1)
    pieces: list[pd.DataFrame] = []
    states = pd.to_numeric(working.get("hmm_state"), errors="coerce").fillna(-1).astype(int)
    for regime in range(6):
        group = working.loc[states == regime].copy()
        if group.empty:
            continue
        replace = len(group) < per_regime_target
        positions = rng.choice(np.arange(len(group)), size=per_regime_target, replace=replace)
        pieces.append(group.iloc[positions].copy())
    combined = pd.concat(pieces, axis=0, ignore_index=True) if pieces else working.head(0).copy()
    if len(combined) < target_rows and not combined.empty:
        extra_positions = rng.choice(np.arange(len(combined)), size=target_rows - len(combined), replace=True)
        combined = pd.concat([combined, combined.iloc[extra_positions].copy()], axis=0, ignore_index=True)
    return combined


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the V20 SJD dataset from the V20 feature frame.")
    parser.add_argument("--target-rows", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20)
    args = parser.parse_args()

    if not V20_FEATURES_PATH.exists():
        raise SystemExit(f"Missing V20 feature frame at {V20_FEATURES_PATH}. Run build_v20_features.py first.")

    feature_frame = pd.read_parquet(V20_FEATURES_PATH).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    labels = rule_based_sjd_labels(feature_frame)
    feature_columns = [
        col
        for col in feature_frame.select_dtypes(include=["number"]).columns
        if not str(col).startswith("target_") and not str(col).startswith("future_return_")
    ]
    dataset = pd.concat([feature_frame[feature_columns].copy(), labels], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    balanced = _balanced_resample(dataset, target_rows=max(int(args.target_rows), 1), seed=int(args.seed))
    balanced["macro_feature_count"] = len([col for col in MACRO_FEATURE_COLS if col in balanced.columns])
    balanced["source_unique_rows"] = int(len(dataset))
    V20_SJD_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    balanced.to_parquet(V20_SJD_DATASET_PATH)
    report = {
        "dataset_rows": int(len(balanced)),
        "source_unique_rows": int(len(dataset)),
        "target_rows_requested": int(args.target_rows),
        "feature_count": int(len(feature_columns)),
        "regime_counts": {
            str(int(k)): int(v)
            for k, v in balanced["hmm_state"].astype(int).value_counts().sort_index().to_dict().items()
        },
        "stance_counts": {str(k): int(v) for k, v in balanced["stance"].value_counts().to_dict().items()},
        "confidence_counts": {str(k): int(v) for k, v in balanced["confidence"].value_counts().to_dict().items()},
        "macro_feature_count": int(len([col for col in MACRO_FEATURE_COLS if col in balanced.columns])),
        "note": "Balanced dataset is resampled with replacement locally to approximate the prompt target when raw unique rows are insufficient.",
    }
    report_path = OUTPUTS_V20_DIR / "sjd_dataset_v20_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(V20_SJD_DATASET_PATH), **report}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
