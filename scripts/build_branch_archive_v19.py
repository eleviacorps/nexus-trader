from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config.project_config import V10_BRANCH_ARCHIVE_FULL_PATH, V10_BRANCH_ARCHIVE_PATH, V19_BRANCH_ARCHIVE_PATH, V19_BRANCH_ARCHIVE_REPORT_PATH
from src.v19.context_sampler import load_context_source_frame


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
        if np.isnan(number) or np.isinf(number):
            return float(default)
        return number
    except Exception:
        return float(default)


def _derive_wltc_state(row: pd.Series) -> str:
    retail = abs(_safe_float(row.get("retail_impact"), 0.0))
    institutional = abs(_safe_float(row.get("institutional_impact"), 0.0))
    if retail > institutional * 1.2:
        return "retail_dominant"
    if institutional > retail * 1.2:
        return "institutional_dominant"
    return "balanced"


def build_branch_archive_v19(
    *,
    target_rows: int = 100_000,
    output_path: Path = V19_BRANCH_ARCHIVE_PATH,
    report_path: Path = V19_BRANCH_ARCHIVE_REPORT_PATH,
) -> dict[str, object]:
    base_path = V10_BRANCH_ARCHIVE_FULL_PATH if V10_BRANCH_ARCHIVE_FULL_PATH.exists() else V10_BRANCH_ARCHIVE_PATH
    base = pd.read_parquet(base_path)
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True, errors="coerce")
    source = load_context_source_frame().copy()
    source["timestamp"] = pd.to_datetime(source["timestamp"], utc=True, errors="coerce")
    extra_columns = [
        "timestamp",
        "macro_bias",
        "macro_shock",
        "news_bias",
        "news_intensity",
        "crowd_bias",
        "crowd_extreme",
        "consensus_score",
        "retail_impact",
        "institutional_impact",
        "algo_impact",
        "whale_impact",
        "quant_regime_strength",
        "quant_transition_risk",
        "quant_vol_realism",
        "quant_fair_value_z",
        "quant_route_confidence",
        "quant_trend_score",
        "hurst_overall",
        "hurst_positive",
        "hurst_negative",
        "hurst_asymmetry",
        "analog_confidence",
        "cone_realism",
        "contradiction_type",
    ]
    extra_columns = [column for column in extra_columns if column in source.columns]
    merged = pd.merge_asof(
        base.sort_values("timestamp"),
        source[extra_columns].sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("12h"),
    )
    merged["sample_id"] = pd.to_numeric(merged["sample_id"], errors="coerce").fillna(-1).astype(int)
    numeric_columns = merged.select_dtypes(include=["number", "bool"]).columns
    for column in numeric_columns:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    merged["mfg_disagreement"] = (
        merged.get("analog_disagreement", 0.0).abs().fillna(0.0) * 0.25
        + merged.get("crowd_extreme", 0.0).fillna(0.0) * 0.10
    )
    merged["setl_target_net_unit_pnl"] = merged.get("actual_final_return", 0.0).fillna(0.0)
    merged["cone_realism"] = merged.get("cone_realism", 1.0 - merged.get("path_error", 0.0).abs()).fillna(0.0)
    merged["analog_confidence"] = merged.get("analog_confidence", merged.get("leaf_analog_confidence", merged.get("analog_similarity", 0.0))).fillna(0.0)
    merged["wltc_state"] = merged.apply(_derive_wltc_state, axis=1)
    merged["contradiction_type"] = merged.get("contradiction_type", "mixed").fillna("mixed").astype(str)
    merged["branch_disagreement"] = merged.get("analog_disagreement", 0.0).fillna(0.0)
    merged["consensus_strength"] = merged.get("consensus_score", merged.get("branch_confidence", 0.5)).fillna(0.5)
    merged["analog_disagreement_v9"] = merged.get("analog_disagreement", 0.0).fillna(0.0)
    merged["crowd_consistency_v9"] = merged.get("crowd_consistency", 0.5).fillna(0.5)
    merged["news_consistency_v9"] = merged.get("news_consistency", 0.5).fillna(0.5)
    merged["macro_consistency_v9"] = merged.get("macro_alignment", 0.5).fillna(0.5)
    for label in ("full_agreement", "partial_disagreement", "full_disagreement", "mixed"):
        merged[f"contradiction_{label}"] = (merged["contradiction_type"] == label).astype(float)
    for label in ("retail_dominant", "institutional_dominant", "balanced"):
        merged[f"wltc_state_{label}"] = (merged["wltc_state"] == label).astype(float)

    base_rows = len(merged)
    if base_rows < int(target_rows):
        extra = merged.sample(n=int(target_rows) - base_rows, replace=True, random_state=42).copy()
        next_sample_id = int(merged["sample_id"].max()) + 1
        extra["sample_id"] = np.arange(next_sample_id, next_sample_id + len(extra), dtype=np.int64)
        extra["branch_id"] = extra["branch_id"].astype(int) + 1000
        extra["v19_resampled"] = 1.0
        merged["v19_resampled"] = 0.0
        merged = pd.concat([merged, extra], ignore_index=True)
    else:
        merged = merged.sample(n=int(target_rows), replace=False, random_state=42).copy()
        merged["v19_resampled"] = 0.0

    merged = merged.sort_values(["timestamp", "sample_id", "branch_id"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    report = {
        "archive_path": str(output_path),
        "source_archive_path": str(base_path),
        "rows": int(len(merged)),
        "source_rows": int(base_rows),
        "resampled_rows": int(merged["v19_resampled"].sum()),
        "regime_counts": merged["dominant_regime"].value_counts(dropna=False).to_dict(),
        "contradiction_counts": merged["contradiction_type"].value_counts(dropna=False).to_dict(),
        "columns": merged.columns.tolist(),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the V19 enlarged branch archive with V17/V18 context features.")
    parser.add_argument("--target-rows", type=int, default=100000)
    args = parser.parse_args()
    report = build_branch_archive_v19(target_rows=int(args.target_rows))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
