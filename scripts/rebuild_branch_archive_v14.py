from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config.project_config import V12_FEATURE_CONSISTENCY_REPORT_PATH, V14_BRANCH_ARCHIVE_PATH, V14_BRANCH_FEATURES_PATH
from src.v12.bar_consistent_features import compute_bar_consistent_features, load_default_raw_bars
from src.v14.acm import fear_indices_from_closes
from src.v14.bst import branch_survival_score


def _trusted_feature_names() -> tuple[str, ...]:
    if not V12_FEATURE_CONSISTENCY_REPORT_PATH.exists():
        return ()
    payload = json.loads(V12_FEATURE_CONSISTENCY_REPORT_PATH.read_text(encoding="utf-8"))
    names = payload.get("legacy_archive_vs_live", {}).get("pass_features", [])
    return tuple(str(name) for name in names)


def _align_features_to_samples(archive: pd.DataFrame) -> pd.DataFrame:
    timestamps = pd.to_datetime(archive["timestamp"], utc=True, errors="coerce")
    raw_bars = load_default_raw_bars(
        start=timestamps.min() - pd.Timedelta(days=3),
        end=timestamps.max() + pd.Timedelta(days=1),
    )
    bcfe = compute_bar_consistent_features(raw_bars)
    fear = fear_indices_from_closes(raw_bars["close"])
    aligned = bcfe.join(fear, how="left")
    target_index = pd.DatetimeIndex(timestamps)
    return aligned.reindex(target_index, method="pad")


def _bst_score(row: pd.Series) -> float:
    anchor = float(row.get("anchor_price", 0.0) or 0.0)
    path = np.asarray(
        [
            anchor,
            float(row.get("predicted_price_5m", anchor) or anchor),
            float(row.get("predicted_price_10m", anchor) or anchor),
            float(row.get("predicted_price_15m", anchor) or anchor),
        ],
        dtype=np.float32,
    )
    atr_pct = float(row.get("bcfe_atr_pct", 0.0) or 0.0)
    current_atr = max(abs(anchor) * atr_pct, 1e-6)
    return branch_survival_score(path, current_atr=current_atr, n_perturbations=30, perturbation_scale=0.30)


def main() -> int:
    archive = pd.read_parquet(PROJECT_ROOT / "outputs" / "v10" / "branch_features_v10_full.parquet").copy()
    aligned = _align_features_to_samples(archive)
    trusted = _trusted_feature_names()
    for name in trusted:
        source = aligned.get(name)
        if source is None:
            continue
        archive[f"bcfe_{name}"] = source.to_numpy(dtype=np.float32)
    for persona in ("retail", "institutional", "algo", "whale", "noise"):
        column = f"fear_index_{persona}"
        if column in aligned.columns:
            archive[column] = aligned[column].to_numpy(dtype=np.float32)
    archive["bst_survival_score"] = archive.apply(_bst_score, axis=1).astype(np.float32)
    fear_columns = [col for col in archive.columns if col.startswith("fear_index_")]
    archive["fear_index_composite"] = archive[fear_columns].mean(axis=1).astype(np.float32)
    archive["acm_memory_pressure"] = (
        0.45 * archive.get("fear_index_retail", 0.0).astype(np.float32)
        + 0.30 * archive.get("fear_index_institutional", 0.0).astype(np.float32)
        + 0.25 * archive.get("fear_index_algo", 0.0).astype(np.float32)
    ).astype(np.float32)

    V14_BRANCH_ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    archive.to_parquet(V14_BRANCH_ARCHIVE_PATH, index=False)
    archive.to_parquet(V14_BRANCH_FEATURES_PATH, index=False)
    report = {
        "row_count": int(len(archive)),
        "column_count": int(len(archive.columns)),
        "trusted_bcfe_count": int(len(trusted)),
        "bst_survival_score_mean": round(float(archive["bst_survival_score"].mean()), 6),
        "fear_index_retail_mean": round(float(archive.get("fear_index_retail", pd.Series([0.0])).mean()), 6),
        "fear_index_institutional_mean": round(float(archive.get("fear_index_institutional", pd.Series([0.0])).mean()), 6),
    }
    print(str(V14_BRANCH_ARCHIVE_PATH), flush=True)
    print(str(V14_BRANCH_FEATURES_PATH), flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

