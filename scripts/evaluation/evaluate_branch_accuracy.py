from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import MODELS_DIR, OUTPUTS_DIR
from src.v25.branch_quality_model import BranchQualityModel


BRANCH_ARCHIVE_PATH = OUTPUTS_DIR / "v19" / "branch_archive_100k.parquet"
MODEL_PATH = MODELS_DIR / "v25" / "branch_quality_model.json"
REPORT_PATH = OUTPUTS_DIR / "v25" / "branch_accuracy_evaluation.json"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _prepare_eval_rows(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["sample_key"] = working["sample_id"].astype(str)
    working["cabr_proxy"] = working["branch_confidence"].clip(0.0, 1.0)
    working["historical_outcome_fit"] = (
        working.groupby(["dominant_regime", "branch_direction"])["actual_final_return"]
        .transform(lambda s: float(np.mean(np.sign(s) == np.sign(s.mean())) if len(s) else 0.5))
        .clip(0.0, 1.0)
    )
    working["label_hit"] = (
        (np.sign(pd.to_numeric(working["actual_final_return"], errors="coerce").fillna(0.0))
         == np.sign(pd.to_numeric(working["branch_move_size"], errors="coerce").fillna(0.0)))
    ).astype(float)
    return working


def main() -> None:
    if not BRANCH_ARCHIVE_PATH.exists():
        raise FileNotFoundError(f"Missing archive for evaluation: {BRANCH_ARCHIVE_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing branch quality model artifact: {MODEL_PATH}")
    OUTPUTS_DIR.joinpath("v25").mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(BRANCH_ARCHIVE_PATH)
    frame = _prepare_eval_rows(frame)
    eval_frame = frame.loc[frame["year"] >= 2024].copy()
    if eval_frame.empty:
        eval_frame = frame.tail(25000).copy()

    model = BranchQualityModel.load(MODEL_PATH)
    if not model.fitted:
        raise RuntimeError("Loaded branch quality model is not fitted.")

    # Score each branch with new blend.
    blended_scores: list[float] = []
    quality_scores: list[float] = []
    for _, row in eval_frame.iterrows():
        sample = {
            "path": [
                _safe_float(row.get("anchor_price"), 0.0),
                _safe_float(row.get("predicted_price_5m"), 0.0),
                _safe_float(row.get("predicted_price_10m"), 0.0),
                _safe_float(row.get("predicted_price_15m"), 0.0),
            ],
            "branch_volatility": _safe_float(row.get("branch_volatility"), 0.0),
            "branch_acceleration": abs(_safe_float(row.get("branch_move_zscore"), 0.0)),
            "regime_consistency": _safe_float(row.get("hmm_regime_match"), 0.0),
            "analog_similarity": _safe_float(row.get("analog_similarity"), 0.0),
            "specialist_bot_agreement": float(
                np.clip(
                    0.5
                    + 0.25 * _safe_float(row.get("news_consistency"), 0.0)
                    + 0.25 * _safe_float(row.get("crowd_consistency"), 0.0),
                    0.0,
                    1.0,
                )
            ),
            "cabr_score": _safe_float(row.get("cabr_proxy"), 0.0),
            "minority_disagreement": float(np.clip(_safe_float(row.get("branch_disagreement"), 0.0), 0.0, 1.0)),
            "historical_outcome_fit": _safe_float(row.get("historical_outcome_fit"), 0.5),
        }
        prediction = model.predict(sample)
        quality_scores.append(prediction.quality_score)
        blended_scores.append(prediction.blended_rank_score)
    eval_frame["quality_score"] = np.asarray(quality_scores, dtype=np.float64)
    eval_frame["blended_score"] = np.asarray(blended_scores, dtype=np.float64)

    # Compare top-1 branch hit rate per sample_key.
    baseline_top = eval_frame.sort_values(["sample_key", "cabr_proxy"], ascending=[True, False]).groupby("sample_key", as_index=False).head(1)
    blended_top = eval_frame.sort_values(["sample_key", "blended_score"], ascending=[True, False]).groupby("sample_key", as_index=False).head(1)

    baseline_accuracy = float(np.mean(baseline_top["label_hit"])) if len(baseline_top) else 0.0
    blended_accuracy = float(np.mean(blended_top["label_hit"])) if len(blended_top) else 0.0
    improvement_ratio = ((blended_accuracy - baseline_accuracy) / baseline_accuracy) if baseline_accuracy > 0 else 0.0
    report = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "dataset_rows": int(len(eval_frame)),
        "sample_count": int(eval_frame["sample_key"].nunique()),
        "baseline_cabr_top1_accuracy": round(baseline_accuracy, 6),
        "v25_blended_top1_accuracy": round(blended_accuracy, 6),
        "branch_realism_improvement_ratio": round(float(improvement_ratio), 6),
        "branch_realism_improvement_pct": round(float(improvement_ratio * 100.0), 3),
        "target_branch_realism_improvement_pct": 15.0,
        "target_reached": bool((improvement_ratio * 100.0) >= 15.0),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


