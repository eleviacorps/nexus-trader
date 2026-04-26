from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import MODELS_DIR, OUTPUTS_DIR
from src.v25.branch_quality_model import BranchQualityModel


BRANCH_ARCHIVE_PATH = OUTPUTS_DIR / "v19" / "branch_archive_100k.parquet"
MODEL_PATH = MODELS_DIR / "v25" / "branch_quality_model.json"
REPORT_PATH = OUTPUTS_DIR / "v25" / "branch_quality_training_report.json"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _prepare_samples(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    working = frame.copy()
    working["path_error_abs"] = working["path_error"].abs()
    path_error_cut = float(working["path_error_abs"].quantile(0.50))
    invalid_cut = float(working["path_error_abs"].quantile(0.75))
    historical_fit = (
        working.groupby(["dominant_regime", "branch_direction"])["actual_final_return"]
        .transform(lambda s: float(np.mean(np.sign(s) == np.sign(s.mean())) if len(s) else 0.5))
        .clip(0.0, 1.0)
    )
    samples: list[dict[str, Any]] = []
    for _, row in working.iterrows():
        direction = 1.0 if _safe_float(row.get("branch_move_size"), 0.0) >= 0.0 else -1.0
        actual = _safe_float(row.get("actual_final_return"), 0.0)
        move_size = _safe_float(row.get("branch_move_size"), 0.0)
        hit_target = float((np.sign(actual) == np.sign(move_size)) and (abs(actual) >= max(1e-6, 0.25 * abs(move_size))))
        invalidated = float((np.sign(actual) != np.sign(move_size)) and (_safe_float(row.get("path_error_abs"), 0.0) >= invalid_cut))
        realism = float((_safe_float(row.get("volatility_realism"), 0.0) >= 0.45) and (_safe_float(row.get("path_error_abs"), 0.0) <= path_error_cut))
        path = [
            _safe_float(row.get("anchor_price"), 0.0),
            _safe_float(row.get("predicted_price_5m"), 0.0),
            _safe_float(row.get("predicted_price_10m"), 0.0),
            _safe_float(row.get("predicted_price_15m"), 0.0),
        ]
        samples.append(
            {
                "path": path,
                "branch_volatility": _safe_float(row.get("branch_volatility"), 0.0),
                "branch_acceleration": abs(_safe_float(row.get("branch_move_zscore"), 0.0)),
                "regime_consistency": _safe_float(row.get("hmm_regime_match"), 0.0),
                "analog_similarity": _safe_float(row.get("analog_similarity"), 0.0),
                # proxy for specialist-bot agreement in archive space
                "specialist_bot_agreement": float(
                    np.clip(
                        0.5
                        + 0.25 * _safe_float(row.get("news_consistency"), 0.0)
                        + 0.25 * _safe_float(row.get("crowd_consistency"), 0.0),
                        0.0,
                        1.0,
                    )
                ),
                # archive has no direct CABR field; branch_confidence is used as CABR proxy for V25 blend.
                "cabr_score": float(np.clip(_safe_float(row.get("branch_confidence"), 0.5), 0.0, 1.0)),
                "minority_disagreement": float(np.clip(_safe_float(row.get("branch_disagreement"), 0.0), 0.0, 1.0)),
                "historical_outcome_fit": float(np.clip(_safe_float(historical_fit.loc[row.name], 0.5), 0.0, 1.0)),
                "label_realism": realism,
                "label_hits_target": hit_target,
                "label_invalidated": invalidated,
                "meta_year": int(row.get("year", 0)),
                "meta_direction": "BUY" if direction >= 0 else "SELL",
            }
        )
    return samples


def main() -> None:
    if not BRANCH_ARCHIVE_PATH.exists():
        raise FileNotFoundError(f"Missing branch archive: {BRANCH_ARCHIVE_PATH}")
    OUTPUTS_DIR.joinpath("v25").mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(BRANCH_ARCHIVE_PATH)
    samples = _prepare_samples(frame)
    if not samples:
        raise RuntimeError("No branch samples prepared for training.")

    train = [item for item in samples if int(item.get("meta_year", 0)) <= 2023]
    valid = [item for item in samples if int(item.get("meta_year", 0)) > 2023]
    if len(train) < 1000:
        split = int(len(samples) * 0.8)
        train = samples[:split]
        valid = samples[split:]

    model = BranchQualityModel()
    train_stats = model.fit(train)
    model.save(MODEL_PATH)

    # quick validation
    predictions = [model.predict(item) for item in valid[: min(len(valid), 15000)]]
    realism_mean = float(np.mean([item.branch_realism_score for item in predictions])) if predictions else 0.0
    hit_mean = float(np.mean([item.probability_branch_hits_target for item in predictions])) if predictions else 0.0
    invalid_mean = float(np.mean([item.probability_branch_is_invalidated for item in predictions])) if predictions else 0.0

    report = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "model_path": str(MODEL_PATH),
        "dataset": {
            "source": str(BRANCH_ARCHIVE_PATH),
            "total_rows": len(samples),
            "train_rows": len(train),
            "valid_rows": len(valid),
        },
        "training": train_stats,
        "validation": {
            "mean_realism_probability": realism_mean,
            "mean_hit_probability": hit_mean,
            "mean_invalid_probability": invalid_mean,
        },
        "ranking_formula": "0.50 * CABR + 0.30 * branch_quality_model + 0.20 * historical_analog_fit",
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

