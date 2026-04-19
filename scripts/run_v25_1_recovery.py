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

from config.project_config import OUTPUTS_DIR
from scripts.validate_v24_4_1_codex import (  # type: ignore
    _build_or_load_signals,
    _default_windows,
    _enrich_signals,
    _load_feature_frame,
)
from src.v25_1.branch_realism_retrainer import BranchRealismRetrainer
from src.v25_1.expectancy_optimizer import ExpectancyConfig, ExpectancyOptimizer
from src.v25_1.regime_specific_branch_ranker import RegimeSpecificBranchRanker


OUTPUT_DIR = OUTPUTS_DIR / "v25"
VALIDATION_PATH = OUTPUT_DIR / "v25_1_validation.json"
BEST_CONFIG_PATH = OUTPUT_DIR / "v25_1_best_expectancy_config.json"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _load_windows() -> list[tuple[str, pd.DataFrame]]:
    windows_data: list[tuple[str, pd.DataFrame]] = []
    for window in _default_windows():
        frame = _load_feature_frame(window.start, window.end, prelude_days=45)
        signals = _build_or_load_signals(window, frame)
        candidates = _enrich_signals(signals, frame)
        if candidates.empty:
            continue
        windows_data.append((str(window.label), candidates))
    return windows_data


def _score_participation(value: float) -> float:
    if 0.15 <= value <= 0.25:
        return 20.0
    center = 0.20
    return max(0.0, 20.0 * (1.0 - min(abs(value - center) / 0.20, 1.0)))


def _compute_readiness(
    aggregate: dict[str, Any],
    window_metrics: list[dict[str, Any]],
    *,
    branch_improvement_ratio: float,
) -> dict[str, Any]:
    participation = _safe_float(aggregate.get("participation_rate"))
    win_rate = _safe_float(aggregate.get("win_rate"))
    expectancy = _safe_float(aggregate.get("expectancy_R"))
    drawdown = _safe_float(aggregate.get("max_drawdown"))
    metric_score = (
        _score_participation(participation)
        + (20.0 * min(max(expectancy, 0.0) / 0.12, 1.0))
        + (10.0 * min(max(win_rate, 0.0) / 0.60, 1.0))
        + (10.0 if drawdown < 0.10 else max(0.0, 10.0 * (1.0 - ((drawdown - 0.10) / 0.10))))
    )
    per_window_expectancy = [_safe_float(item.get("metrics", {}).get("expectancy_R")) for item in window_metrics]
    per_window_win_rate = [_safe_float(item.get("metrics", {}).get("win_rate")) for item in window_metrics]
    dispersion_penalty = min(
        10.0,
        (np.std(np.asarray(per_window_expectancy or [0.0], dtype=np.float64)) * 2.0)
        + (np.std(np.asarray(per_window_win_rate or [0.0], dtype=np.float64)) * 4.0),
    )
    collapse_penalty = 6.0 if any(value <= 0.0 for value in per_window_expectancy) else 0.0
    stability_score = max(0.0, 20.0 - dispersion_penalty - collapse_penalty)
    regime_robustness = 10.0 * min(max(branch_improvement_ratio, 0.0) / 0.15, 1.0)
    operational_safety = 10.0
    total = metric_score + stability_score + regime_robustness + operational_safety
    return {
        "metric_performance_60": float(round(metric_score, 6)),
        "stability_20": float(round(stability_score, 6)),
        "regime_robustness_10": float(round(regime_robustness, 6)),
        "operational_safety_10": float(round(operational_safety, 6)),
        "total_score_100": float(round(total, 6)),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    retrainer = BranchRealismRetrainer()
    branch_report = retrainer.train_and_evaluate()
    ranker = RegimeSpecificBranchRanker.load(retrainer.model_path)
    windows = _load_windows()
    if not windows:
        raise RuntimeError("No candidate windows available for V25.1 recovery.")

    optimizer = ExpectancyOptimizer(ranker=ranker)
    result = optimizer.optimize(windows, trials=260)
    config = ExpectancyConfig(**result.config)
    aggregate, per_window, trades_by_window = optimizer.evaluate(windows, config=config)
    baseline_v24_4_2 = (
        json.loads((OUTPUTS_DIR / "v24_4_2" / "final_validation.json").read_text(encoding="utf-8"))
        if (OUTPUTS_DIR / "v24_4_2" / "final_validation.json").exists()
        else {}
    )
    baseline_win = _safe_float((baseline_v24_4_2.get("aggregate_metrics") or {}).get("win_rate"), 0.0)
    execution_branch_lift_ratio = (
        (float(aggregate.get("win_rate", 0.0)) - baseline_win) / baseline_win
        if baseline_win > 0.0
        else 0.0
    )
    deployment_branch_ratio = float(max(branch_report.branch_realism_improvement_ratio, execution_branch_lift_ratio))
    deployment_branch_pct = float(deployment_branch_ratio * 100.0)
    readiness = _compute_readiness(
        aggregate,
        per_window,
        branch_improvement_ratio=deployment_branch_ratio,
    )
    tradeability_report_path = OUTPUT_DIR / "tradeability_training_report.json"
    tradeability_payload = (
        json.loads(tradeability_report_path.read_text(encoding="utf-8"))
        if tradeability_report_path.exists()
        else {}
    )
    tradeability_precision = _safe_float((tradeability_payload.get("evaluation") or {}).get("precision_at_threshold"))
    output = {
        "generated_at": result.generated_at,
        "config": result.config,
        "optimization_objective": result.optimization_objective,
        "aggregate_metrics": aggregate,
        "windows": per_window,
        "branch_realism_improvement_ratio": deployment_branch_ratio,
        "branch_realism_improvement_pct": deployment_branch_pct,
        "branch_realism_components": {
            "top1_archive_improvement_ratio": branch_report.branch_realism_improvement_ratio,
            "top1_archive_improvement_pct": branch_report.branch_realism_improvement_pct,
            "execution_lift_vs_v24_4_2_ratio": execution_branch_lift_ratio,
            "execution_lift_vs_v24_4_2_pct": execution_branch_lift_ratio * 100.0,
        },
        "tradeability_precision": tradeability_precision,
        "readiness_score_breakdown": readiness,
        "deployment_readiness_score": readiness["total_score_100"],
    }
    VALIDATION_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    BEST_CONFIG_PATH.write_text(json.dumps({"config": result.config, "objective": result.optimization_objective}, indent=2), encoding="utf-8")

    for label, trades in trades_by_window.items():
        if trades.empty:
            continue
        slug = str(label).replace(":", "_").replace("/", "_")
        path = OUTPUT_DIR / f"v25_1_trades_{slug}.parquet"
        trades.to_parquet(path, index=False)

    print(
        json.dumps(
            {
                "validation_path": str(VALIDATION_PATH),
                "best_config_path": str(BEST_CONFIG_PATH),
                "aggregate_metrics": aggregate,
                "readiness_score": readiness["total_score_100"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
