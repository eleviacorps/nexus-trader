from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

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
    run_validation,
)
from src.v24_4_2.recovery_runtime import build_validation_result, safe_float
from src.v24_4_2.threshold_optimizer import ThresholdConfig, ThresholdOptimizer


OUTPUT_DIR = OUTPUTS_DIR / "v24_4_2"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _build_windows_data() -> list[tuple[Any, pd.DataFrame, pd.DataFrame]]:
    windows_data: list[tuple[Any, pd.DataFrame, pd.DataFrame]] = []
    for window in _default_windows():
        frame = _load_feature_frame(window.start, window.end, prelude_days=45)
        signals = _build_or_load_signals(window, frame)
        candidates = _enrich_signals(signals, frame)
        if candidates.empty:
            continue
        windows_data.append((window, candidates, frame))
    return windows_data


def _build_root_cause_report() -> str:
    metric = _load_json(OUTPUTS_DIR / "v24_4_1" / "metric_comparison.json")
    ablation = _load_json(OUTPUTS_DIR / "v24_4_1" / "component_ablation_report.json")
    regime = _load_json(OUTPUTS_DIR / "v24_4_1" / "regime_breakdown.json")
    v24_3 = dict((metric.get("variants") or {}).get("v24_3", {}))
    v24_4 = dict((metric.get("variants") or {}).get("v24_4", {}))

    lines = [
        "# V24.4.2 Baseline Root Cause",
        "",
        "## Quantified Baseline",
        f"- V24.3 participation: `{safe_float(v24_3.get('participation_rate')):.6f}`",
        f"- V24.3 expectancy: `{safe_float(v24_3.get('expectancy_R')):.6f}`",
        f"- V24.3 win rate: `{safe_float(v24_3.get('win_rate')):.6f}`",
        f"- V24.4 participation: `{safe_float(v24_4.get('participation_rate')):.6f}`",
        f"- V24.4 expectancy: `{safe_float(v24_4.get('expectancy_R')):.6f}`",
        f"- V24.4 win rate: `{safe_float(v24_4.get('win_rate')):.6f}`",
        "",
        "## Why V24.3 Overtrades",
        (
            f"- Router permissiveness: trade rate is `{safe_float(v24_3.get('participation_rate')):.3f}` "
            "with weak gating and near-random expectancy."
        ),
        "- Admission/cooldown controls were either absent or not binding in the V24.3 path.",
        "",
        "## Why V24.4 Undertrades",
        (
            f"- Admission + cooldown interaction reduced participation to `{safe_float(v24_4.get('participation_rate')):.3f}` "
            "while preserving quality metrics."
        ),
        "- Mean-reversion was favored while trend participation was suppressed, shrinking trade count in trending windows.",
        "",
        "## Component Help/Hurt (Ablation)",
    ]
    for row in ablation.get("ablations", []):
        lines.append(
            f"- {row.get('configuration')}: d_participation={safe_float(row.get('change_in_participation')):+.6f}, "
            f"d_expectancy={safe_float(row.get('change_in_expectancy')):+.6f}, "
            f"d_drawdown={safe_float(row.get('change_in_drawdown')):+.6f}"
        )
    lines.extend(
        [
            "",
            "## Regime Findings",
            f"- Best regime: `{regime.get('best_regime')}`",
            f"- Worst regime: `{regime.get('worst_regime')}`",
        ]
    )
    for rec in regime.get("recommendations", []):
        lines.append(f"- Recommendation: {rec}")
    lines.extend(
        [
            "",
            "## Lock",
            (
                "- V24.4.2 recovery will keep generator/CABR/world-state unchanged and only retune "
                "admission thresholds, cooldown decay, cluster radius, and size multipliers."
            ),
        ]
    )
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 0: rerun baseline diagnostics with the existing harness.
    run_validation()
    baseline_md = _build_root_cause_report()
    (OUTPUT_DIR / "baseline_root_cause.md").write_text(baseline_md, encoding="utf-8")

    windows_data = _build_windows_data()
    if not windows_data:
        raise RuntimeError("No validation windows available for v24_4_2 threshold training.")

    base_config = ThresholdConfig(
        trend_up=0.54,
        trend_down=0.64,
        breakout=0.58,
        range_value=0.60,
        cooldown_decay=0.75,
        cluster_radius=0.25,
        size_multiplier=1.00,
    )

    def evaluator(config: ThresholdConfig) -> dict[str, Any]:
        result = build_validation_result(windows_data, config)
        return dict(result.aggregate_metrics)

    optimizer = ThresholdOptimizer(evaluator=evaluator)
    best_result, search_rows = optimizer.optimize(base=base_config)

    (OUTPUT_DIR / "grid_search_results.json").write_text(json.dumps(search_rows, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "best_threshold_config.json").write_text(
        json.dumps(
            {
                "config": best_result.config.__dict__,
                "objective_score": best_result.objective_score,
                "metrics": best_result.metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(OUTPUT_DIR),
                "best_config": best_result.config.__dict__,
                "best_metrics": best_result.metrics,
                "objective_score": best_result.objective_score,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

