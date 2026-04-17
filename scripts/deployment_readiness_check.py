from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import OUTPUTS_DIR


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _load_validation() -> dict[str, Any]:
    path = OUTPUTS_DIR / "v24_4_2" / "final_validation.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing validation artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _performance_score(windows: list[dict[str, Any]]) -> tuple[float, dict[str, float]]:
    if not windows:
        return 0.0, {"participation": 0.0, "expectancy": 0.0, "win_rate": 0.0, "drawdown": 0.0}
    participation_scores = []
    expectancy_scores = []
    win_scores = []
    drawdown_scores = []
    for window in windows:
        metrics = window.get("metrics", {})
        p = _safe_float(metrics.get("participation_rate"))
        e = _safe_float(metrics.get("expectancy_R"))
        w = _safe_float(metrics.get("win_rate"))
        d = _safe_float(metrics.get("max_drawdown"))
        participation_scores.append(max(0.0, 1.0 - (min(abs(p - 0.15), abs(p - 0.30)) / 0.15)) if not (0.15 <= p <= 0.30) else 1.0)
        expectancy_scores.append(max(0.0, min(1.0, e / 0.12)))
        win_scores.append(max(0.0, min(1.0, w / 0.60)))
        drawdown_scores.append(1.0 if d <= 0.18 else max(0.0, 1.0 - ((d - 0.18) / 0.18)))

    parts = {
        "participation": float(sum(participation_scores) / len(participation_scores)) * 20.0,
        "expectancy": float(sum(expectancy_scores) / len(expectancy_scores)) * 20.0,
        "win_rate": float(sum(win_scores) / len(win_scores)) * 10.0,
        "drawdown": float(sum(drawdown_scores) / len(drawdown_scores)) * 10.0,
    }
    return sum(parts.values()), parts


def _stability_score(windows: list[dict[str, Any]]) -> tuple[float, dict[str, float]]:
    if not windows:
        return 0.0, {"dispersion_penalty": 20.0, "collapse_penalty": 20.0}
    participations = [_safe_float(w.get("metrics", {}).get("participation_rate")) for w in windows]
    expectancies = [_safe_float(w.get("metrics", {}).get("expectancy_R")) for w in windows]
    drawdowns = [_safe_float(w.get("metrics", {}).get("max_drawdown")) for w in windows]
    p_disp = max(participations) - min(participations) if participations else 1.0
    e_disp = max(expectancies) - min(expectancies) if expectancies else 1.0
    d_disp = max(drawdowns) - min(drawdowns) if drawdowns else 1.0

    dispersion_penalty = min(12.0, (p_disp * 20.0) + (abs(e_disp) * 10.0) + (d_disp * 8.0))
    collapse_penalty = 8.0 if any(_safe_float(w.get("metrics", {}).get("expectancy_R")) <= 0.0 for w in windows) else 0.0
    score = max(0.0, 20.0 - dispersion_penalty - collapse_penalty)
    return score, {"dispersion_penalty": dispersion_penalty, "collapse_penalty": collapse_penalty}


def _regime_robustness_score(regime_breakdown: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    if not regime_breakdown:
        return 0.0, {"positive_regimes": 0, "required_regimes": 3}
    positive_regimes = 0
    for regime, metrics in regime_breakdown.items():
        expectancy = _safe_float(metrics.get("expectancy_R", metrics.get("expectancy", 0.0)))
        if expectancy > 0.0:
            positive_regimes += 1
    score = min(10.0, (positive_regimes / 3.0) * 10.0)
    return score, {"positive_regimes": positive_regimes, "required_regimes": 3}


def _operational_safety_score(ops: dict[str, Any], total_candidates: int) -> tuple[float, dict[str, Any]]:
    sell_blocks = int(ops.get("sell_guard_blocks", 0))
    cooldown_blocks = int(ops.get("cooldown_blocks", 0))
    cluster_blocks = int(ops.get("cluster_blocks", 0))
    candidates = max(int(total_candidates), 1)
    sell_activity = sell_blocks / candidates
    cooldown_activity = cooldown_blocks / candidates
    cluster_activity = cluster_blocks / candidates

    score = 10.0
    if sell_activity <= 0.0:
        score -= 3.0
    if sell_activity > 0.40:
        score -= 2.0
    if cooldown_activity <= 0.0:
        score -= 2.0
    if cluster_activity > 0.35:
        score -= 3.0
    return max(0.0, score), {
        "sell_guard_activity_rate": round(sell_activity, 6),
        "cooldown_activity_rate": round(cooldown_activity, 6),
        "cluster_activity_rate": round(cluster_activity, 6),
    }


def main() -> None:
    payload = _load_validation()
    windows = list(payload.get("windows", []))
    aggregate = dict(payload.get("aggregate_metrics", {}))
    regime_breakdown = dict(payload.get("regime_breakdown", {}))
    ops = dict(payload.get("operational_safety", {}))
    total_candidates = int(ops.get("total_candidates", sum(int(w.get("metrics", {}).get("number_of_trades", 0)) for w in windows)))

    metric_score, metric_parts = _performance_score(windows)
    stability_score, stability_parts = _stability_score(windows)
    regime_score, regime_parts = _regime_robustness_score(regime_breakdown)
    safety_score, safety_parts = _operational_safety_score(ops, total_candidates=total_candidates)
    total_score = float(metric_score + stability_score + regime_score + safety_score)

    gates = {
        "participation_range_15_30": 0.15 <= _safe_float(aggregate.get("participation_rate")) <= 0.30,
        "expectancy_gt_0_12R": _safe_float(aggregate.get("expectancy_R")) > 0.12,
        "win_rate_gt_60pct": _safe_float(aggregate.get("win_rate")) > 0.60,
        "drawdown_lt_18pct": _safe_float(aggregate.get("max_drawdown")) < 0.18,
        "stable_across_windows": stability_score >= 12.0,
        "regime_positive_3_major": int(regime_parts.get("positive_regimes", 0)) >= 3,
    }
    all_gates_pass = all(bool(value) for value in gates.values())

    if total_score < 50:
        band = "0-49 = not deployable"
    elif total_score < 75:
        band = "50-74 = more tuning needed"
    elif total_score < 90:
        band = "75-89 = paper-trading only"
    else:
        band = "90+ = deployment-ready"

    deployment_status = "READY" if (total_score >= 90.0 and all_gates_pass) else "BLOCKED"
    output = {
        "generated_at": payload.get("generated_at"),
        "score_model": "continuous_weighted",
        "score_breakdown": {
            "metric_performance_60": round(metric_score, 6),
            "stability_20": round(stability_score, 6),
            "regime_robustness_10": round(regime_score, 6),
            "operational_safety_10": round(safety_score, 6),
            "total_score_100": round(total_score, 6),
            "metric_parts": metric_parts,
            "stability_parts": stability_parts,
            "regime_parts": regime_parts,
            "safety_parts": safety_parts,
        },
        "gates": gates,
        "all_gates_pass": all_gates_pass,
        "band": band,
        "deployment_status": deployment_status,
        "aggregate_metrics": aggregate,
        "window_count": len(windows),
    }

    output_dir = OUTPUTS_DIR / "deployment"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "deployment_readiness.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), "deployment_status": deployment_status, "total_score": total_score}, indent=2))


if __name__ == "__main__":
    main()

