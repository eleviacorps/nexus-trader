from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
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


def _load_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return default or {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else (default or {})
    except Exception:
        return default or {}


def main() -> None:
    out_dir = OUTPUTS_DIR / "v25"
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = _load_json(OUTPUTS_DIR / "v24_4_1" / "metric_comparison.json")
    v25_val = _load_json(out_dir / "v25_1_validation.json")
    branch = _load_json(out_dir / "branch_realism_report.json")
    tradeability = _load_json(out_dir / "tradeability_training_report.json")
    readiness = _load_json(OUTPUTS_DIR / "deployment" / "deployment_readiness.json")
    live_proxy = _load_json(OUTPUTS_DIR / "live" / "live_paper_report.json")

    v24_3 = dict((baseline.get("variants") or {}).get("v24_3", {}))
    v24_4 = dict((baseline.get("variants") or {}).get("v24_4", {}))
    v25_metrics = dict(v25_val.get("aggregate_metrics", {}))
    v25_record = {
        "number_of_trades": int(v25_metrics.get("number_of_trades", 0)),
        "participation_rate": _safe_float(v25_metrics.get("participation_rate")),
        "win_rate": _safe_float(v25_metrics.get("win_rate")),
        "expectancy_R": _safe_float(v25_metrics.get("expectancy_R")),
        "max_drawdown": _safe_float(v25_metrics.get("max_drawdown")),
        "branch_realism_improvement_ratio": _safe_float(v25_val.get("branch_realism_improvement_ratio", branch.get("branch_realism_improvement_ratio"))),
        "tradeability_precision": _safe_float((tradeability.get("evaluation") or {}).get("precision_at_threshold")),
        "readiness_score": _safe_float(v25_val.get("deployment_readiness_score", (readiness.get("score_breakdown") or {}).get("total_score_100"))),
        "proxy_live_positive": bool(live_proxy.get("proxy_positive", False)),
    }
    comparison = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "variants": {
            "v24_3": {
                "participation_rate": _safe_float(v24_3.get("participation_rate")),
                "win_rate": _safe_float(v24_3.get("win_rate")),
                "expectancy_R": _safe_float(v24_3.get("expectancy_R")),
                "max_drawdown": _safe_float(v24_3.get("max_drawdown")),
            },
            "v24_4": {
                "participation_rate": _safe_float(v24_4.get("participation_rate")),
                "win_rate": _safe_float(v24_4.get("win_rate")),
                "expectancy_R": _safe_float(v24_4.get("expectancy_R")),
                "max_drawdown": _safe_float(v24_4.get("max_drawdown")),
            },
            "v25": v25_record,
        },
        "targets": {
            "participation_range": [0.15, 0.25],
            "win_rate_gt": 0.60,
            "expectancy_R_gt": 0.12,
            "drawdown_lt": 0.10,
            "branch_realism_improvement_gt": 0.15,
            "tradeability_precision_gt": 0.65,
            "readiness_score_gt": 90.0,
            "proxy_live_positive": True,
        },
    }
    v24_4_participation = _safe_float(v24_4.get("participation_rate"))
    v25_better_checks = {
        "participation_in_target_band": 0.15 <= v25_record["participation_rate"] <= 0.25,
        "win_rate_meets_target": v25_record["win_rate"] > 0.60,
        "expectancy_better_than_v24_4": v25_record["expectancy_R"] > _safe_float(v24_4.get("expectancy_R")),
        "drawdown_within_target": v25_record["max_drawdown"] < 0.10,
        "branch_realism_meets_target": v25_record["branch_realism_improvement_ratio"] > 0.15,
        "readiness_meets_target": v25_record["readiness_score"] > 90.0,
        "proxy_positive": bool(v25_record["proxy_live_positive"]),
        "v24_4_under_participated": v24_4_participation < 0.15,
    }
    comparison["better_than_v24_4_checks"] = v25_better_checks
    comparison["is_v25_truly_better_than_v24_4"] = bool(all(v25_better_checks.values()))

    json_path = out_dir / "final_metric_comparison.json"
    json_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    lines = [
        "# V25 Final Metric Comparison",
        "",
        f"Generated at: `{comparison['generated_at']}`",
        "",
        "## Metrics",
        f"- V24.3 participation `{comparison['variants']['v24_3']['participation_rate']:.6f}`, win `{comparison['variants']['v24_3']['win_rate']:.6f}`, expectancy `{comparison['variants']['v24_3']['expectancy_R']:.6f}`, drawdown `{comparison['variants']['v24_3']['max_drawdown']:.6f}`",
        f"- V24.4 participation `{comparison['variants']['v24_4']['participation_rate']:.6f}`, win `{comparison['variants']['v24_4']['win_rate']:.6f}`, expectancy `{comparison['variants']['v24_4']['expectancy_R']:.6f}`, drawdown `{comparison['variants']['v24_4']['max_drawdown']:.6f}`",
        f"- V25 participation `{v25_record['participation_rate']:.6f}`, win `{v25_record['win_rate']:.6f}`, expectancy `{v25_record['expectancy_R']:.6f}`, drawdown `{v25_record['max_drawdown']:.6f}`",
        f"- V25 branch realism improvement `{v25_record['branch_realism_improvement_ratio']:.6f}`",
        f"- V25 tradeability precision `{v25_record['tradeability_precision']:.6f}`",
        f"- V25 readiness score `{v25_record['readiness_score']:.6f}`",
        f"- V25 proxy live positive `{v25_record['proxy_live_positive']}`",
        "",
        f"Is V25 truly better than V24.4? `{comparison['is_v25_truly_better_than_v24_4']}`",
    ]
    md_path = out_dir / "final_metric_comparison.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "md": str(md_path), "is_v25_truly_better_than_v24_4": comparison["is_v25_truly_better_than_v24_4"]}, indent=2))


if __name__ == "__main__":
    main()
