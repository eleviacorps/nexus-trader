from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "outputs" / "evaluation"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def horizon_metric(report: dict[str, Any], horizon: str, field: str, default: float = 0.0) -> float:
    try:
        return float(report["overall"]["horizon_metrics"][horizon][field])
    except Exception:
        try:
            return float(report["folds"][0]["metrics_raw"]["horizons"][horizon][field])
        except Exception:
            return default


def extract_run_summary(run_tag: str) -> dict[str, Any]:
    walkforward_path = EVAL_DIR / f"walkforward_report_{run_tag}.json"
    backtest_path = EVAL_DIR / f"backtest_report_{run_tag}.json"
    training_path = EVAL_DIR / f"training_summary_{run_tag}.json"

    walkforward = load_json(walkforward_path)
    backtest = load_json(backtest_path)
    training = load_json(training_path)

    overall = walkforward.get("overall", {})
    calibrated = overall.get("calibrated_metrics", {})
    raw = overall.get("raw_metrics", {})
    capital = backtest.get("capital_backtests", {})

    return {
        "run_tag": run_tag,
        "years": walkforward.get("years", []),
        "calibration_source_years": walkforward.get("calibration_source_years", []),
        "strategic_roc_auc": float(calibrated.get("roc_auc", raw.get("roc_auc", 0.0)) or 0.0),
        "strategic_accuracy": float(calibrated.get("accuracy", raw.get("accuracy", 0.0)) or 0.0),
        "horizon_15m_roc_auc": horizon_metric(walkforward, "15m", "roc_auc"),
        "horizon_30m_roc_auc": horizon_metric(walkforward, "30m", "roc_auc"),
        "trade_count": int(backtest.get("trade_count", 0) or 0),
        "hold_count": int(backtest.get("hold_count", 0) or 0),
        "participation_rate": float(backtest.get("participation_rate", 0.0) or 0.0),
        "win_rate": float(backtest.get("win_rate", 0.0) or 0.0),
        "avg_unit_pnl": float(backtest.get("avg_unit_pnl", 0.0) or 0.0),
        "decision_threshold": float(backtest.get("decision_threshold", 0.0) or 0.0),
        "confidence_floor": float(backtest.get("confidence_floor", 0.0) or 0.0),
        "gate_threshold": float(backtest.get("gate_threshold", 0.0) or 0.0),
        "hold_threshold": float(backtest.get("hold_threshold", 0.0) or 0.0),
        "capital_backtests": capital,
        "training_test_summary": training.get("test_summary", {}),
    }


def build_markdown(summary: dict[str, Any]) -> str:
    generated_at = summary.get("generated_at", "")
    lines = [
        "# V3 Results Summary",
        "",
        f"Generated at: `{generated_at}`",
        "",
    ]
    for run in summary.get("runs", []):
        capital = run.get("capital_backtests", {})
        usd10 = capital.get("usd_10_fixed_risk", {})
        usd1000 = capital.get("usd_1000_fixed_risk", {})
        lines.extend(
            [
                f"## {run['run_tag']}",
                "",
                f"- years: `{run.get('years', [])}`",
                f"- calibration years: `{run.get('calibration_source_years', [])}`",
                f"- strategic ROC-AUC: `{run.get('strategic_roc_auc', 0.0):.4f}`",
                f"- strategic accuracy: `{run.get('strategic_accuracy', 0.0):.4f}`",
                f"- 15m ROC-AUC: `{run.get('horizon_15m_roc_auc', 0.0):.4f}`",
                f"- 30m ROC-AUC: `{run.get('horizon_30m_roc_auc', 0.0):.4f}`",
                f"- trades: `{run.get('trade_count', 0)}`",
                f"- holds: `{run.get('hold_count', 0)}`",
                f"- participation: `{run.get('participation_rate', 0.0):.4f}`",
                f"- win rate: `{run.get('win_rate', 0.0):.4f}`",
                f"- avg unit pnl: `{run.get('avg_unit_pnl', 0.0):.6f}`",
                f"- $10 fixed-risk final capital: `{float(usd10.get('final_capital', 0.0) or 0.0):.2f}`",
                f"- $10 fixed-risk max DD: `{float(usd10.get('max_drawdown_pct', 0.0) or 0.0):.2f}%`",
                f"- $1000 fixed-risk final capital: `{float(usd1000.get('final_capital', 0.0) or 0.0):.2f}`",
                f"- $1000 fixed-risk max DD: `{float(usd1000.get('max_drawdown_pct', 0.0) or 0.0):.2f}%`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize synced v3 walk-forward and backtest reports.")
    parser.add_argument("--run-tags", nargs="+", default=["mh12_full_v3", "mh12_recent_v3"])
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    required = []
    for run_tag in args.run_tags:
        required.extend(
            [
                EVAL_DIR / f"training_summary_{run_tag}.json",
                EVAL_DIR / f"walkforward_report_{run_tag}.json",
                EVAL_DIR / f"backtest_report_{run_tag}.json",
            ]
        )

    while True:
        missing = [path for path in required if not path.exists()]
        if not missing:
            break
        if not args.wait:
            raise SystemExit(f"missing required files: {[str(path) for path in missing]}")
        time.sleep(max(10, int(args.poll_seconds)))

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runs": [extract_run_summary(run_tag) for run_tag in args.run_tags],
    }

    json_path = EVAL_DIR / "v3_summary.json"
    md_path = EVAL_DIR / "v3_summary.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(summary), encoding="utf-8")
    print(json_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
