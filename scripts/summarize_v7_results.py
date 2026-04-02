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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def nested(report: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = report
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def horizon_metric(report: dict[str, Any], horizon: str, field: str, default: float = 0.0) -> float:
    overall_value = nested(report, "overall", "horizon_metrics", horizon, field, default=None)
    if overall_value is not None:
        return safe_float(overall_value, default)
    folds = report.get("folds", [])
    if folds and isinstance(folds, list):
        first = folds[0]
        if isinstance(first, dict):
            return safe_float(
                nested(first, "metrics_raw", "horizons", horizon, field, default=default),
                default,
            )
    return default


def extract_run_summary(run_tag: str) -> dict[str, Any]:
    walkforward_path = EVAL_DIR / f"walkforward_report_{run_tag}.json"
    backtest_path = EVAL_DIR / f"backtest_report_{run_tag}.json"
    training_path = EVAL_DIR / f"training_summary_{run_tag}.json"
    audit_path = EVAL_DIR / f"model_artifact_leakage_report_{run_tag}.json"

    walkforward = load_json(walkforward_path)
    backtest = load_json(backtest_path)
    training = load_json(training_path)
    audit = load_json(audit_path) if audit_path.exists() else {}

    overall = walkforward.get("overall", {})
    calibrated = overall.get("calibrated_metrics", {})
    raw = overall.get("raw_metrics", {})
    classic = backtest.get("backtest", {})
    event_driven = backtest.get("event_driven_backtest", {})
    by_horizon = backtest.get("event_driven_by_horizon", {})
    capital = classic.get("capital_backtests", {})
    classic_fixed_1000 = capital.get("usd_1000_fixed_risk", {})
    event_capital = event_driven.get("capital_backtests", {})
    event_fixed_1000 = event_capital.get("usd_1000_fixed_risk", {})

    return {
        "run_tag": run_tag,
        "years": walkforward.get("years", []),
        "calibration_source_years": walkforward.get("calibration_source_years", []),
        "strategic_roc_auc": safe_float(calibrated.get("roc_auc", raw.get("roc_auc", 0.0))),
        "strategic_accuracy": safe_float(calibrated.get("accuracy", raw.get("accuracy", 0.0))),
        "horizon_15m_roc_auc": horizon_metric(walkforward, "15m", "roc_auc"),
        "horizon_30m_roc_auc": horizon_metric(walkforward, "30m", "roc_auc"),
        "classic_backtest": {
            "trade_count": safe_int(classic.get("trade_count", 0)),
            "hold_count": safe_int(classic.get("hold_count", 0)),
            "participation_rate": safe_float(classic.get("participation_rate", 0.0)),
            "win_rate": safe_float(classic.get("win_rate", 0.0)),
            "avg_unit_pnl": safe_float(classic.get("avg_unit_pnl", 0.0)),
            "decision_threshold": safe_float(classic.get("decision_threshold", 0.0)),
            "confidence_floor": safe_float(classic.get("confidence_floor", 0.0)),
            "gate_threshold": safe_float(classic.get("gate_threshold", 0.0)),
            "hold_threshold": safe_float(classic.get("hold_threshold", 0.0)),
            "usd_1000_fixed_risk_final": safe_float(classic_fixed_1000.get("final_capital", 0.0)),
            "usd_1000_fixed_risk_max_dd": safe_float(classic_fixed_1000.get("max_drawdown_pct", 0.0)),
        },
        "event_driven_backtest": {
            "trade_count": safe_int(event_driven.get("trade_count", 0)),
            "hold_count": safe_int(event_driven.get("hold_count", 0)),
            "participation_rate": safe_float(event_driven.get("participation_rate", 0.0)),
            "win_rate": safe_float(event_driven.get("win_rate", 0.0)),
            "avg_unit_pnl": safe_float(event_driven.get("avg_unit_pnl", 0.0)),
            "fee_model": event_driven.get("fee_model", ""),
            "slippage_model": event_driven.get("slippage_model", ""),
            "usd_1000_fixed_risk_final": safe_float(event_fixed_1000.get("final_capital_mean", event_fixed_1000.get("final_capital", 0.0))),
            "usd_1000_fixed_risk_max_dd": safe_float(event_fixed_1000.get("max_drawdown_pct_mean", event_fixed_1000.get("max_drawdown_pct", 0.0))),
        },
        "event_driven_by_horizon": {
            horizon: {
                "trade_count": safe_int(payload.get("trade_count", 0)),
                "participation_rate": safe_float(payload.get("participation_rate_mean", payload.get("participation_rate", 0.0))),
                "win_rate": safe_float(payload.get("win_rate_mean", payload.get("win_rate", 0.0))),
                "avg_unit_pnl": safe_float(payload.get("avg_unit_pnl_mean", payload.get("avg_unit_pnl", 0.0))),
            }
            for horizon, payload in by_horizon.items()
        },
        "audit_findings": audit.get("findings", []),
        "precision_gate_train_precision": safe_float(audit.get("precision_gate_train_precision", 0.0)),
        "gate_positive_rate": safe_float(audit.get("gate_positive_rate", 0.0)),
        "training_test_summary": training.get("test_summary", {}),
    }


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# V7 Results Summary",
        "",
        f"Generated at: `{summary.get('generated_at', '')}`",
        "",
        "This summary focuses on the metrics that matter for the current Nexus architecture:",
        "",
        "- raw ranking quality (`ROC-AUC`)",
        "- classic filtered backtest behavior",
        "- event-driven execution-aware backtest behavior",
        "- artifact-audit findings for selector / gate health",
        "",
    ]
    for run in summary.get("runs", []):
        classic = run.get("classic_backtest", {})
        event_driven = run.get("event_driven_backtest", {})
        by_horizon = run.get("event_driven_by_horizon", {})
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
                "",
                "Classic filtered backtest:",
                f"- trades: `{classic.get('trade_count', 0)}`",
                f"- holds: `{classic.get('hold_count', 0)}`",
                f"- participation: `{classic.get('participation_rate', 0.0):.4f}`",
                f"- win rate: `{classic.get('win_rate', 0.0):.4f}`",
                f"- avg unit pnl: `{classic.get('avg_unit_pnl', 0.0):.6f}`",
                f"- $1000 fixed-risk final capital: `{classic.get('usd_1000_fixed_risk_final', 0.0):.2f}`",
                f"- $1000 fixed-risk max DD: `{classic.get('usd_1000_fixed_risk_max_dd', 0.0):.2f}%`",
                "",
                "Event-driven backtest:",
                f"- trades: `{event_driven.get('trade_count', 0)}`",
                f"- holds: `{event_driven.get('hold_count', 0)}`",
                f"- participation: `{event_driven.get('participation_rate', 0.0):.4f}`",
                f"- win rate: `{event_driven.get('win_rate', 0.0):.4f}`",
                f"- avg unit pnl: `{event_driven.get('avg_unit_pnl', 0.0):.6f}`",
                f"- fee model: `{event_driven.get('fee_model', '')}`",
                f"- slippage model: `{event_driven.get('slippage_model', '')}`",
                f"- $1000 fixed-risk final capital: `{event_driven.get('usd_1000_fixed_risk_final', 0.0):.2f}`",
                f"- $1000 fixed-risk max DD: `{event_driven.get('usd_1000_fixed_risk_max_dd', 0.0):.2f}%`",
                "",
                "Event-driven by horizon:",
            ]
        )
        for horizon in ("5m", "10m", "15m", "30m"):
            payload = by_horizon.get(horizon)
            if not payload:
                continue
            lines.append(
                f"- {horizon}: trades `{payload.get('trade_count', 0)}`, participation `{payload.get('participation_rate', 0.0):.4f}`, win rate `{payload.get('win_rate', 0.0):.4f}`, avg unit pnl `{payload.get('avg_unit_pnl', 0.0):.6f}`"
            )
        findings = run.get("audit_findings", [])
        lines.extend(
            [
                "",
                "Artifact audit:",
                f"- findings: `{findings}`",
                f"- precision-gate train precision: `{run.get('precision_gate_train_precision', 0.0):.4f}`",
                f"- gate positive rate: `{run.get('gate_positive_rate', 0.0):.4f}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize synced v7 results, including event-driven backtests and audit findings.")
    parser.add_argument("--run-tags", nargs="+", default=["mh12_full_v7", "mh12_recent_v7"])
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
                EVAL_DIR / f"model_artifact_leakage_report_{run_tag}.json",
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

    json_path = EVAL_DIR / "v7_summary.json"
    md_path = EVAL_DIR / "v7_summary.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(summary), encoding="utf-8")
    print(json_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
