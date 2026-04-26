from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "outputs" / "evaluation"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _tagged(path: Path, run_tag: str) -> Path:
    return path.with_name(f"{path.stem}_{run_tag}{path.suffix}")


def _extract_v7(v7_summary: dict[str, Any], run_tag: str) -> dict[str, Any]:
    for run in v7_summary.get("runs", []):
        if str(run.get("run_tag")) == run_tag:
            by_horizon = run.get("event_driven_by_horizon", {})
            event_15m = by_horizon.get("15m", {})
            return {
                "run_tag": run_tag,
                "strategic_roc_auc": _safe_float(run.get("strategic_roc_auc", 0.0)),
                "horizon_15m_roc_auc": _safe_float(run.get("horizon_15m_roc_auc", 0.0)),
                "classic_win_rate": _safe_float(run.get("classic_backtest", {}).get("win_rate", 0.0)),
                "event_driven_15m_win_rate": _safe_float(event_15m.get("win_rate", 0.0)),
                "event_driven_15m_avg_unit_pnl": _safe_float(event_15m.get("avg_unit_pnl", 0.0)),
            }
    return {
        "run_tag": run_tag,
        "strategic_roc_auc": 0.0,
        "horizon_15m_roc_auc": 0.0,
        "classic_win_rate": 0.0,
        "event_driven_15m_win_rate": 0.0,
        "event_driven_15m_avg_unit_pnl": 0.0,
    }


def _extract_v8(run_tag: str) -> dict[str, Any]:
    training = _load_json(EVAL_DIR / f"training_summary_{run_tag}.json")
    walkforward = _load_json(EVAL_DIR / f"walkforward_report_{run_tag}.json")
    selector_eval = _load_json(EVAL_DIR / f"v8_evaluation_report_{run_tag}.json")
    selector_report = _load_json(EVAL_DIR / f"v8_branch_selector_report_{run_tag}.json")
    archive_report = _load_json(EVAL_DIR / f"v8_branch_archive_report_{run_tag}.json")
    event_15m = selector_eval.get("selector_event_driven_15m", {})
    baseline_15m = selector_eval.get("generator_probability_baseline_15m", {})
    importance = selector_eval.get("feature_importance", {})
    top_features = list(importance.items())[:10]
    regime_exec = selector_eval.get("cone_metrics", {}).get("regime_execution_performance", {})
    hardest_regimes = sorted(
        (
            {
                "regime": regime,
                "selector_win_rate": _safe_float(payload.get("selector_win_rate", 0.0)),
                "baseline_win_rate": _safe_float(payload.get("baseline_win_rate", 0.0)),
            }
            for regime, payload in regime_exec.items()
        ),
        key=lambda item: item["selector_win_rate"],
    )[:3]
    return {
        "run_tag": run_tag,
        "training_test_roc_auc": _safe_float(training.get("test_metrics", {}).get("strategic", {}).get("roc_auc", training.get("test_metrics", {}).get("roc_auc", 0.0))),
        "training_15m_roc_auc": _safe_float(training.get("test_metrics", {}).get("horizons", {}).get("15m", {}).get("roc_auc", 0.0)),
        "walkforward_strategic_roc_auc": _safe_float(walkforward.get("overall", {}).get("calibrated_metrics", {}).get("roc_auc", 0.0)),
        "walkforward_15m_roc_auc": _safe_float(walkforward.get("overall", {}).get("horizon_metrics", {}).get("15m", {}).get("roc_auc", 0.0)),
        "selector_top1_branch_accuracy": _safe_float(selector_eval.get("top1_branch_accuracy", 0.0)),
        "selector_top3_branch_containment": _safe_float(selector_eval.get("top3_branch_containment", 0.0)),
        "selector_error_improvement": _safe_float(selector_eval.get("selector_error_improvement", 0.0)),
        "selector_event_driven_15m_win_rate": _safe_float(event_15m.get("win_rate", 0.0)),
        "selector_event_driven_15m_avg_unit_pnl": _safe_float(event_15m.get("avg_unit_pnl", 0.0)),
        "baseline_event_driven_15m_win_rate": _safe_float(baseline_15m.get("win_rate", 0.0)),
        "baseline_event_driven_15m_avg_unit_pnl": _safe_float(baseline_15m.get("avg_unit_pnl", 0.0)),
        "cone_point_containment": _safe_float(selector_eval.get("cone_metrics", {}).get("top3_point_containment_rate", 0.0)),
        "cone_full_containment": _safe_float(selector_eval.get("cone_metrics", {}).get("top3_full_containment_rate", 0.0)),
        "minority_branch_usefulness": _safe_float(selector_eval.get("cone_metrics", {}).get("minority_branch_usefulness", 0.0)),
        "selector_provider": selector_report.get("provider", "none"),
        "selector_feature_importance_top": top_features,
        "hardest_regimes": hardest_regimes,
        "archive_samples": int(archive_report.get("sample_count", 0)),
        "archive_rows": int(archive_report.get("branches", 0)),
    }


def _comparison(v7: dict[str, Any], v8: dict[str, Any]) -> dict[str, Any]:
    return {
        "delta_15m_event_win_rate": round(v8["selector_event_driven_15m_win_rate"] - v7["event_driven_15m_win_rate"], 6),
        "delta_15m_event_avg_unit_pnl": round(v8["selector_event_driven_15m_avg_unit_pnl"] - v7["event_driven_15m_avg_unit_pnl"], 6),
        "delta_15m_roc_auc": round(v8["walkforward_15m_roc_auc"] - v7["horizon_15m_roc_auc"], 6),
        "selector_beats_generator_baseline": bool(v8["selector_event_driven_15m_win_rate"] >= v8["baseline_event_driven_15m_win_rate"]),
        "selector_beats_v7_15m": bool(v8["selector_event_driven_15m_win_rate"] >= v7["event_driven_15m_win_rate"]),
        "hmm_helped": bool(any(str(name).startswith("hmm_") for name, _ in v8.get("selector_feature_importance_top", []))),
        "analog_helped": bool(any(str(name).startswith("analog_") for name, _ in v8.get("selector_feature_importance_top", []))),
        "fair_value_helped": bool(any("fair_value" in str(name) or "mean_reversion" in str(name) for name, _ in v8.get("selector_feature_importance_top", []))),
        "moved_closer_to_market_future_simulator": bool(
            v8["selector_top1_branch_accuracy"] > 0.0
            and v8["selector_top3_branch_containment"] >= 0.50
            and v8["selector_error_improvement"] > 0.0
        ),
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# V8 Results Summary",
        "",
        f"Generated at: `{summary['generated_at']}`",
        "",
        "V8 changes the project story from broad classifier quality to 15-minute branch selection quality.",
        "",
    ]
    for run in summary["runs"]:
        v7 = run["v7"]
        v8 = run["v8"]
        comp = run["comparison"]
        lines.extend(
            [
                f"## {run['run_tag']}",
                "",
                "V7 baseline:",
                f"- strategic ROC-AUC: `{v7['strategic_roc_auc']:.4f}`",
                f"- 15m ROC-AUC: `{v7['horizon_15m_roc_auc']:.4f}`",
                f"- 15m event-driven win rate: `{v7['event_driven_15m_win_rate']:.4f}`",
                f"- 15m event-driven avg unit pnl: `{v7['event_driven_15m_avg_unit_pnl']:.6f}`",
                "",
                "V8 result:",
                f"- walk-forward strategic ROC-AUC: `{v8['walkforward_strategic_roc_auc']:.4f}`",
                f"- walk-forward 15m ROC-AUC: `{v8['walkforward_15m_roc_auc']:.4f}`",
                f"- selector top-1 branch accuracy: `{v8['selector_top1_branch_accuracy']:.4f}`",
                f"- selector top-3 containment: `{v8['selector_top3_branch_containment']:.4f}`",
                f"- selector error improvement vs generator: `{v8['selector_error_improvement']:.6f}`",
                f"- selector 15m event-driven win rate: `{v8['selector_event_driven_15m_win_rate']:.4f}`",
                f"- selector 15m event-driven avg unit pnl: `{v8['selector_event_driven_15m_avg_unit_pnl']:.6f}`",
                f"- generator baseline 15m event-driven win rate: `{v8['baseline_event_driven_15m_win_rate']:.4f}`",
                f"- top-3 cone point containment: `{v8['cone_point_containment']:.4f}`",
                f"- top-3 cone full containment: `{v8['cone_full_containment']:.4f}`",
                f"- minority branch usefulness: `{v8['minority_branch_usefulness']:.4f}`",
                "",
                "Comparison:",
                f"- delta 15m event-driven win rate vs V7: `{comp['delta_15m_event_win_rate']:+.4f}`",
                f"- delta 15m avg unit pnl vs V7: `{comp['delta_15m_event_avg_unit_pnl']:+.6f}`",
                f"- delta 15m ROC-AUC vs V7: `{comp['delta_15m_roc_auc']:+.4f}`",
                f"- selector beats generator baseline: `{comp['selector_beats_generator_baseline']}`",
                f"- selector beats V7 15m event-driven win rate: `{comp['selector_beats_v7_15m']}`",
                f"- HMM helped: `{comp['hmm_helped']}`",
                f"- analog helped: `{comp['analog_helped']}`",
                f"- fair-value / mean-reversion helped: `{comp['fair_value_helped']}`",
                f"- moved closer to a true market-future simulator: `{comp['moved_closer_to_market_future_simulator']}`",
                "",
                "Top selector features:",
            ]
        )
        for name, value in v8["selector_feature_importance_top"][:8]:
            lines.append(f"- {name}: `{float(value):.6f}`")
        lines.append("")
        lines.append("Hardest regimes:")
        for payload in v8["hardest_regimes"]:
            lines.append(
                f"- {payload['regime']}: selector win `{payload['selector_win_rate']:.4f}`, baseline win `{payload['baseline_win_rate']:.4f}`"
            )
        lines.append("")
    lines.extend(
        [
            "## Overall",
            "",
            summary["overall_conclusion"],
            "",
            "## V9 Direction",
            "",
        ]
    )
    for step in summary["next_steps"]:
        lines.append(f"- {step}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize final V8 results against V7.")
    parser.add_argument("--run-tags", nargs="+", default=["mh12_full_v8", "mh12_recent_v8"])
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    required: list[Path] = [EVAL_DIR / "v7_summary.json"]
    for run_tag in args.run_tags:
        required.extend(
            [
                EVAL_DIR / f"training_summary_{run_tag}.json",
                EVAL_DIR / f"walkforward_report_{run_tag}.json",
                EVAL_DIR / f"v8_evaluation_report_{run_tag}.json",
                EVAL_DIR / f"v8_branch_selector_report_{run_tag}.json",
                EVAL_DIR / f"v8_branch_archive_report_{run_tag}.json",
            ]
        )
    while True:
        missing = [path for path in required if not path.exists()]
        if not missing:
            break
        if not args.wait:
            raise SystemExit(f"missing required files: {[str(path) for path in missing]}")
        time.sleep(max(10, int(args.poll_seconds)))

    v7_summary = _load_json(EVAL_DIR / "v7_summary.json")
    runs = []
    for run_tag in args.run_tags:
        if "recent" in run_tag:
            v7_tag = "mh12_recent_v7"
        else:
            v7_tag = "mh12_full_v7"
        run_payload = {
            "run_tag": run_tag,
            "v7": _extract_v7(v7_summary, v7_tag),
            "v8": _extract_v8(run_tag),
        }
        run_payload["comparison"] = _comparison(run_payload["v7"], run_payload["v8"])
        runs.append(run_payload)

    best_run = max(runs, key=lambda item: item["v8"]["selector_event_driven_15m_win_rate"]) if runs else None
    overall_conclusion = (
        f"V8 {'improved' if best_run and best_run['comparison']['selector_beats_v7_15m'] else 'did not improve'} the project on its real target: "
        f"15-minute event-driven future-path selection. The strongest run was `{best_run['run_tag']}` with selector 15m win rate "
        f"`{best_run['v8']['selector_event_driven_15m_win_rate']:.4f}` and top-1 branch accuracy "
        f"`{best_run['v8']['selector_top1_branch_accuracy']:.4f}`."
        if best_run
        else "V8 did not produce any completed runs."
    )
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runs": runs,
        "overall_conclusion": overall_conclusion,
        "next_steps": [
            "Promote the V8 selector into the live branch ranking path so the UI uses the learned judge instead of the older heuristic selector.",
            "Train a richer branch archive on more dense recent samples around high-impact regimes instead of uniform broad-history sampling.",
            "Add GPT-OSS only as the explanation/judgment sidecar for selected branch rationale and catalyst narrative, not as the numeric selector.",
            "Make V9 optimize directly for event-driven 15m selected-branch PnL and containment, not broad ROC-AUC.",
        ],
    }
    json_path = EVAL_DIR / "v8_summary.json"
    md_path = EVAL_DIR / "v8_summary.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(summary), encoding="utf-8")
    print(json_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
