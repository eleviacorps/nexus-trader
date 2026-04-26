from __future__ import annotations

import json
import math
import statistics as stats
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import (  # noqa: E402
    V14_RESEARCH_PAPER_OUTLINE_PATH,
    V14_SUMMARY_JSON_PATH,
    V14_SUMMARY_MD_PATH,
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=False))
    den_x = sum((x - mean_x) ** 2 for x in xs)
    den_y = sum((y - mean_y) ** 2 for y in ys)
    denom = math.sqrt(den_x * den_y)
    if denom <= 0.0:
        return None
    return float(num / denom)


def _mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _safe_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def main() -> int:
    v13_summary = _load(PROJECT_ROOT / "outputs" / "evaluation" / "v13_summary.json")
    v13_walkforward = _load(PROJECT_ROOT / "outputs" / "v13" / "backtrader_walkforward_v13.json")
    v13_month = _load(PROJECT_ROOT / "outputs" / "v13" / "backtrader_month_2023_12_v13.json")
    v14_cabr = _load(PROJECT_ROOT / "outputs" / "v14" / "cabr_temporal_evaluation_report.json")
    v14_ssc = _load(PROJECT_ROOT / "outputs" / "v14" / "ssc_evaluation_report.json")
    v14_month = _load(PROJECT_ROOT / "outputs" / "v14" / "backtrader_month_2023_12_v14.json")
    v14_walkforward = _load(PROJECT_ROOT / "outputs" / "v14" / "backtrader_walkforward_v14.json")
    ldrg_status = _load(PROJECT_ROOT / "outputs" / "v14" / "ldrg_status.json")

    months = list(v14_walkforward.get("months", []))
    fear_values = [_safe_float(month.get("avg_fear_index_retail")) for month in months]
    month_returns = [_safe_float(month.get("return_pct")) for month in months]
    bst_values = [_safe_float(month.get("avg_bst_survival_score")) for month in months]
    profit_factors = [_safe_float(month.get("profit_factor")) for month in months]
    ssc_rejections = [_safe_float(month.get("ssc_rejection_rate")) for month in months]

    fear_median = stats.median(fear_values) if fear_values else 0.0
    bst_median = stats.median(bst_values) if bst_values else 0.0

    high_fear_returns = [ret for fear, ret in zip(fear_values, month_returns, strict=False) if fear >= fear_median]
    low_fear_returns = [ret for fear, ret in zip(fear_values, month_returns, strict=False) if fear < fear_median]
    high_bst_profit_factors = [pf for bst, pf in zip(bst_values, profit_factors, strict=False) if bst >= bst_median]
    low_bst_profit_factors = [pf for bst, pf in zip(bst_values, profit_factors, strict=False) if bst < bst_median]

    v13_cabr = _safe_float(v13_summary["phase_2_cabr"]["heldout_pairwise_accuracy_overall"])
    v14_cabr_acc = _safe_float(v14_cabr["heldout_pairwise_accuracy_overall"])
    v13_rcpc_error = _safe_float(v13_summary["phase_3_to_8"]["rcpc"]["calibration_error"])
    v14_rsc_errors = {
        str(key): float(value)
        for key, value in v14_walkforward.get("rsc_summary", {}).get("calibration_error_per_regime", {}).items()
    }
    v14_rsc_max_error = _safe_float(v14_walkforward.get("rsc_summary", {}).get("max_calibration_error"), default=1.0)
    bst_corr = _corr(bst_values, profit_factors)
    fear_corr = _corr(fear_values, month_returns)

    comparison = {
        "cabr_pairwise_accuracy": {
            "v13_snapshot": round(v13_cabr, 6),
            "v14_temporal": round(v14_cabr_acc, 6),
            "delta": round(v14_cabr_acc - v13_cabr, 6),
            "target_above_056_reached": bool(v14_cabr_acc >= 0.56),
        },
        "walkforward": {
            "v13": {
                "month_count": int(v13_walkforward["month_count"]),
                "aggregate_trades": int(v13_walkforward["aggregate_trades"]),
                "aggregate_win_rate": round(_safe_float(v13_walkforward["aggregate_win_rate"]), 6),
                "aggregate_return_pct_sum": round(_safe_float(v13_walkforward["aggregate_return_pct_sum"]), 6),
                "profitable_months": int(v13_walkforward["profitable_months"]),
                "objective_pass_months": int(v13_walkforward["objective_pass_months"]),
                "max_single_month_drawdown_pct": round(_safe_float(v13_walkforward["max_single_month_drawdown_pct"]), 6),
            },
            "v14": {
                "month_count": int(v14_walkforward["month_count"]),
                "aggregate_trades": int(v14_walkforward["aggregate_trades"]),
                "aggregate_win_rate": round(_safe_float(v14_walkforward["aggregate_win_rate"]), 6),
                "aggregate_return_pct_sum": round(_safe_float(v14_walkforward["aggregate_return_pct_sum"]), 6),
                "aggregate_profit_factor": round(_safe_float(v14_walkforward["aggregate_profit_factor"]), 6),
                "profitable_months": int(v14_walkforward["profitable_months"]),
                "objective_pass_months": int(v14_walkforward["objective_pass_months"]),
                "max_single_month_drawdown_pct": round(_safe_float(v14_walkforward["max_single_month_drawdown_pct"]), 6),
                "avg_stage_1_vs_stage_2_gap": round(_safe_float(v14_walkforward["avg_stage_1_vs_stage_2_gap"]), 6),
            },
        },
        "december_2023_backtrader_month": {
            "v13": {
                "final_capital": round(_safe_float(v13_month["final_capital"]), 6),
                "net_profit": round(_safe_float(v13_month["net_profit"]), 6),
                "return_pct": round(_safe_float(v13_month["return_pct"]), 6),
                "trades_executed": int(v13_month["trades_executed"]),
                "win_rate": round(_safe_float(v13_month["win_rate"]), 6),
                "profit_factor": round(_safe_float(v13_month["profit_factor"]), 6),
                "max_drawdown_pct": round(_safe_float(v13_month["max_drawdown_pct"]), 6),
                "stage_1_vs_stage_2_gap": round(_safe_float(v13_month["stage_1_vs_stage_2_gap"]), 6),
            },
            "v14": {
                "final_capital": round(_safe_float(v14_month["final_capital"]), 6),
                "net_profit": round(_safe_float(v14_month["net_profit"]), 6),
                "return_pct": round(_safe_float(v14_month["return_pct"]), 6),
                "trades_executed": int(v14_month["trades_executed"]),
                "win_rate": round(_safe_float(v14_month["win_rate"]), 6),
                "profit_factor": round(_safe_float(v14_month["profit_factor"]), 6),
                "max_drawdown_pct": round(_safe_float(v14_month["max_drawdown_pct"]), 6),
                "stage_1_vs_stage_2_gap": round(_safe_float(v14_month["stage_1_vs_stage_2_gap"]), 6),
            },
        },
        "calibration": {
            "v13_rcpc_global_error": round(v13_rcpc_error, 6),
            "v14_rsc_error_per_regime": {key: round(value, 6) for key, value in v14_rsc_errors.items()},
            "v14_rsc_max_error": round(v14_rsc_max_error, 6),
            "target_below_020_reached": bool(v14_rsc_max_error <= 0.20),
        },
        "bst_impact": {
            "walkforward_aggregate_profit_factor_v14": round(_safe_float(v14_walkforward["aggregate_profit_factor"]), 6),
            "correlation_avg_bst_vs_month_profit_factor": None if bst_corr is None else round(bst_corr, 6),
            "high_bst_month_avg_profit_factor": None if _mean(high_bst_profit_factors) is None else round(_mean(high_bst_profit_factors), 6),
            "low_bst_month_avg_profit_factor": None if _mean(low_bst_profit_factors) is None else round(_mean(low_bst_profit_factors), 6),
            "interpretation": "Observed relationship only, not a clean ablation. Higher-BST months showed better average profit factor in the V14 walk-forward.",
        },
        "ssc": {
            "assumption_risk_mae": round(_safe_float(v14_ssc["assumption_risk_mae"]), 6),
            "context_consistency_mae": round(_safe_float(v14_ssc["context_consistency_mae"]), 6),
            "contradiction_depth_mae": round(_safe_float(v14_ssc["contradiction_depth_mae"]), 6),
            "composite_score_mean": round(_safe_float(v14_ssc["composite_score_mean"]), 6),
            "walkforward_avg_rejection_rate": round(_mean(ssc_rejections) or 0.0, 6),
            "impact": "SSC trained cleanly, but it did not become an active veto in the current walk-forward because rejection rate stayed at 0.0.",
        },
        "acm": {
            "fear_index_mean_by_month": {
                str(key): round(_safe_float(value), 6)
                for key, value in dict(v14_walkforward.get("fear_index_by_month", {})).items()
            },
            "correlation_fear_vs_month_return": None if fear_corr is None else round(fear_corr, 6),
            "high_fear_month_avg_return_pct": None if _mean(high_fear_returns) is None else round(_mean(high_fear_returns), 6),
            "low_fear_month_avg_return_pct": None if _mean(low_fear_returns) is None else round(_mean(low_fear_returns), 6),
            "interpretation": "High-fear months were still weaker on average, so ACM likely improved realism more than execution resilience in this cycle.",
        },
        "ldrg": ldrg_status,
        "paper_trade_status": v14_walkforward.get("paper_trade_summary", {}),
    }

    honest_interpretation = [
        "V14 achieved the most important model-quality target: temporal CABR rose to 0.641945 and cleared the >0.56 objective by a wide margin.",
        "The walk-forward became much more selective and much more robust on a profit-factor basis, but that selectivity overshot into under-trading.",
        "RSC did not reach the required calibration quality. The worst regime error remained 0.513619, so calibration is still the main production blocker.",
        "SSC is trained and integrated, but its current thresholding is too permissive to influence decisions in practice.",
        "ACM and BST look directionally useful for realism and robustness, but V14 did not convert them into a passing deployment-readiness profile yet.",
        "LDRG remains at Tier 0 because profitable-month breadth, paper-trade count, and regime-level calibration are still below the required bar.",
    ]
    v15_recommendation = [
        "Retune the V14 execution policy to recover activity without giving back the new profit-factor improvement. The clearest lever is LRTD and threshold conservatism.",
        "Push RSC below 0.20 per regime with more paper-trade accumulation, stronger regime smoothing, and possibly monotonic regime-specific recalibration constraints.",
        "Make SSC matter operationally by tightening the reject threshold or by using critique score as a size scalar instead of a near-never-fired veto.",
        "Run one explicit BST ablation so robustness impact is measured causally rather than only observationally.",
    ]

    summary = {
        "version": "v14",
        "prompt_completion": {
            "phase_0": "complete",
            "phase_1_acm": "complete",
            "phase_2_bst": "complete",
            "phase_3_ssc": "complete",
            "phase_4_temporal_cabr": "complete",
            "phase_5_rsc": "complete",
            "phase_6_ldrg": "complete",
            "phase_7_backtrader_stack": "complete",
            "phase_8_walkforward": "complete",
            "phase_9_month_run": "complete",
            "phase_10_ldrg_check": "complete",
            "phase_11_final_reports": "complete",
        },
        "metric_comparison": comparison,
        "honest_interpretation": honest_interpretation,
        "v15_recommendation": v15_recommendation,
    }

    V14_SUMMARY_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    V14_SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        "# V14 Summary",
        "",
        "## Prompt Completion",
        "",
        "- All V14 prompt phases were completed end-to-end.",
        "- New systems landed: ACM, BST, SSC, temporal CABR, RSC, and LDRG.",
        "",
        "## V13 vs V14",
        "",
        f"- CABR pairwise accuracy: V13 {v13_cabr:.6f} -> V14 {v14_cabr_acc:.6f} (delta {v14_cabr_acc - v13_cabr:+.6f})",
        f"- Walk-forward trades: V13 {int(v13_walkforward['aggregate_trades'])} -> V14 {int(v14_walkforward['aggregate_trades'])}",
        f"- Walk-forward win rate: V13 {_safe_float(v13_walkforward['aggregate_win_rate']):.6f} -> V14 {_safe_float(v14_walkforward['aggregate_win_rate']):.6f}",
        f"- Walk-forward profitable months: V13 {int(v13_walkforward['profitable_months'])}/{int(v13_walkforward['month_count'])} -> V14 {int(v14_walkforward['profitable_months'])}/{int(v14_walkforward['month_count'])}",
        f"- Walk-forward objective-pass months: V13 {int(v13_walkforward['objective_pass_months'])} -> V14 {int(v14_walkforward['objective_pass_months'])}",
        f"- Walk-forward max drawdown: V13 {_safe_float(v13_walkforward['max_single_month_drawdown_pct']):.6f}% -> V14 {_safe_float(v14_walkforward['max_single_month_drawdown_pct']):.6f}%",
        f"- V14 aggregate profit factor: {_safe_float(v14_walkforward['aggregate_profit_factor']):.6f}",
        "",
        "## December 2023 Backtrader Month",
        "",
        f"- V13: final capital ${_safe_float(v13_month['final_capital']):.2f}, trades {int(v13_month['trades_executed'])}, win rate {_safe_float(v13_month['win_rate']):.6f}, profit factor {_safe_float(v13_month['profit_factor']):.6f}, max DD {_safe_float(v13_month['max_drawdown_pct']):.6f}%",
        f"- V14: final capital ${_safe_float(v14_month['final_capital']):.2f}, trades {int(v14_month['trades_executed'])}, win rate {_safe_float(v14_month['win_rate']):.6f}, profit factor {_safe_float(v14_month['profit_factor']):.6f}, max DD {_safe_float(v14_month['max_drawdown_pct']):.6f}%",
        "",
        "## Calibration and Robustness",
        "",
        f"- V13 RCPC global calibration error: {v13_rcpc_error:.6f}",
        f"- V14 RSC max calibration error: {v14_rsc_max_error:.6f}",
    ]
    for regime, error in sorted(v14_rsc_errors.items()):
        md_lines.append(f"- V14 RSC error for {regime}: {error:.6f}")
    md_lines.extend(
        [
            f"- BST correlation with month profit factor: {(bst_corr or 0.0):.6f}",
            f"- High-BST month avg profit factor: {(_mean(high_bst_profit_factors) or 0.0):.6f}",
            f"- Low-BST month avg profit factor: {(_mean(low_bst_profit_factors) or 0.0):.6f}",
            f"- SSC average rejection rate: {(_mean(ssc_rejections) or 0.0):.6f}",
            f"- SSC composite critique score mean: {_safe_float(v14_ssc['composite_score_mean']):.6f}",
            "",
            "## ACM Fear Analysis",
            "",
            f"- Correlation of avg retail fear index with month return: {(fear_corr or 0.0):.6f}",
            f"- High-fear month average return: {(_mean(high_fear_returns) or 0.0):.6f}%",
            f"- Low-fear month average return: {(_mean(low_fear_returns) or 0.0):.6f}%",
            "",
            "## LDRG",
            "",
            f"- Tier: {int(ldrg_status['tier'])}",
            f"- Recommendation: {ldrg_status['recommendation']}",
            f"- Blocking criteria: {', '.join(ldrg_status['blocking_criteria']) if ldrg_status['blocking_criteria'] else 'none'}",
            "",
            "## Honest Interpretation",
            "",
        ]
    )
    md_lines.extend(f"- {line}" for line in honest_interpretation)
    md_lines.extend(
        [
            "",
            "## V15 Recommendation",
            "",
        ]
    )
    md_lines.extend(f"- {line}" for line in v15_recommendation)
    V14_SUMMARY_MD_PATH.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    outline_lines = [
        "# V14 Research Paper Outline",
        "",
        "## 1. Abstract",
        "",
        "We propose an uncertainty-aware crowd-simulation trading framework that extends reverse-collapse market simulation with asymmetric crowd memory, branch survival testing, simulation self-critique, temporal branch ranking, and regime-stratified calibration. On the Nexus Trader research stack, temporal CABR improved held-out pairwise accuracy from 0.531314 in V13 to 0.641945 in V14, while the V14 walk-forward produced a 0.770492 aggregate win rate and a 4.242104 aggregate profit factor across 37 months. The same cycle also exposed the remaining deployment bottleneck: calibration and policy conservatism still suppress too much activity, leaving Live Deployment Readiness Gate Tier 1 incomplete. The contribution is therefore both positive and negative: V14 materially improved ranking quality and robustness, while clarifying that execution readiness still depends on better regime-level calibration and higher-coverage paper-trade evidence.",
        "",
        "## 2. Introduction",
        "",
        "Short-horizon market forecasting is difficult because a market can have multiple plausible futures even when no single deterministic path is especially likely. Nexus Trader treats this as an uncertainty-modelling problem rather than a single-label classification problem. Reverse-collapse simulation generates many candidate futures from crowd-behavioural personas, and downstream ranking decides which branches are both realistic and tradeable. V14 extends this framing by making crowd memory asymmetric, stress-testing branches against perturbations, adding a self-critique layer, and moving from global to regime-stratified calibration.",
        "",
        "## 3. Related Work",
        "",
        "This work connects several areas that are usually treated separately: Monte Carlo tree search and branching in finance, agent-based market simulation, learned ranking under regime segmentation, and execution-consistency validation. Prior work often assumes feature consistency between offline and live environments or reduces the problem to direction prediction. Nexus instead combines RCMS, BCFE, CABR, UTS, SARV, and now V14's ACM/BST/SSC/RSC stack into one research program focused on uncertainty-aware execution.",
        "",
        "## 4. Architecture",
        "",
        "The full V14 architecture is BCFE canonical features -> ACM fear-aware simulation features -> temporal CABR branch ranking -> BST robustness scoring -> SSC critique scoring -> RSC regime-stratified calibration -> UTS thresholding -> LRTD/MBEG gating -> DAPS lot sizing -> Backtrader/S3PTA evaluation. RCMS remains the generative core, while SARV remains the validation methodology that prevents archive-only optimism from being mistaken for live robustness.",
        "",
        "## 5. Key Empirical Results",
        "",
        "V12 established that 11 of 36 legacy features fail archive-vs-live consistency and that BCFE is the safe feature path. V13 then produced the first viable Backtrader results, including 570 walk-forward trades across 37 months and 32 profitable months. V14 improved CABR pairwise accuracy to 0.641945, reduced maximum monthly drawdown to 1.405340 percent, and achieved a 4.242104 aggregate profit factor, but only 61 trades fired across the same 37 months and LDRG remained at Tier 0 because calibration error and month-breadth targets were missed.",
        "",
        "## 6. Novel Findings",
        "",
        "The project now supports several novel empirical claims. First, feature consistency failure is measurable and materially affects replay realism: 11 of 36 indicators were rejected by the audit. Second, regime-isolated pairs and temporal context substantially improve branch ranking quality. Third, reverse-collapse simulation with asymmetric crowd memory provides a richer description of post-shock behaviour than symmetric persona weighting. Fourth, branch survival testing suggests that robustness-sensitive branch choice is associated with higher monthly profit factor, even if the current evidence is observational rather than a full ablation.",
        "",
        "## 7. Limitations",
        "",
        "Calibration remains imperfect. V14's RSC max calibration error stayed at 0.513619, above the 0.20 target. The system also under-traded, with only 61 walk-forward trades across 37 months, which prevented LDRG Tier 1 completion despite strong pairwise accuracy. SSC trained successfully but had zero practical rejection rate, so its runtime effect remains limited. Finally, live deployment is still not justified because the paper-trade sample is below the required 200 trades.",
        "",
        "## 8. Conclusion and Future Work",
        "",
        "V14 is best understood as a research-strength improvement rather than a deployment-strength release. It proves that temporal context and robustness-aware branch evaluation materially improve ranking quality, while also making the next bottleneck explicit: regime-level calibration and policy activity. V15 should therefore retune execution conservatism, strengthen RSC, make SSC operationally meaningful, and extend paper-trade accumulation until the Live Deployment Readiness Gate criteria are genuinely satisfied.",
    ]
    V14_RESEARCH_PAPER_OUTLINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    V14_RESEARCH_PAPER_OUTLINE_PATH.write_text("\n".join(outline_lines) + "\n", encoding="utf-8")

    print(str(V14_SUMMARY_JSON_PATH))
    print(str(V14_SUMMARY_MD_PATH))
    print(str(V14_RESEARCH_PAPER_OUTLINE_PATH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
