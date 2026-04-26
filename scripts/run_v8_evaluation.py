from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None

from config.project_config import V8_BRANCH_ARCHIVE_PATH, V8_BRANCH_SELECTOR_PATH, V8_EVALUATION_REPORT_PATH  # noqa: E402
from src.v8.branch_selector_v8 import evaluate_branch_selector, load_branch_selector_v8  # noqa: E402


def tagged_path(path: Path, run_tag: str) -> Path:
    if not run_tag:
        return path
    return path.with_name(f"{path.stem}_{run_tag}{path.suffix}")


def load_archive(path: Path):
    if pd is None:
        raise ImportError("pandas is required for v8 evaluation.")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _net_unit_pnl(direction: float, entry_open: float, exit_close: float, volatility_scale: float, confidence: float) -> float:
    gross_return = float(direction) * ((float(exit_close) - float(entry_open)) / max(float(entry_open), 1e-6))
    if gross_return > 0:
        gross_unit = 1.0
    elif gross_return < 0:
        gross_unit = -1.0
    else:
        gross_unit = 0.0
    fee_penalty = 0.0004
    slippage_penalty = 0.0002 + (0.0002 * min(max(float(volatility_scale), 0.0), 4.0) / 4.0) + (0.0001 * (1.0 - min(max(float(confidence), 0.0), 1.0)))
    return float(gross_unit - fee_penalty - (2.0 * slippage_penalty))


def _build_trade_summary(records: list[dict[str, float]]) -> dict[str, Any]:
    if not records:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "avg_unit_pnl": 0.0,
            "long_win_rate": 0.0,
            "short_win_rate": 0.0,
        }
    pnl = np.asarray([record["net_unit_pnl"] for record in records], dtype=np.float32)
    direction = np.asarray([record["direction"] for record in records], dtype=np.float32)
    wins = pnl > 0
    long_mask = direction > 0
    short_mask = direction < 0
    cumulative = np.cumsum(pnl)
    peak = np.maximum.accumulate(cumulative) if len(cumulative) else np.asarray([], dtype=np.float32)
    drawdown = peak - cumulative if len(cumulative) else np.asarray([], dtype=np.float32)
    return {
        "trade_count": int(len(records)),
        "win_rate": round(float(np.mean(wins)), 6),
        "avg_unit_pnl": round(float(np.mean(pnl)), 6),
        "cumulative_unit_pnl": round(float(cumulative[-1]) if len(cumulative) else 0.0, 6),
        "max_drawdown_units": round(float(np.max(drawdown)) if len(drawdown) else 0.0, 6),
        "long_win_rate": round(float(np.mean(wins[long_mask])) if long_mask.any() else 0.0, 6),
        "short_win_rate": round(float(np.mean(wins[short_mask])) if short_mask.any() else 0.0, 6),
    }


def _feature_importance(selector_payload: dict[str, Any]) -> dict[str, float]:
    model = selector_payload.get("model")
    feature_names = list(selector_payload.get("feature_names", []))
    if model is None or not feature_names:
        return {}
    raw_values = None
    if hasattr(model, "feature_importances_"):
        raw_values = getattr(model, "feature_importances_")
    elif hasattr(model, "coef_"):
        coef = np.asarray(getattr(model, "coef_"), dtype=np.float32)
        raw_values = np.abs(coef.reshape(-1))
    if raw_values is None:
        return {}
    values = np.asarray(raw_values, dtype=np.float32).reshape(-1)
    limit = min(len(values), len(feature_names))
    pairs = {feature_names[index]: float(values[index]) for index in range(limit)}
    return dict(sorted(pairs.items(), key=lambda item: item[1], reverse=True))


def _cone_metrics(group) -> tuple[float, float]:
    ranked = group.sort_values("selector_score", ascending=False).head(3)
    paths = ranked[["predicted_price_5m", "predicted_price_10m", "predicted_price_15m"]].to_numpy(dtype=np.float32)
    actual = ranked.iloc[0][["actual_price_5m", "actual_price_10m", "actual_price_15m"]].to_numpy(dtype=np.float32)
    if paths.size == 0:
        return 0.0, 0.0
    lower = np.min(paths, axis=0)
    upper = np.max(paths, axis=0)
    inside = (actual >= lower) & (actual <= upper)
    return float(np.mean(inside)), float(np.all(inside))


def _evaluate_execution(frame) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    selector_records: list[dict[str, float]] = []
    baseline_records: list[dict[str, float]] = []
    cone_point_rates: list[float] = []
    cone_full_rates: list[float] = []
    regime_stats: dict[str, dict[str, float]] = {}
    minority_useful: list[float] = []

    for _, group in frame.groupby("sample_id", sort=False):
        ranked = group.sort_values("selector_score", ascending=False)
        selected = ranked.iloc[0]
        baseline = group.sort_values("generator_probability", ascending=False).iloc[0]
        cone_point, cone_full = _cone_metrics(group)
        cone_point_rates.append(cone_point)
        cone_full_rates.append(cone_full)
        for chosen, bucket in ((selected, selector_records), (baseline, baseline_records)):
            direction = float(chosen["branch_direction"])
            net_pnl = _net_unit_pnl(
                direction=direction,
                entry_open=float(chosen["entry_open_price"]),
                exit_close=float(chosen["exit_close_price_15m"]),
                volatility_scale=float(chosen.get("volatility_scale", 1.0)),
                confidence=float(chosen.get("branch_confidence", chosen.get("generator_probability", 0.5))),
            )
            bucket.append(
                {
                    "direction": direction,
                    "net_unit_pnl": net_pnl,
                }
            )
        if len(ranked) > 1:
            minority = ranked.iloc[1]
            minority_useful.append(
                float(
                    int(
                        np.sign(float(minority["actual_final_return"])) == np.sign(float(minority["branch_direction"]))
                        and float(minority["branch_direction"]) != float(selected["branch_direction"])
                    )
                )
            )
        regime = str(selected.get("dominant_regime", "unknown"))
        bucket = regime_stats.setdefault(regime, {"count": 0.0, "selector_win": 0.0, "baseline_win": 0.0})
        bucket["count"] += 1.0
        bucket["selector_win"] += 1.0 if selector_records[-1]["net_unit_pnl"] > 0 else 0.0
        bucket["baseline_win"] += 1.0 if baseline_records[-1]["net_unit_pnl"] > 0 else 0.0

    for payload in regime_stats.values():
        count = max(payload["count"], 1.0)
        payload["selector_win_rate"] = round(payload["selector_win"] / count, 6)
        payload["baseline_win_rate"] = round(payload["baseline_win"] / count, 6)

    selector_summary = _build_trade_summary(selector_records)
    baseline_summary = _build_trade_summary(baseline_records)
    cone_summary = {
        "top3_point_containment_rate": round(float(np.mean(cone_point_rates)) if cone_point_rates else 0.0, 6),
        "top3_full_containment_rate": round(float(np.mean(cone_full_rates)) if cone_full_rates else 0.0, 6),
        "minority_branch_usefulness": round(float(np.mean(minority_useful)) if minority_useful else 0.0, 6),
        "regime_execution_performance": regime_stats,
    }
    return selector_summary, baseline_summary, cone_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the V8 branch selector on a held-out branch archive.")
    parser.add_argument("--archive", default="")
    parser.add_argument("--selector", default="")
    parser.add_argument("--run-tag", default="")
    args = parser.parse_args()

    archive_path = Path(args.archive) if args.archive else tagged_path(V8_BRANCH_ARCHIVE_PATH, args.run_tag)
    if not archive_path.exists():
        csv_fallback = archive_path.with_suffix(".csv")
        if csv_fallback.exists():
            archive_path = csv_fallback
        else:
            raise FileNotFoundError(f"Branch archive not found: {archive_path}")
    selector_path = Path(args.selector) if args.selector else tagged_path(V8_BRANCH_SELECTOR_PATH, args.run_tag)
    if not selector_path.exists():
        json_fallback = selector_path.with_suffix(".json")
        if json_fallback.exists():
            selector_path = json_fallback
        else:
            raise FileNotFoundError(f"Branch selector not found: {selector_path}")

    frame = load_archive(archive_path)
    selector = load_branch_selector_v8(selector_path)
    base_metrics = evaluate_branch_selector(frame, selector)
    frame = frame.copy()
    frame["selector_score"] = selector.score(frame)
    selector_summary, baseline_summary, cone_summary = _evaluate_execution(frame)

    report = {
        "run_tag": args.run_tag,
        "archive_path": str(archive_path),
        "selector_path": str(selector_path),
        "selector_provider": str(selector.payload.get("provider", "none")),
        "top1_branch_accuracy": float(base_metrics.get("top1_branch_accuracy", 0.0)),
        "top3_branch_containment": float(base_metrics.get("top3_branch_containment", 0.0)),
        "average_selected_path_error": float(base_metrics.get("average_selected_path_error", 0.0)),
        "average_generator_baseline_error": float(base_metrics.get("average_generator_baseline_error", 0.0)),
        "selector_error_improvement": float(base_metrics.get("selector_error_improvement", 0.0)),
        "direction_only_event_win_rate": float(base_metrics.get("event_driven_15m_win_rate", 0.0)),
        "selector_event_driven_15m": selector_summary,
        "generator_probability_baseline_15m": baseline_summary,
        "cone_metrics": cone_summary,
        "regime_performance": base_metrics.get("regime_performance", {}),
        "feature_importance": _feature_importance(selector.payload),
    }
    report_path = tagged_path(V8_EVALUATION_REPORT_PATH, args.run_tag)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
