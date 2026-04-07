from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.service.llm_sidecar import request_local_sjd_judge
from src.v16.confidence_tier import ConfidenceTier
from src.v19.runtime import (
    build_archive_v19_candidate_frame,
    build_sjd_context_from_candidate,
    infer_sqt_label,
    load_v19_branch_archive,
    mode_allows_trade,
    predict_lepl_action,
    score_v19_candidates,
    suggested_lot_for_trade,
)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
        if np.isnan(number) or np.isinf(number):
            return float(default)
        return number
    except Exception:
        return float(default)


def _confidence_scalar(raw: object) -> float:
    return {
        "VERY_LOW": 0.0,
        "LOW": 1.0,
        "MODERATE": 2.0,
        "HIGH": 3.0,
    }.get(str(raw or "LOW").strip().upper(), 1.0)


def _trade_cost_pips(*, spread_pips: float, commission_usd: float, contract_size_oz: float, pip_size: float) -> float:
    commission_pips = float(commission_usd) / max(float(contract_size_oz) * float(pip_size), 1e-6)
    return float(spread_pips + commission_pips)


def _profit_factor(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    gross_positive = float(values[values > 0.0].sum()) if np.any(values > 0.0) else 0.0
    gross_negative = float(abs(values[values < 0.0].sum())) if np.any(values < 0.0) else 0.0
    if gross_negative <= 0.0:
        return 0.0
    return float(gross_positive / gross_negative)


def _out_path(month: str, mode: str) -> Path:
    return PROJECT_ROOT / "outputs" / "v19" / f"backtrader_month_{month.replace('-', '_')}_{mode}_native.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the native V19 month evaluation using the V19 branch archive, local SJD, and LEPL.")
    parser.add_argument("--month", default="2023-12")
    parser.add_argument("--mode", default="frequency", choices=["frequency", "precision"])
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--capital", type=float, default=1000.0)
    parser.add_argument("--spread-pips", type=float, default=0.5)
    parser.add_argument("--commission-usd", type=float, default=7.0)
    parser.add_argument("--contract-size-oz", type=float, default=100.0)
    parser.add_argument("--pip-size", type=float, default=0.1)
    parser.add_argument("--min-cabr-score", type=float, default=0.52)
    parser.add_argument("--min-cpm-score", type=float, default=0.48)
    parser.add_argument("--strict-lepl", action="store_true")
    args = parser.parse_args()

    archive = load_v19_branch_archive(args.month)
    if archive.empty:
        raise SystemExit(f"No V19 branch-archive rows were available for {args.month}.")

    candidate_frame = build_archive_v19_candidate_frame(archive)
    scored = score_v19_candidates(candidate_frame)
    if scored.empty:
        raise SystemExit(f"V19 scoring produced no candidate rows for {args.month}.")

    scored["timestamp"] = pd.to_datetime(scored["timestamp"], utc=True, errors="coerce")
    scored = scored.loc[scored["timestamp"].notna()].copy()
    top_rows = (
        scored.sort_values(["sample_id", "v19_cabr_raw_score"], ascending=[True, False])
        .groupby("sample_id", as_index=False)
        .first()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    cost_pips = _trade_cost_pips(
        spread_pips=float(args.spread_pips),
        commission_usd=float(args.commission_usd),
        contract_size_oz=float(args.contract_size_oz),
        pip_size=float(args.pip_size),
    )
    pip_value_per_lot = float(args.contract_size_oz) * float(args.pip_size)
    equity = float(args.capital)
    peak_equity = equity
    max_drawdown = 0.0
    next_available = pd.Timestamp.min.tz_localize("UTC")
    trades: list[dict[str, object]] = []
    skip_breakdown: dict[str, int] = {}
    lepl_breakdown: dict[str, int] = {}
    local_call_breakdown: dict[str, int] = {}
    tier_breakdown: dict[str, int] = {}
    sqt_breakdown: dict[str, int] = {}

    for row in top_rows.to_dict(orient="records"):
        timestamp = pd.Timestamp(row.get("timestamp"))
        if timestamp < next_available:
            skip_breakdown["cooldown_15m"] = skip_breakdown.get("cooldown_15m", 0) + 1
            continue

        context = build_sjd_context_from_candidate(row)
        local_envelope = request_local_sjd_judge(str(args.symbol).upper(), context)
        local_content = dict(local_envelope.get("content", {}) if isinstance(local_envelope.get("content"), dict) else {})
        local_call = str(local_content.get("final_call", "SKIP")).upper()
        local_call_breakdown[local_call] = local_call_breakdown.get(local_call, 0) + 1

        sqt_label = str(row.get("sqt_label", infer_sqt_label(_safe_float(row.get("cpm_score"), 0.0), _safe_float(row.get("v19_cabr_score"), 0.0)))).upper()
        sqt_breakdown[sqt_label] = sqt_breakdown.get(sqt_label, 0) + 1
        tier = ConfidenceTier(str(row.get("confidence_tier", "low")))
        tier_breakdown[tier.value] = tier_breakdown.get(tier.value, 0) + 1

        lepl_action, lepl_probabilities, lepl_features = predict_lepl_action(
            local_judge_content=local_content,
            row=row,
            has_open_position=False,
            open_position_pnl=0.0,
        )
        lepl_breakdown[lepl_action] = lepl_breakdown.get(lepl_action, 0) + 1

        if not mode_allows_trade(args.mode, tier):
            skip_breakdown[f"mode_gate_{tier.value}"] = skip_breakdown.get(f"mode_gate_{tier.value}", 0) + 1
            continue
        if _safe_float(row.get("v19_cabr_score"), 0.0) < float(args.min_cabr_score):
            skip_breakdown["cabr_below_threshold"] = skip_breakdown.get("cabr_below_threshold", 0) + 1
            continue
        if _safe_float(row.get("cpm_score"), 0.0) < float(args.min_cpm_score):
            skip_breakdown["cpm_below_threshold"] = skip_breakdown.get("cpm_below_threshold", 0) + 1
            continue

        permissive_frequency_pass = (
            args.mode == "frequency"
            and _confidence_scalar(local_content.get("confidence", "LOW")) >= 2.0
            and _safe_float(row.get("v19_cabr_score"), 0.0) >= float(args.min_cabr_score)
            and _safe_float(row.get("cpm_score"), 0.0) >= float(args.min_cpm_score)
        )
        if args.strict_lepl:
            policy_allows = lepl_action == "ENTER"
        else:
            policy_allows = lepl_action == "ENTER" or permissive_frequency_pass
        if not policy_allows:
            skip_breakdown[f"lepl_{lepl_action.lower()}"] = skip_breakdown.get(f"lepl_{lepl_action.lower()}", 0) + 1
            continue

        runtime_call = str(row.get("decision_direction", "HOLD")).upper()
        if runtime_call not in {"BUY", "SELL"}:
            skip_breakdown["runtime_hold"] = skip_breakdown.get("runtime_hold", 0) + 1
            continue

        entry_price = _safe_float(row.get("entry_open_price"), _safe_float(row.get("anchor_price"), 0.0))
        exit_price = _safe_float(row.get("actual_price_15m"), _safe_float(row.get("exit_close_price_15m"), entry_price))
        stop_loss = local_content.get("stop_loss")
        if stop_loss is not None:
            stop_pips = abs(entry_price - _safe_float(stop_loss, entry_price)) / max(float(args.pip_size), 1e-9)
        else:
            stop_pips = max(10.0, min(35.0, _safe_float(row.get("cone_width_pips"), 15.0) * 0.35))
        lot = suggested_lot_for_trade(
            equity=equity,
            tier=tier,
            sqt_label=sqt_label,
            mode=args.mode,
            stop_pips=stop_pips,
            pip_value_per_lot=pip_value_per_lot,
        )
        direction_sign = 1.0 if runtime_call == "BUY" else -1.0
        gross_pips = ((exit_price - entry_price) * direction_sign) / max(float(args.pip_size), 1e-9)
        net_pips = gross_pips - cost_pips
        pnl_usd = net_pips * pip_value_per_lot * lot
        equity += pnl_usd
        peak_equity = max(peak_equity, equity)
        max_drawdown = max(max_drawdown, 0.0 if peak_equity <= 0.0 else (peak_equity - equity) / peak_equity)
        next_available = timestamp + pd.Timedelta(minutes=15)
        trades.append(
            {
                "timestamp": timestamp.isoformat(),
                "direction": runtime_call,
                "entry_price": round(entry_price, 5),
                "exit_price": round(exit_price, 5),
                "gross_pips": round(gross_pips, 6),
                "net_pips": round(net_pips, 6),
                "lot": round(lot, 2),
                "pnl_usd": round(pnl_usd, 6),
                "equity_after_trade": round(equity, 6),
                "cabr_score": round(_safe_float(row.get("v19_cabr_score"), 0.0), 6),
                "cpm_score": round(_safe_float(row.get("cpm_score"), 0.0), 6),
                "confidence_tier": tier.value,
                "sqt_label": sqt_label,
                "runtime_call": runtime_call,
                "local_call": local_call,
                "local_agrees_with_runtime": bool(local_call in {"BUY", "SELL"} and local_call == runtime_call),
                "local_confidence": str(local_content.get("confidence", "LOW")).upper(),
                "local_summary": str(local_content.get("final_summary", "")),
                "lepl_action": lepl_action,
                "lepl_probabilities": {str(key): round(_safe_float(value), 6) for key, value in lepl_probabilities.items()},
                "lepl_features": {key: (round(_safe_float(value), 6) if isinstance(value, (int, float)) else value) for key, value in lepl_features.items()},
            }
        )

    trade_pnls = np.asarray([_safe_float(item.get("pnl_usd"), 0.0) for item in trades], dtype=np.float64)
    trade_pips = np.asarray([_safe_float(item.get("net_pips"), 0.0) for item in trades], dtype=np.float64)
    lots = np.asarray([_safe_float(item.get("lot"), 0.0) for item in trades], dtype=np.float64)
    report = {
        "version": "v19",
        "engine": "v19_native_month",
        "mode": args.mode,
        "month": args.month,
        "symbol": str(args.symbol).upper(),
        "source_archive": str(PROJECT_ROOT / "outputs" / "v19" / "branch_archive_100k.parquet"),
        "research_constraints": {
            "start_capital_usd": float(args.capital),
            "spread_pips": float(args.spread_pips),
            "commission_usd_round_trip": float(args.commission_usd),
            "horizon_minutes": 15,
            "native_v19_stack": True,
            "strict_lepl": bool(args.strict_lepl),
        },
        "candidate_rows": int(len(scored)),
        "top_branch_rows": int(len(top_rows)),
        "trades_executed": int(len(trades)),
        "start_capital": round(float(args.capital), 6),
        "final_capital": round(float(equity), 6),
        "net_profit": round(float(equity - float(args.capital)), 6),
        "return_pct": round(((float(equity) / float(args.capital)) - 1.0) * 100.0, 6),
        "win_rate": round(float(np.mean(trade_pnls > 0.0)) if len(trades) else 0.0, 6),
        "profit_factor": round(_profit_factor(trade_pnls), 6),
        "max_drawdown_pct": round(float(max_drawdown * 100.0), 6),
        "net_pips": round(float(trade_pips.sum()) if len(trades) else 0.0, 6),
        "avg_win_pips": round(float(np.mean(trade_pips[trade_pips > 0.0])) if np.any(trade_pips > 0.0) else 0.0, 6),
        "avg_loss_pips": round(float(np.mean(trade_pips[trade_pips < 0.0])) if np.any(trade_pips < 0.0) else 0.0, 6),
        "lot_summary": {
            "min_lot": round(float(lots.min()) if len(lots) else 0.0, 6),
            "max_lot": round(float(lots.max()) if len(lots) else 0.0, 6),
            "avg_lot": round(float(lots.mean()) if len(lots) else 0.0, 6),
        },
        "local_call_breakdown": local_call_breakdown,
        "lepl_action_breakdown": lepl_breakdown,
        "confidence_tier_breakdown": tier_breakdown,
        "sqt_breakdown": sqt_breakdown,
        "skip_reason_breakdown": skip_breakdown,
        "trade_log": trades,
    }

    out_path = _out_path(args.month, args.mode)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(str(out_path), flush=True)
    print(json.dumps({k: v for k, v in report.items() if k != "trade_log"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
