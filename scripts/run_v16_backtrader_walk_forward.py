from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config.project_config import (
    V14_BRANCH_FEATURES_PATH,
    V14_CABR_TEMPORAL_MODEL_PATH,
    V16_BACKTRADER_WALKFORWARD_FREQUENCY_PATH,
    V16_BACKTRADER_WALKFORWARD_PRECISION_PATH,
    V17_CABR_MODEL_PATH,
    V17_MMM_FEATURES_PATH,
)
from src.v12.tctl import replay_candidates_with_online_bcfe
from src.v13.cabr import augment_cabr_context, load_cabr_model, load_v13_candidate_frames, score_cabr_model
from src.v16.confidence_tier import ConfidenceTier, classify_confidence


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(value)))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return float(default)
        return number
    except Exception:
        return float(default)


def _mode_allows_trade(mode: str, tier: ConfidenceTier) -> bool:
    normalized = str(mode).strip().lower()
    if normalized == "precision":
        return tier in {ConfidenceTier.HIGH, ConfidenceTier.VERY_HIGH}
    return tier in {ConfidenceTier.MODERATE, ConfidenceTier.HIGH, ConfidenceTier.VERY_HIGH}


def _series(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype=np.float64)


def _report_path(mode: str) -> Path:
    if str(mode).strip().lower() == "precision":
        return PROJECT_ROOT / "outputs" / "v16" / "backtrader_walkforward_precision.json"
    return PROJECT_ROOT / "outputs" / "v16" / "backtrader_walkforward_frequency.json"


def _checkpoint_path(version: str) -> Path:
    if str(version).strip().lower() == "v17":
        return V17_CABR_MODEL_PATH
    return V14_CABR_TEMPORAL_MODEL_PATH


def _mmm_frame() -> pd.DataFrame | None:
    if not V17_MMM_FEATURES_PATH.exists():
        return None
    return pd.read_parquet(V17_MMM_FEATURES_PATH)


def _prepare_holdout_frame(cabr_version: str) -> tuple[pd.DataFrame, str]:
    archive = pd.read_parquet(V14_BRANCH_FEATURES_PATH)
    _, valid_frame, _, _ = load_v13_candidate_frames(
        archive,
        use_temporal_context=True,
        n_context_bars=12,
    )
    replay = replay_candidates_with_online_bcfe(valid_frame)
    replay = augment_cabr_context(replay, mmm_features=_mmm_frame() if str(cabr_version).strip().lower() == "v17" else None)

    checkpoint = _checkpoint_path(cabr_version)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing CABR checkpoint for {cabr_version}: {checkpoint}")
    model, branch_cols, context_cols, model_payload = load_cabr_model(path=checkpoint, map_location="cpu")
    for column in branch_cols:
        if column not in replay.columns:
            replay[column] = 0.0
    for column in context_cols:
        if column not in replay.columns:
            replay[column] = 0.5 if "hurst" in column else 0.0
    replay = replay.sort_values(["timestamp", "sample_id", "branch_id"]).reset_index(drop=True)
    replay["cabr_raw_score"] = score_cabr_model(
        model,
        replay,
        branch_feature_names=branch_cols,
        context_feature_names=context_cols,
        device="cpu",
    )
    replay["cabr_score_mean"] = float(np.mean(replay["cabr_raw_score"].to_numpy(dtype=np.float32)))
    replay["cabr_score_std"] = float(np.std(replay["cabr_raw_score"].to_numpy(dtype=np.float32)) or 1.0)
    replay["timestamp"] = pd.to_datetime(replay["timestamp"], utc=True, errors="coerce")
    replay["month"] = replay["timestamp"].dt.to_period("M").astype(str)
    return replay, str(model_payload.get("checkpoint_path", checkpoint))


def _selected_trades(replay: pd.DataFrame, mode: str, spread_pips: float, commission_usd: float, pip_size: float, contract_size_oz: float) -> pd.DataFrame:
    if replay.empty:
        return replay.copy()
    chosen = replay.sort_values(["sample_id", "cabr_raw_score"], ascending=[True, False]).groupby("sample_id", as_index=False).first()
    score_mean = float(chosen["cabr_raw_score"].mean())
    score_std = float(chosen["cabr_raw_score"].std(ddof=0) or 1.0)
    cabr_norm = np.vectorize(lambda x: _sigmoid((float(x) - score_mean) / max(score_std, 1e-6)))(chosen["cabr_raw_score"].to_numpy(dtype=np.float64))
    bst_proxy = (
        0.30 * _series(chosen, "order_flow_plausibility", 0.5).to_numpy()
        + 0.25 * _series(chosen, "execution_realism", 0.5).to_numpy()
        + 0.25 * _series(chosen, "regime_consistency", 0.5).to_numpy()
        + 0.20 * _series(chosen, "inside_confidence_cone", 0.5).to_numpy()
    )
    cpm_proxy = (
        0.35 * _series(chosen, "model_confidence_prob_15m", 0.5).to_numpy()
        + 0.25 * _series(chosen, "generator_probability", 0.5).to_numpy()
        + 0.20 * _series(chosen, "branch_confidence", 0.5).to_numpy()
        + 0.20 * (1.0 - np.clip(np.abs(_series(chosen, "path_error", 0.0).to_numpy()), 0.0, 1.0))
    )
    cone_width_pips = (
        np.abs(_series(chosen, "predicted_price_15m", 0.0).to_numpy() - _series(chosen, "anchor_price", 0.0).to_numpy())
        / max(float(pip_size), 1e-9)
    )

    tiers = [
        classify_confidence(float(cabr_norm[idx]), float(np.clip(bst_proxy[idx], 0.0, 1.0)), float(cone_width_pips[idx]), float(np.clip(cpm_proxy[idx], 0.0, 1.0)))
        for idx in range(len(chosen))
    ]
    chosen["confidence_tier"] = [tier.value for tier in tiers]
    chosen["should_trade"] = [_mode_allows_trade(mode, tier) for tier in tiers]

    anchor_price = _series(chosen, "anchor_price", 0.0)
    predicted_terminal = _series(chosen, "predicted_price_15m", 0.0)
    chosen["predicted_terminal"] = predicted_terminal.where(predicted_terminal != 0.0, anchor_price).to_numpy()
    chosen["direction"] = np.where(chosen["predicted_terminal"] >= anchor_price.to_numpy(), 1.0, -1.0)
    entry_price = _series(chosen, "entry_open_price", 0.0)
    chosen["entry_price"] = entry_price.where(entry_price != 0.0, anchor_price).to_numpy()
    exit_price = _series(chosen, "actual_price_15m", 0.0)
    fallback_exit = _series(chosen, "exit_close_price_15m", 0.0).where(_series(chosen, "exit_close_price_15m", 0.0) != 0.0, anchor_price)
    chosen["exit_price"] = exit_price.where(exit_price != 0.0, fallback_exit).to_numpy()
    chosen["gross_pips"] = ((chosen["exit_price"] - chosen["entry_price"]) * chosen["direction"]) / max(float(pip_size), 1e-9)
    commission_pips = float(commission_usd) / max(float(contract_size_oz) * float(pip_size), 1e-6)
    cost_pips = float(spread_pips) + commission_pips
    chosen["net_pips"] = chosen["gross_pips"] - cost_pips
    chosen["cone_hit"] = chosen.get("inside_confidence_cone", 0.0).astype(float).clip(0.0, 1.0)

    filtered = chosen.loc[chosen["should_trade"]].copy().sort_values("timestamp").reset_index(drop=True)
    if filtered.empty:
        return filtered

    selected_rows: list[dict[str, object]] = []
    next_available = pd.Timestamp.min.tz_localize("UTC")
    for row in filtered.itertuples(index=False):
        timestamp = pd.Timestamp(row.timestamp)
        if timestamp < next_available:
            continue
        selected_rows.append(row._asdict())
        next_available = timestamp + pd.Timedelta(minutes=15)
    return pd.DataFrame(selected_rows)


def _month_summary(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "cone_hit_rate": 0.0,
            "net_pips": 0.0,
        }
    net_pips = frame["net_pips"].to_numpy(dtype=np.float64)
    profits = float(net_pips[net_pips > 0.0].sum()) if np.any(net_pips > 0.0) else 0.0
    losses = float(np.abs(net_pips[net_pips < 0.0].sum())) if np.any(net_pips < 0.0) else 0.0
    return {
        "trades": int(len(frame)),
        "win_rate": round(float(np.mean(net_pips > 0.0)), 6),
        "profit_factor": round(profits / losses, 6) if losses > 0.0 else 0.0,
        "cone_hit_rate": round(float(frame["cone_hit"].mean()), 6),
        "net_pips": round(float(net_pips.sum()), 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the deferred V16 15-minute walk-forward evaluation.")
    parser.add_argument("--mode", default="frequency", choices=["frequency", "precision"])
    parser.add_argument("--cabr_version", default="v14")
    parser.add_argument("--capital", type=float, default=1000.0)
    parser.add_argument("--spread_pips", type=float, default=0.5)
    parser.add_argument("--commission_usd", type=float, default=7.0)
    parser.add_argument("--pip_size", type=float, default=0.1)
    parser.add_argument("--contract_size_oz", type=float, default=100.0)
    parser.add_argument("--broker_leverage", type=float, default=1.0)
    args = parser.parse_args()

    replay, checkpoint_used = _prepare_holdout_frame(args.cabr_version)
    trades = _selected_trades(
        replay,
        mode=args.mode,
        spread_pips=float(args.spread_pips),
        commission_usd=float(args.commission_usd),
        pip_size=float(args.pip_size),
        contract_size_oz=float(args.contract_size_oz),
    )

    months = []
    for month in sorted(replay["month"].dropna().unique().tolist()):
        subset = trades.loc[trades["month"] == month].copy() if not trades.empty else pd.DataFrame()
        months.append({"month": str(month)} | _month_summary(subset))

    trade_pips = trades["net_pips"].to_numpy(dtype=np.float64) if not trades.empty else np.asarray([], dtype=np.float64)
    profits = float(trade_pips[trade_pips > 0.0].sum()) if np.any(trade_pips > 0.0) else 0.0
    losses = float(np.abs(trade_pips[trade_pips < 0.0].sum())) if np.any(trade_pips < 0.0) else 0.0
    worst_month = min(months, key=lambda item: (float(item["net_pips"]), float(item["profit_factor"]))) if months else None
    target_trade_floor = 80 if args.mode == "precision" else 400
    target_win_rate = 0.65 if args.mode == "precision" else 0.58
    target_profit_factor = 0.0 if args.mode == "precision" else 1.8
    report = {
        "version": "v16",
        "engine": "v16_proxy_walkforward",
        "mode": args.mode,
        "cabr_version": args.cabr_version,
        "checkpoint_used": checkpoint_used,
        "research_constraints": {
            "start_capital_usd": float(args.capital),
            "spread_pips": float(args.spread_pips),
            "commission_usd_round_trip": float(args.commission_usd),
            "broker_leverage": float(args.broker_leverage),
            "horizon_minutes": 15,
        },
        "aggregate_trades": int(len(trades)),
        "month_count": int(len(months)),
        "aggregate_win_rate": round(float(np.mean(trade_pips > 0.0)) if len(trades) else 0.0, 6),
        "aggregate_profit_factor": round(profits / losses, 6) if losses > 0.0 else 0.0,
        "cone_hit_rate": round(float(trades["cone_hit"].mean()) if len(trades) else 0.0, 6),
        "months": months,
        "worst_month": worst_month,
        "targets": {
            "trade_floor": target_trade_floor,
            "win_rate": target_win_rate,
            "profit_factor": target_profit_factor,
            "cone_hit_rate": 0.55,
        },
        "targets_met": {
            "trade_floor": int(len(trades)) >= target_trade_floor,
            "win_rate": (float(np.mean(trade_pips > 0.0)) if len(trades) else 0.0) > target_win_rate,
            "profit_factor": True if args.mode == "precision" else (round(profits / losses, 6) if losses > 0.0 else 0.0) > target_profit_factor,
            "cone_hit_rate": (float(trades["cone_hit"].mean()) if len(trades) else 0.0) > 0.55,
        },
        "trade_log_preview": (
            trades.loc[:, ["timestamp", "month", "confidence_tier", "net_pips", "cone_hit"]]
            .head(25)
            .to_dict(orient="records")
            if not trades.empty
            else []
        ),
    }
    if worst_month is not None and (
        not report["targets_met"]["trade_floor"]
        or not report["targets_met"]["win_rate"]
        or not report["targets_met"]["profit_factor"]
        or not report["targets_met"]["cone_hit_rate"]
    ):
        report["worst_month_reason"] = (
            f"{worst_month['month']} had {worst_month['trades']} trades, "
            f"win rate {worst_month['win_rate']:.2%}, "
            f"profit factor {worst_month['profit_factor']:.4f}, "
            f"cone hit rate {worst_month['cone_hit_rate']:.2%}, "
            f"net pips {worst_month['net_pips']:.2f}."
        )

    out_path = _report_path(args.mode)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    config_path = V16_BACKTRADER_WALKFORWARD_PRECISION_PATH if args.mode == "precision" else V16_BACKTRADER_WALKFORWARD_FREQUENCY_PATH
    config_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(str(out_path), flush=True)
    print(json.dumps({k: v for k, v in report.items() if k not in {"months", "trade_log_preview"}}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
