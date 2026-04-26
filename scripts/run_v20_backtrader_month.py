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

from config.project_config import V20_BACKTEST_MONTH_REPORT_PATH, V20_CONFORMAL_CONE_PATH, V20_HMM_MODEL_PATH
from src.v12.bar_consistent_features import load_default_raw_bars
from src.v12.bar_consistent_features import compute_bar_consistent_features
from src.v16.confidence_tier import ConfidenceTier, classify_confidence
from src.v16.sel import sel_lot_size
from src.v20.conformal_cone import ConformalCone
from src.v20.frequency_features import build_frequency_feature_frame
from src.v20.macro_features import compute_macro_features
from src.v20.regime_detector import train_hmm
from src.v20.rl_executor import HierarchicalExecutor
from src.v20.runtime import _build_branch_candidates, _direction_signal, _safe_float
from src.v20.sjd_v20 import latest_sjd_decision


def _profit_factor(values: np.ndarray) -> float:
    gross_positive = float(values[values > 0.0].sum()) if np.any(values > 0.0) else 0.0
    gross_negative = float(abs(values[values < 0.0].sum())) if np.any(values < 0.0) else 0.0
    return float(gross_positive / gross_negative) if gross_negative > 0.0 else 0.0


def _train_local_hmm(raw_15m: pd.DataFrame, month_start: pd.Timestamp):
    train_bars = raw_15m.loc[raw_15m.index < month_start].copy()
    macro = compute_macro_features(train_bars)
    close = pd.to_numeric(train_bars["close"], errors="coerce").ffill().bfill()
    volume = pd.to_numeric(train_bars["volume"], errors="coerce").ffill().bfill()
    features = pd.DataFrame(index=train_bars.index)
    features["log_return"] = np.log(close).diff().fillna(0.0)
    features["realized_vol_20"] = pd.to_numeric(macro["macro_realized_vol_20"], errors="coerce").fillna(0.0)
    features["volume_zscore"] = ((volume - volume.rolling(96, min_periods=12).mean()) / volume.rolling(96, min_periods=12).std(ddof=0).replace(0.0, np.nan)).fillna(0.0)
    features["macro_vol_regime_class"] = pd.to_numeric(macro["macro_vol_regime_class"], errors="coerce").fillna(0.0)
    features["macro_jump_flag"] = pd.to_numeric(macro["macro_jump_flag"], errors="coerce").fillna(0.0)
    detector, _, _ = train_hmm(features)
    detector.save(V20_HMM_MODEL_PATH)
    return detector


def _build_fast_backtest_features(raw_15m: pd.DataFrame, detector) -> tuple[pd.DataFrame, dict[str, object]]:
    micro = compute_bar_consistent_features(raw_15m)
    macro = compute_macro_features(raw_15m)
    close = pd.to_numeric(micro["close"], errors="coerce").ffill().bfill()
    returns = close.pct_change().fillna(0.0)
    freq = build_frequency_feature_frame(close, window=min(240, len(close)))
    hurst_overall = (0.5 + returns.rolling(48, min_periods=12).apply(lambda values: float(np.corrcoef(values[:-1], values[1:])[0, 1]) if len(values) > 2 and np.std(values[:-1]) > 0 and np.std(values[1:]) > 0 else 0.0, raw=True).fillna(0.0).clip(-0.45, 0.45) * 0.4).clip(0.1, 0.9)
    hurst_positive = (hurst_overall + returns.clip(lower=0.0).rolling(48, min_periods=12).mean().fillna(0.0) * 10.0).clip(0.1, 0.95)
    hurst_negative = (hurst_overall + (-returns.clip(upper=0.0)).rolling(48, min_periods=12).mean().fillna(0.0) * 10.0).clip(0.1, 0.95)
    wltc_strength = (returns.clip(lower=0.0).rolling(32, min_periods=8).mean().fillna(0.0) * 40.0).clip(0.0, 1.0)
    mfg_mean_belief = returns.ewm(span=24, adjust=False).mean().fillna(0.0)
    mfg_disagreement = returns.rolling(24, min_periods=6).std(ddof=0).fillna(0.0) * 25.0
    regime_source = pd.DataFrame(
        {
            "log_return": np.log(close).diff().fillna(0.0),
            "realized_vol_20": pd.to_numeric(macro["macro_realized_vol_20"], errors="coerce").fillna(0.0),
            "volume_zscore": ((pd.to_numeric(raw_15m["volume"], errors="coerce").ffill().bfill() - pd.to_numeric(raw_15m["volume"], errors="coerce").ffill().bfill().rolling(96, min_periods=12).mean()) / pd.to_numeric(raw_15m["volume"], errors="coerce").ffill().bfill().rolling(96, min_periods=12).std(ddof=0).replace(0.0, np.nan)).fillna(0.0),
            "macro_vol_regime_class": pd.to_numeric(macro["macro_vol_regime_class"], errors="coerce").fillna(0.0),
            "macro_jump_flag": pd.to_numeric(macro["macro_jump_flag"], errors="coerce").fillna(0.0),
        },
        index=raw_15m.index,
    )
    regime = detector.transform(regime_source)
    features = pd.concat(
        [
            micro,
            macro,
            freq,
            pd.DataFrame(
                {
                    "hurst_overall": hurst_overall,
                    "hurst_positive": hurst_positive,
                    "hurst_negative": hurst_negative,
                    "hurst_asymmetry": hurst_positive - hurst_negative,
                    "mfg_mean_belief": mfg_mean_belief,
                    "mfg_disagreement": mfg_disagreement,
                    "wltc_strength": wltc_strength,
                },
                index=micro.index,
            ),
            regime,
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return features, {"feature_count": int(len(features.columns))}


def _calibrate_conformal(feature_frame: pd.DataFrame, month_start: pd.Timestamp) -> ConformalCone:
    calibration = feature_frame.loc[feature_frame.index < month_start].tail(96 * 90).copy()
    predictions: list[list[float]] = []
    realized: list[list[float]] = []
    regimes: list[int] = []
    for idx in range(len(calibration) - 1):
        row = calibration.iloc[idx].to_dict()
        row["direction_signal"] = _direction_signal(row)
        current_price = _safe_float(row.get("close"), 0.0)
        next_close = _safe_float(calibration.iloc[idx + 1].get("close"), current_price)
        pred_close = current_price * (1.0 + (_safe_float(row.get("direction_signal"), 0.0) * max(_safe_float(row.get("atr_pct"), 0.001), 1e-4) * 1.8))
        predictions.append([current_price, (current_price + pred_close) / 2.0, pred_close])
        realized.append([current_price, (current_price + next_close) / 2.0, next_close])
        regimes.append(int(_safe_float(row.get("hmm_state"), 0.0)))
    cone = ConformalCone(alpha=0.15)
    cone.calibrate(predictions, realized, regimes)
    cone.save(V20_CONFORMAL_CONE_PATH)
    return cone


def _report_path(month: str) -> Path:
    return PROJECT_ROOT / "outputs" / "v20" / f"backtest_month_{month.replace('-', '_')}_v20.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the V20 15-minute month backtest.")
    parser.add_argument("--month", default="2023-12")
    parser.add_argument("--mode", default="frequency", choices=["frequency", "precision"])
    parser.add_argument("--capital", type=float, default=1000.0)
    parser.add_argument("--spread-pips", type=float, default=0.5)
    parser.add_argument("--commission-usd", type=float, default=7.0)
    parser.add_argument("--pip-size", type=float, default=0.1)
    parser.add_argument("--pip-value-per-lot", type=float, default=10.0)
    args = parser.parse_args()

    month_start = pd.Timestamp(f"{args.month}-01", tz="UTC")
    month_end = month_start + pd.offsets.MonthBegin(1)
    lookback_start = month_start - pd.Timedelta(days=180)
    raw = load_default_raw_bars(start=lookback_start, end=month_end + pd.Timedelta(days=2))
    raw_15m = raw.resample("15min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    detector = _train_local_hmm(raw_15m, month_start)
    features, metadata = _build_fast_backtest_features(raw_15m, detector)
    cone = _calibrate_conformal(features, month_start)
    executor = HierarchicalExecutor()

    month_frame = features.loc[(features.index >= month_start) & (features.index < month_end)].copy()
    market_frame = raw_15m.reindex(month_frame.index)
    equity = float(args.capital)
    peak = equity
    max_drawdown = 0.0
    trades: list[dict[str, object]] = []
    skips: dict[str, int] = {}

    for idx in range(len(month_frame) - 1):
        row = month_frame.iloc[idx].to_dict()
        row["direction_signal"] = _direction_signal(row)
        branches = _build_branch_candidates(row)
        if branches.empty:
            skips["no_branches"] = skips.get("no_branches", 0) + 1
            continue
        best = branches.iloc[0].to_dict()
        current_price = _safe_float(market_frame.iloc[idx]["close"], 0.0)
        next_bar = market_frame.iloc[idx + 1]
        entry_price = _safe_float(next_bar["open"], current_price)
        consensus_price = float(branches.head(5)["predicted_price_15m"].mean())
        consensus_path = np.asarray([current_price, (current_price + consensus_price) / 2.0, consensus_price], dtype=np.float64)
        upper, lower, conformal_confidence = cone.predict(consensus_path, int(_safe_float(row.get("hmm_state"), 0.0)))
        cone_width_pips = abs(float(upper[-1] - lower[-1])) / max(float(args.pip_size), 1e-9)
        cpm_score = float(
            np.clip(
                0.30 * _safe_float(row.get("quant_route_confidence"), 0.5)
                + 0.20 * _safe_float(row.get("quant_regime_strength"), 0.5)
                + 0.15 * (1.0 - min(1.0, abs(_safe_float(row.get("mfg_disagreement"), 0.5))))
                + 0.15 * np.clip(abs(_safe_float(row.get("direction_signal"), 0.0)), 0.0, 1.0)
                + 0.20 * _safe_float(best.get("generator_probability"), 0.5),
                0.0,
                1.0,
            )
        )
        tier = classify_confidence(float(best.get("v20_cabr_score", 0.5)), max(float(conformal_confidence), 0.5), cone_width_pips, cpm_score=cpm_score)
        sjd = latest_sjd_decision(row)
        decision = executor.decide(
            regime_probs=[float(_safe_float(row.get(f"hmm_prob_{i}"), 0.0)) for i in range(6)],
            direction_signal=float(_safe_float(row.get("direction_signal"), 0.0)),
            confidence=float(best.get("v20_cabr_score", 0.5)),
            macro_state=[
                _safe_float(row.get("macro_trend_strength"), 0.0),
                _safe_float(row.get("macro_dxy_zscore_20d"), 0.0),
                _safe_float(row.get("macro_dxy_zscore_60d"), 0.0),
                _safe_float(row.get("macro_realized_vol_20"), 0.0),
                _safe_float(row.get("macro_realized_vol_60"), 0.0),
                _safe_float(row.get("mfg_disagreement"), 0.0),
                _safe_float(row.get("hurst_overall"), 0.5),
                _safe_float(row.get("spectral_entropy"), 0.0),
            ],
            volatility=_safe_float(row.get("macro_realized_vol_20"), 0.0),
            kelly_fraction=sjd.kelly,
            branch_rewards=[float((value - current_price) * (1.0 if str(best.get("decision_direction", "BUY")).upper() == "BUY" else -1.0)) for value in branches["predicted_price_15m"].head(16).tolist()],
        )
        direction_signal = float(_safe_float(row.get("direction_signal"), 0.0))
        cabr_score = float(best.get("v20_cabr_score", 0.5))
        alignment_ok = sjd.final_call == decision.action
        allowed = (
            decision.action in {"BUY", "SELL"}
            and tier in {ConfidenceTier.VERY_HIGH, ConfidenceTier.HIGH, ConfidenceTier.MODERATE}
            and cabr_score >= 0.64
            and cpm_score >= 0.52
            and abs(direction_signal) >= 0.14
            and alignment_ok
        )
        if args.mode == "precision" and tier not in {ConfidenceTier.VERY_HIGH, ConfidenceTier.HIGH}:
            allowed = False
        if not allowed:
            reason_parts = [f"hold_{tier.value}", decision.action.lower()]
            if cabr_score < 0.64:
                reason_parts.append("cabr")
            if cpm_score < 0.52:
                reason_parts.append("cpm")
            if abs(direction_signal) < 0.14:
                reason_parts.append("signal")
            if not alignment_ok:
                reason_parts.append("sjd")
            reason = "_".join(reason_parts)
            skips[reason] = skips.get(reason, 0) + 1
            executor.record_outcome(regime_probs=[float(_safe_float(row.get(f"hmm_prob_{i}"), 0.0)) for i in range(6)], action="HOLD", reward=0.0)
            continue

        lot = sel_lot_size(
            equity=equity,
            confidence_tier=tier,
            sqt_label="GOOD" if conformal_confidence >= 0.85 else "NEUTRAL",
            mode=args.mode,
            stop_pips=max(sjd.sl_offset, 12.0),
            pip_value_per_lot=float(args.pip_value_per_lot),
            max_lot=0.20,
        )
        stop_loss_price = entry_price - (sjd.sl_offset * float(args.pip_size)) if decision.action == "BUY" else entry_price + (sjd.sl_offset * float(args.pip_size))
        take_profit_price = entry_price + (sjd.tp_offset * float(args.pip_size)) if decision.action == "BUY" else entry_price - (sjd.tp_offset * float(args.pip_size))
        next_high = _safe_float(next_bar["high"], entry_price)
        next_low = _safe_float(next_bar["low"], entry_price)
        next_close = _safe_float(next_bar["close"], entry_price)
        exit_reason = "bar_close"
        if decision.action == "BUY":
            if next_low <= stop_loss_price:
                exit_price = stop_loss_price
                exit_reason = "stop_loss"
            elif next_high >= take_profit_price:
                exit_price = take_profit_price
                exit_reason = "take_profit"
            else:
                exit_price = next_close
        else:
            if next_high >= stop_loss_price:
                exit_price = stop_loss_price
                exit_reason = "stop_loss"
            elif next_low <= take_profit_price:
                exit_price = take_profit_price
                exit_reason = "take_profit"
            else:
                exit_price = next_close
        direction_sign = 1.0 if decision.action == "BUY" else -1.0
        gross_pips = ((exit_price - entry_price) * direction_sign) / max(float(args.pip_size), 1e-9)
        commission_pips = float(args.commission_usd) / max(float(args.pip_value_per_lot), 1e-9)
        net_pips = gross_pips - float(args.spread_pips) - commission_pips
        pnl_usd = net_pips * float(args.pip_value_per_lot) * lot
        equity += pnl_usd
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, 0.0 if peak <= 0.0 else (peak - equity) / peak)
        trades.append(
            {
                "timestamp": month_frame.index[idx].isoformat(),
                "direction": decision.action,
                "entry_price": round(entry_price, 5),
                "exit_price": round(exit_price, 5),
                "gross_pips": round(float(gross_pips), 6),
                "net_pips": round(float(net_pips), 6),
                "lot": round(float(lot), 4),
                "pnl_usd": round(float(pnl_usd), 6),
                "equity_after_trade": round(float(equity), 6),
                "return_pct": round(float(pnl_usd / max(args.capital, 1e-9)), 6),
                "exit_reason": exit_reason,
                "stop_loss_price": round(float(stop_loss_price), 5),
                "take_profit_price": round(float(take_profit_price), 5),
                "cabr_score": round(float(best.get("v20_cabr_score", 0.0)), 6),
                "cpm_score": round(float(cpm_score), 6),
                "conformal_confidence": round(float(conformal_confidence), 6),
                "confidence_tier": tier.value,
                "active_sub_agent": decision.active_sub_agent,
                "kelly_fraction": round(float(sjd.kelly), 6),
            }
        )
        executor.record_outcome(regime_probs=[float(_safe_float(row.get(f"hmm_prob_{i}"), 0.0)) for i in range(6)], action=decision.action, reward=float(pnl_usd))

    pnl = np.asarray([_safe_float(item.get("pnl_usd"), 0.0) for item in trades], dtype=np.float64)
    pips = np.asarray([_safe_float(item.get("net_pips"), 0.0) for item in trades], dtype=np.float64)
    lots = np.asarray([_safe_float(item.get("lot"), 0.0) for item in trades], dtype=np.float64)
    report = {
        "version": "v20",
        "engine": "v20_native_month",
        "month": args.month,
        "mode": args.mode,
        "start_capital": round(float(args.capital), 6),
        "final_capital": round(float(equity), 6),
        "net_profit": round(float(equity - args.capital), 6),
        "return_pct": round(((float(equity) / float(args.capital)) - 1.0) * 100.0, 6),
        "trades_executed": int(len(trades)),
        "win_rate": round(float(np.mean(pnl > 0.0)) if pnl.size else 0.0, 6),
        "profit_factor": round(_profit_factor(pnl), 6),
        "max_drawdown_pct": round(float(max_drawdown * 100.0), 6),
        "net_pips": round(float(pips.sum()) if pips.size else 0.0, 6),
        "avg_lot": round(float(lots.mean()) if lots.size else 0.0, 6),
        "skip_breakdown": skips,
        "feature_count": int(metadata.get("feature_count", 0)),
        "hmm_model_path": str(V20_HMM_MODEL_PATH),
        "conformal_path": str(V20_CONFORMAL_CONE_PATH),
        "trades": trades,
    }

    out_path = _report_path(args.month)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.month == "2023-12":
        V20_BACKTEST_MONTH_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        V20_BACKTEST_MONTH_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
