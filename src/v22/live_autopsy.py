from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from config.project_config import OUTPUTS_V22_DIR, V20_HMM_MODEL_PATH
from src.v12.bar_consistent_features import compute_bar_consistent_features, load_default_raw_bars
from src.v20.macro_features import compute_macro_features
from src.v20.regime_detector import RegimeDetector, train_hmm


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _to_utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _trade_profit(trade: Mapping[str, Any]) -> float:
    for key in ("profit", "pnl_usd", "net_pnl", "realized_pnl"):
        if key in trade:
            return _safe_float(trade.get(key), 0.0)
    return 0.0


def _load_mt5_trade_records(
    *,
    session_date: str = "2026-04-10",
    symbol: str = "XAUUSD",
    magic: int = 21042026,
) -> list[dict[str, Any]]:
    try:
        import MetaTrader5 as mt5  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"MetaTrader5 import failed: {exc}") from exc

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    start = datetime.fromisoformat(f"{session_date}T00:00:00+00:00")
    end = start + timedelta(days=1)
    deals = mt5.history_deals_get(start, end)
    orders = mt5.history_orders_get(start, end)

    order_by_position: dict[int, dict[str, Any]] = {}
    for order in orders or []:
        position_id = _safe_int(getattr(order, "position_id", 0), 0)
        if position_id <= 0:
            continue
        current = {
            "position_id": position_id,
            "symbol": str(getattr(order, "symbol", "")),
            "type": _safe_int(getattr(order, "type", 0), 0),
            "volume_initial": _safe_float(getattr(order, "volume_initial", 0.0), 0.0),
            "sl": _safe_float(getattr(order, "sl", 0.0), 0.0),
            "tp": _safe_float(getattr(order, "tp", 0.0), 0.0),
            "comment": str(getattr(order, "comment", "")),
            "magic": _safe_int(getattr(order, "magic", 0), 0),
        }
        existing = order_by_position.get(position_id)
        if existing is None:
            order_by_position[position_id] = current
        else:
            existing_rr_fields = float(existing.get("sl", 0.0) or 0.0) > 0.0 or float(existing.get("tp", 0.0) or 0.0) > 0.0
            current_rr_fields = current["sl"] > 0.0 or current["tp"] > 0.0
            if current_rr_fields or not existing_rr_fields:
                order_by_position[position_id] = current

    grouped: dict[int, dict[str, Any]] = defaultdict(dict)
    for deal in deals or []:
        if str(getattr(deal, "symbol", "")) != symbol:
            continue
        position_id = _safe_int(getattr(deal, "position_id", 0), 0)
        if position_id <= 0:
            continue
        row = {
            "position_id": position_id,
            "time": datetime.fromtimestamp(_safe_int(getattr(deal, "time", 0), 0), tz=timezone.utc).isoformat(),
            "entry": _safe_int(getattr(deal, "entry", 0), 0),
            "type": _safe_int(getattr(deal, "type", 0), 0),
            "volume": _safe_float(getattr(deal, "volume", 0.0), 0.0),
            "price": _safe_float(getattr(deal, "price", 0.0), 0.0),
            "profit": _safe_float(getattr(deal, "profit", 0.0), 0.0),
            "comment": str(getattr(deal, "comment", "")),
            "magic": _safe_int(getattr(deal, "magic", 0), 0),
        }
        if row["entry"] == 0:
            grouped[position_id]["open"] = row
        elif row["entry"] == 1:
            grouped[position_id]["close"] = row

    trades: list[dict[str, Any]] = []
    for position_id, payload in grouped.items():
        opened = payload.get("open")
        closed = payload.get("close")
        if not opened or not closed:
            continue
        if int(opened.get("magic", 0)) != int(magic):
            comment_text = str(opened.get("comment", ""))
            if "NexusTrader" not in comment_text:
                continue
        direction = "BUY" if int(opened.get("type", 0)) == 0 else "SELL"
        entry_price = _safe_float(opened.get("price"), 0.0)
        stop_loss = _safe_float(order_by_position.get(position_id, {}).get("sl"), 0.0)
        take_profit = _safe_float(order_by_position.get(position_id, {}).get("tp"), 0.0)
        risk_distance = abs(entry_price - stop_loss) if stop_loss > 0.0 else 0.0
        reward_distance = abs(take_profit - entry_price) if take_profit > 0.0 else 0.0
        rr_pretrade = float(reward_distance / risk_distance) if risk_distance > 0.0 and reward_distance > 0.0 else None
        trades.append(
            {
                "position_id": position_id,
                "entry_time": str(opened.get("time")),
                "exit_time": str(closed.get("time")),
                "direction": direction,
                "volume": _safe_float(opened.get("volume"), 0.0),
                "entry_price": entry_price,
                "exit_price": _safe_float(closed.get("price"), 0.0),
                "profit": _safe_float(closed.get("profit"), 0.0),
                "entry_comment": str(opened.get("comment", "")),
                "exit_comment": str(closed.get("comment", "")),
                "magic": int(opened.get("magic", 0)),
                "stop_loss": stop_loss if stop_loss > 0.0 else None,
                "take_profit": take_profit if take_profit > 0.0 else None,
                "rr_pretrade": round(rr_pretrade, 6) if rr_pretrade is not None else None,
            }
        )
    trades.sort(key=lambda item: item["entry_time"])
    return trades


def _load_trade_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    if isinstance(payload, Mapping):
        rows = payload.get("trades", payload.get("positions", []))
        if isinstance(rows, list):
            return [dict(item) for item in rows if isinstance(item, Mapping)]
    return []


def _regime_frame_for_session(session_start: pd.Timestamp, session_end: pd.Timestamp) -> pd.DataFrame:
    raw = load_default_raw_bars(start=session_start - pd.Timedelta(days=90), end=session_end + pd.Timedelta(days=1))
    raw_15m = raw.resample("15min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    micro = compute_bar_consistent_features(raw_15m)
    macro = compute_macro_features(raw_15m)
    regime_source = pd.DataFrame(
        {
            "log_return": np.log(pd.to_numeric(raw_15m["close"], errors="coerce").ffill().bfill()).diff().fillna(0.0),
            "realized_vol_20": pd.to_numeric(macro["macro_realized_vol_20"], errors="coerce").fillna(0.0),
            "volume_zscore": (
                (
                    pd.to_numeric(raw_15m["volume"], errors="coerce").ffill().bfill()
                    - pd.to_numeric(raw_15m["volume"], errors="coerce").ffill().bfill().rolling(96, min_periods=12).mean()
                )
                / pd.to_numeric(raw_15m["volume"], errors="coerce").ffill().bfill().rolling(96, min_periods=12).std(ddof=0).replace(0.0, np.nan)
            ).fillna(0.0),
            "macro_vol_regime_class": pd.to_numeric(macro["macro_vol_regime_class"], errors="coerce").fillna(0.0),
            "macro_jump_flag": pd.to_numeric(macro["macro_jump_flag"], errors="coerce").fillna(0.0),
        },
        index=raw_15m.index,
    )
    if V20_HMM_MODEL_PATH.exists():
        detector = RegimeDetector.load(V20_HMM_MODEL_PATH)
    else:
        detector, _, _ = train_hmm(regime_source)
    regime = detector.transform(regime_source)
    frame = pd.concat([raw_15m, micro[["atr_pct"]], regime], axis=1)
    prob_cols = [column for column in frame.columns if str(column).startswith("hmm_prob_")]
    frame["regime_confidence"] = frame[prob_cols].max(axis=1) if prob_cols else 0.0
    trending_states = {"trending_up", "trending_down", "breakout"}
    frame["bar_context"] = frame["hmm_state_name"].apply(lambda value: "trending" if str(value) in trending_states else "ranging")
    return frame


def _attach_regime_context(trades: list[dict[str, Any]], session_start: pd.Timestamp, session_end: pd.Timestamp) -> list[dict[str, Any]]:
    try:
        regime_frame = _regime_frame_for_session(session_start, session_end)
    except Exception:
        return trades
    if regime_frame.empty:
        return trades
    enriched: list[dict[str, Any]] = []
    index = regime_frame.index
    for trade in trades:
        ts = _to_utc(trade.get("entry_time"))
        loc = index.asof(ts)
        if pd.isna(loc):
            enriched.append(dict(trade))
            continue
        if abs(ts - pd.Timestamp(loc)) > pd.Timedelta(minutes=30):
            enriched.append(dict(trade))
            continue
        row = regime_frame.loc[loc]
        item = dict(trade)
        item["entry_bar_time"] = pd.Timestamp(loc).isoformat()
        item["regime_state"] = str(row.get("hmm_state_name", "unknown"))
        item["regime_confidence"] = round(_safe_float(row.get("regime_confidence"), 0.0), 6)
        item["bar_context"] = str(row.get("bar_context", "unknown"))
        item["atr_pct"] = round(_safe_float(row.get("atr_pct"), 0.0), 6)
        enriched.append(item)
    return enriched


def _direction_metrics(trades: list[dict[str, Any]], direction: str) -> dict[str, Any]:
    subset = [trade for trade in trades if str(trade.get("direction", "")).upper() == direction]
    profits = np.asarray([_trade_profit(trade) for trade in subset], dtype=np.float64)
    winners = profits[profits > 0.0]
    losers = profits[profits < 0.0]
    return {
        "trades": int(len(subset)),
        "win_rate": round(float(np.mean(profits > 0.0)) if profits.size else 0.0, 6),
        "net_profit": round(float(profits.sum()) if profits.size else 0.0, 6),
        "avg_winner": round(float(winners.mean()) if winners.size else 0.0, 6),
        "avg_loser_abs": round(float(abs(losers.mean())) if losers.size else 0.0, 6),
    }


def _loss_streak_details(trades: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    longest = 0
    current = 0
    start_time = None
    best_start = None
    best_end = None
    for trade in trades:
        profit = _trade_profit(trade)
        if profit < 0.0:
            current += 1
            start_time = start_time or trade.get("entry_time")
            if current > longest:
                longest = current
                best_start = start_time
                best_end = trade.get("exit_time")
        else:
            current = 0
            start_time = None
    return {
        "max_consecutive_losses": int(longest),
        "streak_start": best_start,
        "streak_end": best_end,
    }


def _direction_streaks(trades: list[dict[str, Any]]) -> dict[str, Any]:
    longest = 0
    longest_direction = ""
    current = 0
    current_direction = ""
    flips = 0
    for trade in trades:
        direction = str(trade.get("direction", "")).upper()
        if direction and current_direction and direction != current_direction:
            flips += 1
        if direction == current_direction:
            current += 1
        else:
            current = 1 if direction else 0
            current_direction = direction
        if current > longest:
            longest = current
            longest_direction = direction
    return {
        "direction_flips": int(flips),
        "longest_same_direction_streak": int(longest),
        "longest_same_direction": longest_direction,
    }


def compute_live_session_diagnostic(trades: list[dict[str, Any]], *, session_date: str) -> dict[str, Any]:
    profits = np.asarray([_trade_profit(trade) for trade in trades], dtype=np.float64)
    winners = profits[profits > 0.0]
    losers = profits[profits < 0.0]
    volumes = [round(_safe_float(trade.get("volume"), 0.0), 2) for trade in trades]
    rr_values = [float(trade["rr_pretrade"]) for trade in trades if trade.get("rr_pretrade") is not None]
    regime_counts = Counter(str(trade.get("regime_state", "unknown")) for trade in trades if trade.get("regime_state"))
    bar_context_counts = Counter(str(trade.get("bar_context", "unknown")) for trade in trades if trade.get("bar_context"))
    avg_winner = float(winners.mean()) if winners.size else 0.0
    avg_loser_abs = float(abs(losers.mean())) if losers.size else 0.0
    realized_rr = float(avg_winner / avg_loser_abs) if avg_loser_abs > 0.0 else 0.0
    direction_baseline = _direction_streaks(trades)
    loss_baseline = _loss_streak_details(trades)
    result = {
        "session_date": session_date,
        "trade_count": int(len(trades)),
        "wins": int(np.sum(profits > 0.0)),
        "losses": int(np.sum(profits < 0.0)),
        "net_profit": round(float(profits.sum()) if profits.size else 0.0, 2),
        "win_rate": round(float(np.mean(profits > 0.0)) if profits.size else 0.0, 6),
        "avg_winner_usd": round(avg_winner, 6),
        "avg_loser_usd_abs": round(avg_loser_abs, 6),
        "realized_rr": round(realized_rr, 6),
        "avg_pretrade_rr": round(float(np.mean(rr_values)) if rr_values else 0.0, 6),
        "pretrade_rr_below_1_count": int(sum(1 for value in rr_values if value < 1.0)),
        "pretrade_rr_below_1_5_count": int(sum(1 for value in rr_values if value < 1.5)),
        "lot_size_distribution": {f"{key:.2f}": count for key, count in sorted(Counter(volumes).items())},
        "by_direction": {
            "BUY": _direction_metrics(trades, "BUY"),
            "SELL": _direction_metrics(trades, "SELL"),
        },
        "loss_streak": loss_baseline,
        "direction_pattern": direction_baseline,
        "regime_counts": dict(sorted(regime_counts.items())),
        "bar_context_counts": dict(sorted(bar_context_counts.items())),
        "retrospective_regime_available": bool(regime_counts),
        "regime_confidence_avg": round(float(np.mean([_safe_float(trade.get("regime_confidence"), 0.0) for trade in trades])), 6) if trades else 0.0,
        "acceptance_baseline": {
            "consecutive_losses_trigger_needed": bool(loss_baseline["max_consecutive_losses"] >= 3),
            "daily_drawdown_trigger_needed": bool(float(profits.sum()) <= -200.0),
            "too_many_sell_trades": bool(_direction_metrics(trades, "SELL")["trades"] >= 15),
            "rr_discipline_problem": bool(realized_rr < 1.0 or int(sum(1 for value in rr_values if value < 1.5)) > 0),
        },
        "notes": [
            "Retrospective HMM regime labels are unavailable when the local price archive does not cover the session date."
        ]
        if not regime_counts
        else [],
        "trades": trades,
    }
    return result


def build_live_session_diagnostic(
    *,
    session_date: str = "2026-04-10",
    source_path: str | Path | None = None,
    symbol: str = "XAUUSD",
    magic: int = 21042026,
) -> dict[str, Any]:
    if source_path is not None and Path(source_path).exists():
        trades = _load_trade_records(Path(source_path))
    else:
        trades = _load_mt5_trade_records(session_date=session_date, symbol=symbol, magic=magic)
    session_start = _to_utc(f"{session_date}T00:00:00+00:00")
    session_end = session_start + pd.Timedelta(days=1)
    trades = _attach_regime_context(trades, session_start, session_end)
    return compute_live_session_diagnostic(trades, session_date=session_date)


def write_live_session_diagnostic(payload: Mapping[str, Any], *, session_date: str) -> Path:
    OUTPUTS_V22_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_V22_DIR / f"live_session_diagnostic_{session_date.replace('-', '_')}.json"
    path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")
    return path


__all__ = [
    "build_live_session_diagnostic",
    "compute_live_session_diagnostic",
    "write_live_session_diagnostic",
]
