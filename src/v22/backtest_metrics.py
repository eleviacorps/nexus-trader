from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _trade_pnl(trade: Mapping[str, Any]) -> float:
    for key in ("pnl_usd", "profit", "net_pnl", "realized_pnl"):
        if key in trade:
            return _safe_float(trade.get(key), 0.0)
    return 0.0


def _trade_lot(trade: Mapping[str, Any]) -> float:
    for key in ("lot", "volume", "size"):
        if key in trade:
            return _safe_float(trade.get(key), 0.0)
    return 0.0


def _streaks(values: Sequence[float]) -> tuple[int, int]:
    longest_win = 0
    longest_loss = 0
    current_win = 0
    current_loss = 0
    for value in values:
        if value > 0.0:
            current_win += 1
            current_loss = 0
        elif value < 0.0:
            current_loss += 1
            current_win = 0
        else:
            current_win = 0
            current_loss = 0
        longest_win = max(longest_win, current_win)
        longest_loss = max(longest_loss, current_loss)
    return longest_win, longest_loss


def compute_trade_health_metrics(trades: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pnl = np.asarray([_trade_pnl(trade) for trade in trades], dtype=np.float64)
    lots = np.asarray([_trade_lot(trade) for trade in trades], dtype=np.float64)
    winners = pnl[pnl > 0.0]
    losers = pnl[pnl < 0.0]
    avg_winner = float(winners.mean()) if winners.size else 0.0
    avg_loser_abs = float(abs(losers.mean())) if losers.size else 0.0
    realized_rr = float(avg_winner / avg_loser_abs) if avg_loser_abs > 0.0 else 0.0
    longest_win, longest_loss = _streaks(pnl.tolist())
    directions = [str(trade.get("direction", "")).upper() for trade in trades]
    direction_flips = sum(1 for idx in range(1, len(directions)) if directions[idx] and directions[idx - 1] and directions[idx] != directions[idx - 1])
    return {
        "trade_count": int(len(trades)),
        "avg_realized_rr": round(realized_rr, 6),
        "average_winner_usd": round(avg_winner, 6),
        "average_loser_usd_abs": round(avg_loser_abs, 6),
        "longest_win_streak": int(longest_win),
        "longest_loss_streak": int(longest_loss),
        "direction_flips": int(direction_flips),
        "win_rate": round(float(np.mean(pnl > 0.0)) if pnl.size else 0.0, 6),
        "avg_lot": round(float(lots.mean()) if lots.size else 0.0, 6),
        "median_lot": round(float(np.median(lots)) if lots.size else 0.0, 6),
        "max_lot": round(float(lots.max()) if lots.size else 0.0, 6),
    }


def attach_v22_month_metrics(report: Mapping[str, Any], *, mode: str | None = None) -> dict[str, Any]:
    output = dict(report)
    trades = list(output.get("trades", []) if isinstance(output.get("trades"), list) else [])
    trade_metrics = compute_trade_health_metrics(trades)
    trade_count = int(output.get("trade_count", output.get("trades_executed", trade_metrics["trade_count"])) or trade_metrics["trade_count"])
    output["trade_health"] = trade_metrics
    output["v22_new_metrics"] = {
        "circuit_breaker_triggers_per_month": output.get("circuit_breaker_triggers_per_month"),
        "avg_realized_rr": trade_metrics["avg_realized_rr"],
        "direction_persistence_violations": output.get("direction_persistence_violations"),
        "ensemble_agreement_rate": output.get("ensemble_agreement_rate"),
        "discriminator_rejection_rate": output.get("discriminator_rejection_rate"),
        "meta_label_accept_rate": output.get("meta_label_accept_rate"),
        "regime_confidence_avg": output.get("regime_confidence_avg"),
        "research_mode_win_rate": output.get("research_mode_win_rate"),
        "production_mode_win_rate": output.get("production_mode_win_rate", trade_metrics["win_rate"]),
        "production_trades_per_month": trade_count,
        "trade_frequency_target_met": 50 <= trade_count <= 200,
        "longest_loss_streak": trade_metrics["longest_loss_streak"],
        "direction_flips": trade_metrics["direction_flips"],
        "avg_lot": trade_metrics["avg_lot"],
    }
    output["evaluation_mode"] = mode or output.get("mode")
    return output


__all__ = ["attach_v22_month_metrics", "compute_trade_health_metrics"]
