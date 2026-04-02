from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .fees import FeeModel, ZeroFeeModel
from .results import BacktestSummary, TradeRecord
from .slippage import NoSlippageModel, SlippageModel


@dataclass(frozen=True)
class DirectionalBacktestConfig:
    decision_threshold: float = 0.5
    confidence_floor: float = 0.12
    gate_threshold: float = 0.5
    hold_threshold: float = 0.5
    fee_model: FeeModel = ZeroFeeModel()
    slippage_model: SlippageModel = NoSlippageModel()


def confidence_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float32)
    return np.clip(np.abs(values - 0.5) * 2.0, 0.0, 1.0)


def capital_backtest_from_unit_pnl(
    pnl: np.ndarray,
    *,
    initial_capital: float,
    risk_fraction: float = 0.02,
) -> dict[str, Any]:
    pnl = np.asarray(pnl, dtype=np.float32)
    equity = float(initial_capital)
    peak = equity
    max_drawdown_pct = 0.0
    winning_trades = 0
    losing_trades = 0
    trade_count = 0
    overflowed = False
    log10_equity = float(np.log10(max(initial_capital, 1e-12)))
    for unit in pnl:
        if float(unit) == 0.0:
            peak = max(peak, equity)
            drawdown_pct = 0.0 if peak <= 0 else (peak - equity) / peak
            max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
            continue
        trade_count += 1
        growth_factor = max(1e-12, 1.0 + float(risk_fraction) * float(unit))
        log10_equity += float(np.log10(growth_factor))
        if not overflowed:
            equity *= growth_factor
            if not np.isfinite(equity) or equity > 1e308:
                overflowed = True
        if unit > 0:
            winning_trades += 1
        else:
            losing_trades += 1
        if not overflowed:
            peak = max(peak, equity)
            drawdown_pct = 0.0 if peak <= 0 else (peak - equity) / peak
            max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
    if overflowed:
        return {
            "initial_capital": round(float(initial_capital), 6),
            "mode": "compounding_r_multiple",
            "risk_fraction": float(risk_fraction),
            "final_capital": None,
            "net_profit": None,
            "return_pct": None,
            "max_drawdown_pct": round(float(max_drawdown_pct * 100.0), 6),
            "trade_count": int(trade_count),
            "winning_trades": int(winning_trades),
            "losing_trades": int(losing_trades),
            "overflowed": True,
            "log10_final_capital": round(float(log10_equity), 6),
        }
    return {
        "initial_capital": round(float(initial_capital), 6),
        "mode": "compounding_r_multiple",
        "risk_fraction": float(risk_fraction),
        "final_capital": round(float(equity), 6),
        "net_profit": round(float(equity - initial_capital), 6),
        "return_pct": round(float((equity / initial_capital - 1.0) * 100.0) if initial_capital > 0 else 0.0, 6),
        "max_drawdown_pct": round(float(max_drawdown_pct * 100.0), 6),
        "trade_count": int(trade_count),
        "winning_trades": int(winning_trades),
        "losing_trades": int(losing_trades),
        "overflowed": False,
        "log10_final_capital": round(float(np.log10(max(equity, 1e-12))), 6),
    }


def fixed_risk_capital_backtest_from_unit_pnl(
    pnl: np.ndarray,
    *,
    initial_capital: float,
    risk_fraction: float = 0.02,
) -> dict[str, Any]:
    pnl = np.asarray(pnl, dtype=np.float32)
    stake = float(initial_capital) * float(risk_fraction)
    equity = float(initial_capital)
    equity_curve = []
    winning_trades = 0
    losing_trades = 0
    trade_count = 0
    for unit in pnl:
        if float(unit) == 0.0:
            equity_curve.append(equity)
            continue
        trade_count += 1
        equity = max(0.0, equity + stake * float(unit))
        if unit > 0:
            winning_trades += 1
        else:
            losing_trades += 1
        equity_curve.append(equity)
    equity_array = np.asarray(equity_curve, dtype=np.float32) if equity_curve else np.asarray([initial_capital], dtype=np.float32)
    final_capital = float(equity_array[-1]) if equity_array.size else float(initial_capital)
    peak = np.maximum.accumulate(equity_array) if equity_array.size else np.asarray([initial_capital], dtype=np.float32)
    drawdown = peak - equity_array if equity_array.size else np.asarray([0.0], dtype=np.float32)
    drawdown_pct = np.divide(drawdown, np.maximum(peak, 1e-6)) if drawdown.size else np.asarray([0.0], dtype=np.float32)
    return {
        "initial_capital": round(float(initial_capital), 6),
        "mode": "fixed_r_multiple",
        "risk_fraction": float(risk_fraction),
        "risk_amount": round(float(stake), 6),
        "final_capital": round(final_capital, 6),
        "net_profit": round(final_capital - float(initial_capital), 6),
        "return_pct": round(float((final_capital / initial_capital - 1.0) * 100.0) if initial_capital > 0 else 0.0, 6),
        "max_drawdown_pct": round(float(drawdown_pct.max() * 100.0), 6),
        "trade_count": int(trade_count),
        "winning_trades": int(winning_trades),
        "losing_trades": int(losing_trades),
    }


def directional_backtest(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    decision_threshold: float = 0.5,
    confidence_floor: float = 0.12,
    gate_scores: np.ndarray | None = None,
    gate_threshold: float = 0.5,
    hold_probabilities: np.ndarray | None = None,
    hold_threshold: float = 0.5,
    confidence_probabilities: np.ndarray | None = None,
    fee_model: FeeModel | None = None,
    slippage_model: SlippageModel | None = None,
    volatility_scale: np.ndarray | None = None,
    return_trades: bool = False,
) -> dict[str, Any]:
    targets = np.asarray(targets, dtype=np.float32)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    fee_model = fee_model or ZeroFeeModel()
    slippage_model = slippage_model or NoSlippageModel()
    confidence = confidence_from_probabilities(probabilities)
    long_mask = (probabilities >= decision_threshold) & (confidence >= confidence_floor)
    short_mask = (probabilities <= (1.0 - decision_threshold)) & (confidence >= confidence_floor)
    hold_values = np.asarray(hold_probabilities, dtype=np.float32) if hold_probabilities is not None else None
    if hold_values is not None:
        predicted_hold = hold_values >= float(hold_threshold)
        long_mask = long_mask & (~predicted_hold)
        short_mask = short_mask & (~predicted_hold)
    explicit_conf = np.asarray(confidence_probabilities, dtype=np.float32) if confidence_probabilities is not None else None
    if explicit_conf is not None:
        long_mask = long_mask & (explicit_conf >= confidence_floor)
        short_mask = short_mask & (explicit_conf >= confidence_floor)
    gate_values = np.asarray(gate_scores, dtype=np.float32) if gate_scores is not None else None
    if gate_values is not None:
        gate_mask = gate_values >= float(gate_threshold)
        long_mask = long_mask & gate_mask
        short_mask = short_mask & gate_mask
    volatility_values = np.asarray(volatility_scale, dtype=np.float32) if volatility_scale is not None else None

    signals = np.zeros(len(probabilities), dtype=np.int8)
    signals[long_mask] = 1
    signals[short_mask] = -1
    realized = np.where(targets >= 0.5, 1, -1)
    gross_pnl = np.where(signals == 0, 0.0, np.where(signals == realized, 1.0, -1.0)).astype(np.float32)
    net_pnl = gross_pnl.copy()
    trades: list[TradeRecord] = []
    for idx, direction in enumerate(signals.tolist()):
        if direction == 0:
            continue
        fee_penalty = float(fee_model.cost_fraction(probability=float(probabilities[idx]), confidence=float(confidence[idx]), direction=int(direction)))
        slip_penalty = float(
            slippage_model.cost_fraction(
                probability=float(probabilities[idx]),
                confidence=float(confidence[idx]),
                direction=int(direction),
                volatility_scale=float(volatility_values[idx]) if volatility_values is not None else 1.0,
            )
        )
        net_pnl[idx] = float(gross_pnl[idx]) - fee_penalty - slip_penalty
        if return_trades:
            trades.append(
                TradeRecord(
                    index=int(idx),
                    direction=int(direction),
                    probability=float(probabilities[idx]),
                    confidence=float(confidence[idx]),
                    realized_direction=int(realized[idx]),
                    gross_unit_pnl=float(gross_pnl[idx]),
                    net_unit_pnl=float(net_pnl[idx]),
                    fee_penalty=fee_penalty,
                    slippage_penalty=slip_penalty,
                    hold_probability=float(hold_values[idx]) if hold_values is not None else None,
                    confidence_probability=float(explicit_conf[idx]) if explicit_conf is not None else None,
                    gate_score=float(gate_values[idx]) if gate_values is not None else None,
                )
            )

    cumulative = np.cumsum(net_pnl)
    peak = np.maximum.accumulate(cumulative) if cumulative.size else np.empty(0, dtype=np.float32)
    drawdown = peak - cumulative if cumulative.size else np.empty(0, dtype=np.float32)
    trade_mask = signals != 0
    wins = (net_pnl > 0) & trade_mask
    losses = (net_pnl < 0) & trade_mask
    long_trades = signals == 1
    short_trades = signals == -1
    summary = BacktestSummary(
        trade_count=int(trade_mask.sum()),
        hold_count=int((signals == 0).sum()),
        participation_rate=round(float(trade_mask.mean()) if len(signals) else 0.0, 6),
        win_rate=round(float(wins.sum() / max(1, trade_mask.sum())), 6),
        loss_rate=round(float(losses.sum() / max(1, trade_mask.sum())), 6),
        long_win_rate=round(float(((net_pnl > 0) & long_trades).sum() / max(1, long_trades.sum())), 6),
        short_win_rate=round(float(((net_pnl > 0) & short_trades).sum() / max(1, short_trades.sum())), 6),
        avg_unit_pnl=round(float(net_pnl.mean()) if len(net_pnl) else 0.0, 6),
        gross_avg_unit_pnl=round(float(gross_pnl.mean()) if len(gross_pnl) else 0.0, 6),
        cumulative_unit_pnl=round(float(cumulative[-1]) if len(cumulative) else 0.0, 6),
        gross_cumulative_unit_pnl=round(float(np.cumsum(gross_pnl)[-1]) if len(gross_pnl) else 0.0, 6),
        max_drawdown_units=round(float(drawdown.max()) if len(drawdown) else 0.0, 6),
        decision_threshold=float(decision_threshold),
        confidence_floor=float(confidence_floor),
        gate_threshold=float(gate_threshold),
        hold_threshold=float(hold_threshold),
        capital_backtests={
            "usd_10": capital_backtest_from_unit_pnl(net_pnl, initial_capital=10.0, risk_fraction=0.02),
            "usd_1000": capital_backtest_from_unit_pnl(net_pnl, initial_capital=1000.0, risk_fraction=0.02),
            "usd_10_fixed_risk": fixed_risk_capital_backtest_from_unit_pnl(net_pnl, initial_capital=10.0, risk_fraction=0.02),
            "usd_1000_fixed_risk": fixed_risk_capital_backtest_from_unit_pnl(net_pnl, initial_capital=1000.0, risk_fraction=0.02),
        },
        fee_model=fee_model.__class__.__name__,
        slippage_model=slippage_model.__class__.__name__,
    )
    payload = summary.to_dict()
    if return_trades:
        payload["trades"] = [trade.to_dict() for trade in trades]
    return payload
