from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .events import FillEvent, MarketBar, SimOrder
from .fees import FeeModel, ZeroFeeModel
from .results import BacktestSummary, TradeRecord
from .slippage import NoSlippageModel, SlippageModel
from .engine import capital_backtest_from_unit_pnl, fixed_risk_capital_backtest_from_unit_pnl


@dataclass(frozen=True)
class EventDrivenBacktestConfig:
    hold_bars: int = 1
    position_size: float = 1.0
    fee_model: FeeModel = ZeroFeeModel()
    slippage_model: SlippageModel = NoSlippageModel()
    decision_threshold: float = 0.5
    confidence_floor: float = 0.12
    gate_threshold: float = 0.5
    hold_threshold: float = 0.5


def _validate_bar_arrays(open_prices: np.ndarray, high_prices: np.ndarray, low_prices: np.ndarray, close_prices: np.ndarray) -> int:
    lengths = {len(np.asarray(open_prices)), len(np.asarray(high_prices)), len(np.asarray(low_prices)), len(np.asarray(close_prices))}
    if len(lengths) != 1:
        raise ValueError("All OHLC arrays must have the same length.")
    length = lengths.pop()
    if length < 2:
        raise ValueError("Need at least 2 bars for event-driven backtesting.")
    return int(length)


def _build_market_bars(
    open_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    close_prices: np.ndarray,
    *,
    volumes: np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
) -> list[MarketBar]:
    length = _validate_bar_arrays(open_prices, high_prices, low_prices, close_prices)
    vol = np.asarray(volumes, dtype=np.float32) if volumes is not None else np.zeros(length, dtype=np.float32)
    ts = np.asarray(timestamps) if timestamps is not None else np.asarray([None] * length, dtype=object)
    return [
        MarketBar(
            index=idx,
            open=float(open_prices[idx]),
            high=float(high_prices[idx]),
            low=float(low_prices[idx]),
            close=float(close_prices[idx]),
            volume=float(vol[idx]),
            timestamp=None if ts[idx] is None else str(ts[idx]),
        )
        for idx in range(length)
    ]


def event_driven_directional_backtest(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    open_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    close_prices: np.ndarray,
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
    timestamps: np.ndarray | None = None,
    volumes: np.ndarray | None = None,
    hold_bars: int = 1,
    position_size: float = 1.0,
) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    confidence = np.clip(np.abs(probabilities - 0.5) * 2.0, 0.0, 1.0)
    fee_model = fee_model or ZeroFeeModel()
    slippage_model = slippage_model or NoSlippageModel()
    volatility_values = np.asarray(volatility_scale, dtype=np.float32) if volatility_scale is not None else None
    gate_values = np.asarray(gate_scores, dtype=np.float32) if gate_scores is not None else None
    hold_values = np.asarray(hold_probabilities, dtype=np.float32) if hold_probabilities is not None else None
    explicit_conf = np.asarray(confidence_probabilities, dtype=np.float32) if confidence_probabilities is not None else None

    bars = _build_market_bars(open_prices, high_prices, low_prices, close_prices, volumes=volumes, timestamps=timestamps)
    usable = min(len(probabilities), len(bars) - 1)
    gross_pnl = np.zeros(usable, dtype=np.float32)
    net_pnl = np.zeros(usable, dtype=np.float32)
    trades: list[TradeRecord] = []
    orders: list[SimOrder] = []
    fills: list[FillEvent] = []

    for idx in range(usable):
        prob = float(probabilities[idx])
        conf = float(confidence[idx])
        direction = 0
        if prob >= decision_threshold and conf >= confidence_floor:
            direction = 1
        elif prob <= (1.0 - decision_threshold) and conf >= confidence_floor:
            direction = -1
        if hold_values is not None and float(hold_values[idx]) >= hold_threshold:
            direction = 0
        if explicit_conf is not None and float(explicit_conf[idx]) < confidence_floor:
            direction = 0
        if gate_values is not None and float(gate_values[idx]) < gate_threshold:
            direction = 0
        if direction == 0:
            continue

        entry_bar = bars[idx + 1]
        exit_index = min(idx + 1 + max(1, int(hold_bars)), len(bars) - 1)
        exit_bar = bars[exit_index]
        order = SimOrder(
            index=len(orders),
            direction=direction,
            size=float(position_size),
            signal_probability=prob,
            signal_confidence=conf,
            created_bar_index=idx,
        )
        orders.append(order)

        vol_scale = float(volatility_values[idx]) if volatility_values is not None else 1.0
        fee_fraction = float(fee_model.cost_fraction(probability=prob, confidence=conf, direction=direction))
        entry_slip_fraction = float(slippage_model.cost_fraction(probability=prob, confidence=conf, direction=direction, volatility_scale=vol_scale))
        exit_slip_fraction = float(slippage_model.cost_fraction(probability=prob, confidence=conf, direction=-direction, volatility_scale=vol_scale))

        entry_price = float(entry_bar.open) * (1.0 + entry_slip_fraction * direction)
        exit_price = float(exit_bar.close) * (1.0 - exit_slip_fraction * direction)
        gross_return = direction * ((exit_price - entry_price) / max(1e-12, entry_price))
        gross_unit = float(np.sign(gross_return))
        cost_penalty = fee_fraction + entry_slip_fraction + exit_slip_fraction
        net_unit = gross_unit - cost_penalty

        gross_pnl[idx] = gross_unit
        net_pnl[idx] = net_unit

        fills.append(
            FillEvent(
                order_index=order.index,
                bar_index=entry_bar.index,
                direction=direction,
                size=float(position_size),
                price=entry_price,
                fee_fraction=fee_fraction,
                slippage_fraction=entry_slip_fraction,
                timestamp=entry_bar.timestamp,
            )
        )
        fills.append(
            FillEvent(
                order_index=order.index,
                bar_index=exit_bar.index,
                direction=-direction,
                size=float(position_size),
                price=exit_price,
                fee_fraction=0.0,
                slippage_fraction=exit_slip_fraction,
                timestamp=exit_bar.timestamp,
            )
        )
        trades.append(
            TradeRecord(
                index=int(idx),
                direction=int(direction),
                probability=prob,
                confidence=conf,
                realized_direction=int(1 if targets[idx] >= 0.5 else -1),
                gross_unit_pnl=float(gross_unit),
                net_unit_pnl=float(net_unit),
                fee_penalty=float(fee_fraction),
                slippage_penalty=float(entry_slip_fraction + exit_slip_fraction),
                hold_probability=float(hold_values[idx]) if hold_values is not None else None,
                confidence_probability=float(explicit_conf[idx]) if explicit_conf is not None else None,
                gate_score=float(gate_values[idx]) if gate_values is not None else None,
            )
        )

    cumulative = np.cumsum(net_pnl)
    peak = np.maximum.accumulate(cumulative) if cumulative.size else np.empty(0, dtype=np.float32)
    drawdown = peak - cumulative if cumulative.size else np.empty(0, dtype=np.float32)
    signals = np.asarray([trade.direction for trade in trades], dtype=np.int8) if trades else np.empty(0, dtype=np.int8)
    wins = np.asarray([trade.net_unit_pnl > 0 for trade in trades], dtype=bool) if trades else np.empty(0, dtype=bool)
    losses = np.asarray([trade.net_unit_pnl < 0 for trade in trades], dtype=bool) if trades else np.empty(0, dtype=bool)
    long_trades = signals == 1
    short_trades = signals == -1
    trade_count = int(len(trades))
    hold_count = int(usable - trade_count)

    summary = BacktestSummary(
        trade_count=trade_count,
        hold_count=hold_count,
        participation_rate=round(float(trade_count / max(1, usable)), 6),
        win_rate=round(float(wins.sum() / max(1, trade_count)), 6),
        loss_rate=round(float(losses.sum() / max(1, trade_count)), 6),
        long_win_rate=round(float((wins & long_trades).sum() / max(1, long_trades.sum())), 6),
        short_win_rate=round(float((wins & short_trades).sum() / max(1, short_trades.sum())), 6),
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
    payload["execution_mode"] = "event_driven"
    payload["hold_bars"] = int(hold_bars)
    payload["position_size"] = float(position_size)
    payload["orders"] = [order.to_dict() for order in orders]
    payload["fills"] = [fill.to_dict() for fill in fills]
    payload["trades"] = [trade.to_dict() for trade in trades]
    return payload
