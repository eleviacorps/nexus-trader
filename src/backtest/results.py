from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TradeRecord:
    index: int
    direction: int
    probability: float
    confidence: float
    realized_direction: int
    gross_unit_pnl: float
    net_unit_pnl: float
    fee_penalty: float
    slippage_penalty: float
    hold_probability: float | None = None
    confidence_probability: float | None = None
    gate_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BacktestSummary:
    trade_count: int
    hold_count: int
    participation_rate: float
    win_rate: float
    loss_rate: float
    long_win_rate: float
    short_win_rate: float
    avg_unit_pnl: float
    gross_avg_unit_pnl: float
    cumulative_unit_pnl: float
    gross_cumulative_unit_pnl: float
    max_drawdown_units: float
    decision_threshold: float
    confidence_floor: float
    gate_threshold: float
    hold_threshold: float
    capital_backtests: dict[str, Any]
    fee_model: str
    slippage_model: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
