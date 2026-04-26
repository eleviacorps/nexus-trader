from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class MarketBar:
    index: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SimOrder:
    index: int
    direction: int
    size: float
    signal_probability: float
    signal_confidence: float
    created_bar_index: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FillEvent:
    order_index: int
    bar_index: int
    direction: int
    size: float
    price: float
    fee_fraction: float
    slippage_fraction: float
    timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
