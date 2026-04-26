"""Trading subsystem for packaged runtime."""

from __future__ import annotations

from nexus_packaged.trading.auto_trader import AutoTradeConfig, AutoTrader
from nexus_packaged.trading.lot_calculator import LotCalculator
from nexus_packaged.trading.manual_trader import ManualOrderRequest, ManualTrader
from nexus_packaged.trading.trade_manager import SessionSummary, TradeManager

__all__ = [
    "AutoTradeConfig",
    "AutoTrader",
    "LotCalculator",
    "ManualOrderRequest",
    "ManualTrader",
    "SessionSummary",
    "TradeManager",
]

