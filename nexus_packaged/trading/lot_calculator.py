"""Lot sizing logic."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any


class LotCalculator:
    """Calculates lot size under different policies."""

    @staticmethod
    def _round_to_step(value: float, step: float) -> float:
        if step <= 0:
            return value
        return round(round(value / step) * step, 8)

    def calculate(
        self,
        mode: str,
        config: Any,
        account_equity: float,
        sl_pips: float,
        pip_value: float,
        confidence: float,
        win_rate_history: float,
        broker_lot_min: float = 0.01,
        broker_lot_max: float = 500.0,
        broker_lot_step: float = 0.01,
        risk_reward: float = 2.0,
    ) -> float:
        """Return lot size using fixed/range/risk_pct/kelly mode."""
        mode_n = str(mode or "fixed").lower()
        lot: float

        if mode_n == "fixed":
            lot = float(getattr(config, "fixed_lot_size", 0.01))
        elif mode_n == "range":
            lot_min = float(getattr(config, "lot_min", broker_lot_min))
            lot_max = float(getattr(config, "lot_max", broker_lot_max))
            range_mode = str(getattr(config, "lot_range_mode", "confidence")).lower()
            if range_mode == "random":
                lot = random.uniform(lot_min, lot_max)
            else:
                lot = lot_min + (lot_max - lot_min) * float(max(0.0, min(1.0, confidence)))
        elif mode_n == "risk_pct":
            risk_pct = float(getattr(config, "risk_pct_per_trade", 1.0))
            monetary_risk = float(account_equity) * (risk_pct / 100.0)
            lot = monetary_risk / max(1e-12, float(sl_pips) * float(pip_value))
        elif mode_n == "kelly":
            rr = float(risk_reward)
            win_rate = float(max(0.0, min(1.0, win_rate_history)))
            kelly_full = win_rate - ((1.0 - win_rate) / max(1e-12, rr))
            kelly_fraction = float(getattr(config, "kelly_fraction", 0.25))
            kelly_fractional = kelly_full * kelly_fraction
            base_lot = float(getattr(config, "fixed_lot_size", 0.01))
            lot = base_lot * kelly_fractional
            lot = max(float(getattr(config, "lot_min", broker_lot_min)), lot)
        else:
            lot = float(getattr(config, "fixed_lot_size", 0.01))

        lower = max(float(getattr(config, "lot_min", broker_lot_min)), broker_lot_min)
        upper = min(float(getattr(config, "lot_max", broker_lot_max)), broker_lot_max)
        lot = min(max(lot, lower), upper)
        lot = self._round_to_step(lot, broker_lot_step)
        lot = min(max(lot, lower), upper)
        return float(lot)

