from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class NexusTradeEnv:
    equity: float = 1000.0
    peak_equity: float = 1000.0
    max_lot: float = 0.20
    max_risk_pct: float = 0.02
    max_drawdown_limit: float = 0.15
    min_confidence: float = 0.55
    pnl_history: list[float] = field(default_factory=list)

    def _current_drawdown(self) -> float:
        if self.peak_equity <= 0.0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity)

    def _rolling_vol(self) -> float:
        if len(self.pnl_history) < 2:
            return 1.0
        return float(np.std(np.asarray(self.pnl_history[-20:], dtype=np.float64), ddof=0) + 1e-6)

    def _conformal_confidence(self, confidence: float) -> float:
        return float(confidence)

    def _risk_adjusted_reward(self, pnl: float, lot_size: float) -> float:
        if lot_size <= 0.0:
            return 0.0
        sharpe_reward = float(pnl) / self._rolling_vol()
        drawdown_penalty = -3.0 * max(0.0, self._current_drawdown() - 0.05)
        return float(sharpe_reward + drawdown_penalty)

    def step(
        self,
        *,
        action: tuple[str, float],
        branch_rewards: list[float],
        confidence: float,
        realized_pnl: float,
    ) -> tuple[dict[str, Any], float]:
        direction, requested_lot = action
        lot = float(np.clip(requested_lot, 0.0, self.max_lot))
        if self._current_drawdown() > self.max_drawdown_limit or self._conformal_confidence(confidence) < self.min_confidence:
            direction, lot = "HOLD", 0.0
            realized_pnl = 0.0
        self.equity += float(realized_pnl)
        self.peak_equity = max(self.peak_equity, self.equity)
        self.pnl_history.append(float(realized_pnl))
        expected_reward = float(np.mean(branch_rewards)) if branch_rewards else 0.0
        reward = self._risk_adjusted_reward(expected_reward + float(realized_pnl), lot)
        return {
            "direction": direction,
            "lot": round(lot, 4),
            "equity": round(self.equity, 6),
            "drawdown": round(self._current_drawdown(), 6),
        }, reward
