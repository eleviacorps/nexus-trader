from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.v14.acm import AsymmetricMemory, PERSONA_LOSS_WEIGHTS


PERSONA_TESTOSTERONE_SENSITIVITY: dict[str, float] = {
    "retail": 1.80,
    "institutional": 0.60,
    "algo": 0.90,
    "whale": 0.40,
    "noise": 2.20,
}

PERSONA_FUNDAMENTAL_TRACKING_BASE: dict[str, float] = {
    "retail": 0.50,
    "institutional": 0.90,
    "algo": 0.75,
    "whale": 0.95,
    "noise": 0.20,
}


@dataclass
class WinnerLoserCycle:
    persona_name: str

    def __post_init__(self) -> None:
        self.fear_memory = AsymmetricMemory(
            persona_name=self.persona_name,
            loss_weight=PERSONA_LOSS_WEIGHTS.get(self.persona_name, 1.8),
        )
        self.t_sensitivity = PERSONA_TESTOSTERONE_SENSITIVITY.get(self.persona_name, 1.0)
        self.fundamental_base = PERSONA_FUNDAMENTAL_TRACKING_BASE.get(self.persona_name, 0.5)
        self._testosterone_level = 0.0
        self._win_streak = 0
        self._recent_returns: list[float] = []

    def update(self, return_pct: float) -> None:
        signed_return = float(return_pct)
        self.fear_memory.update(signed_return)

        if signed_return > 0.0:
            self._win_streak += 1
            boost = min(abs(signed_return) * 1.5, 0.08)
            streak_multiplier = 1.0 + 0.08 * min(self._win_streak, 10)
            delta = boost * streak_multiplier * self.t_sensitivity
            self._testosterone_level = min(1.0, self._testosterone_level * 0.95 + delta)
        else:
            crash_rate = 0.65 + (0.20 * self._testosterone_level)
            self._testosterone_level = max(0.0, self._testosterone_level * crash_rate)
            self._win_streak = 0

        self._recent_returns.append(signed_return)
        if len(self._recent_returns) > 50:
            self._recent_returns.pop(0)

    @property
    def testosterone_index(self) -> float:
        return float(np.clip(self._testosterone_level, 0.0, 1.0))

    @property
    def fear_index(self) -> float:
        return float(self.fear_memory.fear_index)

    def strategy_weight_modifier(self) -> dict[str, float]:
        testosterone = self.testosterone_index
        fear = self.fear_index
        fundamental_tracking = max(
            0.10,
            self.fundamental_base - testosterone * 0.70 * self.t_sensitivity,
        )
        return {
            "trend": max(0.20, 1.0 - fear * 0.60 + testosterone * 0.45),
            "momentum": max(0.20, 1.0 - fear * 0.40 + testosterone * 0.55),
            "mean_rev": min(2.20, 1.0 + fear * 0.80 - testosterone * 0.35),
            "ict": min(1.80, 1.0 + fear * 0.50 - testosterone * 0.10),
            "smc": 1.0,
            "fundamental_tracking": fundamental_tracking,
            "bid_aggressiveness": 1.0 + testosterone * 0.60 * self.t_sensitivity,
        }

    def summary(self) -> dict[str, float | int | str]:
        return {
            "persona": self.persona_name,
            "testosterone_index": round(self.testosterone_index, 4),
            "fear_index": round(self.fear_index, 4),
            "win_streak": int(self._win_streak),
            "fundamental_tracking": round(self.strategy_weight_modifier()["fundamental_tracking"], 4),
            "bid_aggressiveness": round(self.strategy_weight_modifier()["bid_aggressiveness"], 4),
        }


def build_wltc_states(ohlcv_bars: list[dict]) -> dict[str, WinnerLoserCycle]:
    states = {name: WinnerLoserCycle(name) for name in PERSONA_TESTOSTERONE_SENSITIVITY}
    for index in range(1, len(ohlcv_bars)):
        prev_close = float(ohlcv_bars[index - 1].get("close", 0.0) or 0.0)
        curr_close = float(ohlcv_bars[index].get("close", prev_close) or prev_close)
        if prev_close <= 0.0:
            continue
        return_pct = (curr_close - prev_close) / prev_close
        for state in states.values():
            state.update(return_pct)
    return states
