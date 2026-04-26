from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


PERSONA_LOSS_WEIGHTS: dict[str, float] = {
    "retail": 2.20,
    "institutional": 1.30,
    "algo": 1.50,
    "whale": 1.10,
    "noise": 2.50,
}

GAIN_WEIGHT = 1.0
MEMORY_DECAY = 0.92


@dataclass
class AsymmetricMemory:
    persona_name: str
    loss_weight: float = 1.8
    gain_weight: float = GAIN_WEIGHT
    decay: float = MEMORY_DECAY
    _memory: float = field(default=0.0, init=False)

    def update(self, return_pct: float) -> None:
        signed_return = float(return_pct)
        impact = abs(signed_return) * self.loss_weight if signed_return < 0.0 else signed_return * self.gain_weight
        self._memory = float(self._memory * self.decay + impact)

    @property
    def fear_index(self) -> float:
        return float(np.tanh(self._memory * 2.0))

    def strategy_weight_modifier(self) -> dict[str, float]:
        fear = self.fear_index
        return {
            "trend": max(0.3, 1.0 - fear * 0.6),
            "mean_rev": min(2.0, 1.0 + fear * 0.8),
            "ict": min(1.8, 1.0 + fear * 0.5) if self.persona_name == "institutional" else 1.0,
            "momentum": max(0.4, 1.0 - fear * 0.4),
            "smc": 1.0,
        }


def build_acm_memories(ohlcv_sequence: Iterable[Mapping[str, float]]) -> dict[str, AsymmetricMemory]:
    rows = list(ohlcv_sequence)
    memories = {
        name: AsymmetricMemory(persona_name=name, loss_weight=PERSONA_LOSS_WEIGHTS.get(name, 1.8))
        for name in PERSONA_LOSS_WEIGHTS
    }
    for idx in range(1, len(rows)):
        prev_close = float(rows[idx - 1].get("close", 0.0) or 0.0)
        curr_close = float(rows[idx].get("close", prev_close) or prev_close)
        if prev_close == 0.0:
            continue
        return_pct = (curr_close - prev_close) / prev_close
        for memory in memories.values():
            memory.update(return_pct)
    return memories


def fear_indices_from_closes(closes: pd.Series | np.ndarray | list[float]) -> pd.DataFrame:
    close_series = pd.Series(closes, dtype=np.float64).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    returns = close_series.pct_change().fillna(0.0)
    output = pd.DataFrame(index=close_series.index)
    for persona, loss_weight in PERSONA_LOSS_WEIGHTS.items():
        memory = 0.0
        fears: list[float] = []
        for value in returns.tolist():
            impact = abs(float(value)) * loss_weight if float(value) < 0.0 else float(value) * GAIN_WEIGHT
            memory = memory * MEMORY_DECAY + impact
            fears.append(float(np.tanh(memory * 2.0)))
        output[f"fear_index_{persona}"] = fears
    return output.astype(np.float32)
