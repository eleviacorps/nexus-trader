from __future__ import annotations

from dataclasses import dataclass


class FeeModel:
    def cost_fraction(self, *, probability: float, confidence: float, direction: int) -> float:
        return 0.0


@dataclass(frozen=True)
class ZeroFeeModel(FeeModel):
    def cost_fraction(self, *, probability: float, confidence: float, direction: int) -> float:
        return 0.0


@dataclass(frozen=True)
class FixedBpsFeeModel(FeeModel):
    entry_bps: float = 0.0
    exit_bps: float = 0.0

    def cost_fraction(self, *, probability: float, confidence: float, direction: int) -> float:
        total_bps = max(0.0, float(self.entry_bps)) + max(0.0, float(self.exit_bps))
        return total_bps / 10000.0
