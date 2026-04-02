from __future__ import annotations

from dataclasses import dataclass


class SlippageModel:
    def cost_fraction(
        self,
        *,
        probability: float,
        confidence: float,
        direction: int,
        volatility_scale: float = 1.0,
    ) -> float:
        return 0.0


@dataclass(frozen=True)
class NoSlippageModel(SlippageModel):
    def cost_fraction(
        self,
        *,
        probability: float,
        confidence: float,
        direction: int,
        volatility_scale: float = 1.0,
    ) -> float:
        return 0.0


@dataclass(frozen=True)
class FixedBpsSlippageModel(SlippageModel):
    bps: float = 0.0

    def cost_fraction(
        self,
        *,
        probability: float,
        confidence: float,
        direction: int,
        volatility_scale: float = 1.0,
    ) -> float:
        return max(0.0, float(self.bps)) / 10000.0


@dataclass(frozen=True)
class VolatilityScaledSlippageModel(SlippageModel):
    base_bps: float = 0.0
    volatility_weight: float = 0.0
    confidence_weight: float = 0.0

    def cost_fraction(
        self,
        *,
        probability: float,
        confidence: float,
        direction: int,
        volatility_scale: float = 1.0,
    ) -> float:
        bps = max(0.0, float(self.base_bps))
        bps += max(0.0, float(self.volatility_weight)) * max(0.0, float(volatility_scale))
        bps += max(0.0, float(self.confidence_weight)) * max(0.0, float(confidence))
        return bps / 10000.0
