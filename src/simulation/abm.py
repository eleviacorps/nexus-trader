from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Mapping

from src.simulation.personas import Persona


@dataclass
class PersonaDecision:
    persona: str
    n_agents: int
    signal: int
    confidence: float
    impact: float


@dataclass
class SyntheticMarketState:
    open: float
    high: float
    low: float
    close: float
    volume: int
    net_pressure: float
    buy_pressure: float
    sell_pressure: float
    directional_bias: float
    decisions: List[PersonaDecision] = field(default_factory=list)


def simulate_one_step(
    current_row: Mapping[str, float],
    personas: Mapping[str, Persona],
    n_agents: int = 200,
    seed: int | None = None,
) -> SyntheticMarketState:
    rng = random.Random(seed)
    current_price = float(current_row.get("close", 0.0))
    atr = float(current_row.get("atr_14", max(current_price * 0.001, 1e-6)))

    buy_pressure = 0.0
    sell_pressure = 0.0
    total_volume = 0.0
    decisions: List[PersonaDecision] = []

    for persona_name, persona in personas.items():
        count = max(1, int(n_agents * persona.crowd_pct))
        last_signal = 0
        last_confidence = 0.0
        persona_impact = 0.0
        for _ in range(count):
            result = persona.decide(current_row, rng)
            last_signal = result.direction
            last_confidence = result.confidence
            impact = persona.capital_weight * result.confidence
            persona_impact += impact
            if result.direction > 0:
                buy_pressure += impact
            elif result.direction < 0:
                sell_pressure += impact
            total_volume += persona.capital_weight * 0.5
        decisions.append(
            PersonaDecision(
                persona=persona_name,
                n_agents=count,
                signal=last_signal,
                confidence=round(last_confidence, 4),
                impact=round(persona_impact, 4),
            )
        )

    total_pressure = buy_pressure + sell_pressure
    directional_bias = (buy_pressure - sell_pressure) / total_pressure if total_pressure > 0 else 0.0
    price_move = directional_bias * atr * 0.6
    wick_noise = abs(rng.gauss(0.0, atr * 0.15))

    synthetic_open = current_price
    synthetic_close = current_price + price_move
    synthetic_high = max(synthetic_open, synthetic_close) + wick_noise
    synthetic_low = min(synthetic_open, synthetic_close) - wick_noise

    return SyntheticMarketState(
        open=round(synthetic_open, 5),
        high=round(synthetic_high, 5),
        low=round(synthetic_low, 5),
        close=round(synthetic_close, 5),
        volume=int(total_volume * 1000),
        net_pressure=round(buy_pressure - sell_pressure, 5),
        buy_pressure=round(buy_pressure, 5),
        sell_pressure=round(sell_pressure, 5),
        directional_bias=round(directional_bias, 5),
        decisions=decisions,
    )


def persona_vote_breakdown(state: SyntheticMarketState) -> Dict[str, Dict[str, float]]:
    return {
        decision.persona: {
            "n_agents": float(decision.n_agents),
            "signal": float(decision.signal),
            "confidence": float(decision.confidence),
            "impact": float(decision.impact),
        }
        for decision in state.decisions
    }
