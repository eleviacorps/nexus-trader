from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping

import numpy as np

from src.simulation.strategies import StrategySignal, strategy_map


@dataclass
class Persona:
    name: str
    capital_weight: float
    noise_level: float
    strategy_weights: Dict[str, float]
    crowd_pct: float
    description: str = ""

    def decide(self, row: Mapping[str, float], rng: random.Random) -> StrategySignal:
        weighted_signal = 0.0
        weighted_confidence = 0.0
        total_weight = 0.0

        for strategy_name, weight in self.strategy_weights.items():
            fn = strategy_map().get(strategy_name)
            if fn is None:
                continue
            result = fn(row)
            weighted_signal += result.direction * result.confidence * weight
            weighted_confidence += result.confidence * weight
            total_weight += weight

        if total_weight <= 0:
            return StrategySignal(direction=0, confidence=0.0)

        weighted_signal /= total_weight
        weighted_confidence /= total_weight
        final_signal = weighted_signal + self._contextual_bias(row) + rng.gauss(0.0, self.noise_level)
        weighted_confidence = min(0.99, weighted_confidence + self._contextual_confidence_boost(row))

        if final_signal > 0.15:
            return StrategySignal(direction=1, confidence=min(0.99, weighted_confidence))
        if final_signal < -0.15:
            return StrategySignal(direction=-1, confidence=min(0.99, weighted_confidence))
        return StrategySignal(direction=0, confidence=0.0)

    def _contextual_bias(self, row: Mapping[str, float]) -> float:
        macro_bias = float(row.get("macro_bias", 0.0))
        news_bias = float(row.get("news_bias", 0.0))
        crowd_bias = float(row.get("crowd_bias", 0.0))
        crowd_extreme = float(row.get("crowd_extreme", 0.0))
        displacement = float(row.get("displacement", 0.0))

        if self.name == "retail":
            return (0.15 * news_bias) + (0.34 * crowd_bias) + (0.18 * crowd_extreme * (1.0 if displacement >= 0 else -1.0))
        if self.name == "institutional":
            return (0.40 * macro_bias) + (0.12 * news_bias) - (0.10 * crowd_bias * crowd_extreme)
        if self.name == "algo":
            return (0.08 * macro_bias) + (0.05 * news_bias) + (0.06 * np.tanh(displacement))
        if self.name == "whale":
            return (0.30 * macro_bias) - (0.28 * crowd_bias * max(0.3, crowd_extreme)) + (0.10 * news_bias)
        if self.name == "noise":
            return (0.12 * news_bias) + (0.20 * crowd_bias) + rng_gauss_like(crowd_extreme)
        return 0.0

    def _contextual_confidence_boost(self, row: Mapping[str, float]) -> float:
        macro_shock = abs(float(row.get("macro_shock", 0.0)))
        news_intensity = abs(float(row.get("news_intensity", 0.0)))
        crowd_extreme = abs(float(row.get("crowd_extreme", 0.0)))
        if self.name == "institutional":
            return 0.08 * macro_shock + 0.04 * news_intensity
        if self.name == "whale":
            return 0.06 * macro_shock + 0.06 * crowd_extreme
        if self.name == "retail":
            return 0.03 * news_intensity + 0.06 * crowd_extreme
        if self.name == "algo":
            return 0.03 * macro_shock
        if self.name == "noise":
            return 0.02 * crowd_extreme
        return 0.0


def rng_gauss_like(scale: float) -> float:
    return random.uniform(-1.0, 1.0) * min(max(scale, 0.0), 1.0) * 0.08


def default_personas() -> Dict[str, Persona]:
    return {
        "retail": Persona(
            name="retail",
            capital_weight=0.25,
            noise_level=0.35,
            crowd_pct=0.60,
            strategy_weights={"trend": 0.65, "momentum": 0.15, "mean_rev": 0.20},
            description="Emotional trend chaser, news-reactive",
        ),
        "institutional": Persona(
            name="institutional",
            capital_weight=0.35,
            noise_level=0.05,
            crowd_pct=0.15,
            strategy_weights={"ict": 0.40, "smc": 0.30, "mean_rev": 0.30},
            description="Smart money, liquidity-aware macro participant",
        ),
        "algo": Persona(
            name="algo",
            capital_weight=0.20,
            noise_level=0.02,
            crowd_pct=0.20,
            strategy_weights={"momentum": 0.35, "trend": 0.40, "ict": 0.25},
            description="Systematic price-structure trader",
        ),
        "whale": Persona(
            name="whale",
            capital_weight=0.15,
            noise_level=0.01,
            crowd_pct=0.02,
            strategy_weights={"ict": 0.45, "smc": 0.35, "mean_rev": 0.20},
            description="Contrarian liquidity accumulator",
        ),
        "noise": Persona(
            name="noise",
            capital_weight=0.05,
            noise_level=0.90,
            crowd_pct=0.03,
            strategy_weights={"trend": 0.60, "momentum": 0.40},
            description="Unstable crowd noise persona",
        ),
    }


def save_personas(path: Path, personas: Mapping[str, Persona]) -> None:
    payload = {name: asdict(persona) for name, persona in personas.items()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_personas(path: Path) -> Dict[str, Persona]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {name: Persona(**config) for name, config in payload.items()}
