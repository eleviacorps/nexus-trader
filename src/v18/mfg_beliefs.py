from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    from scipy.linalg import expm  # type: ignore
except Exception:  # pragma: no cover
    def expm(matrix: np.ndarray) -> np.ndarray:
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        inv = np.linalg.inv(eigenvectors)
        diag = np.diag(np.exp(eigenvalues))
        return np.real_if_close(eigenvectors @ diag @ inv).astype(np.float64)


DRIFT_STATES = np.array([0.0002, 0.0, -0.0002], dtype=np.float64)
GENERATOR = np.array(
    [
        [-0.033, 0.017, 0.017],
        [0.017, -0.033, 0.017],
        [0.017, 0.017, -0.033],
    ],
    dtype=np.float64,
)
PERSONA_PRIORS: dict[str, np.ndarray] = {
    "retail": np.array([0.33, 0.34, 0.33], dtype=np.float64),
    "institutional": np.array([0.25, 0.50, 0.25], dtype=np.float64),
    "algo": np.array([0.40, 0.30, 0.30], dtype=np.float64),
    "whale": np.array([0.20, 0.30, 0.50], dtype=np.float64),
    "noise": np.array([0.50, 0.25, 0.25], dtype=np.float64),
}
SIGMA = 0.003


def _normalized(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.float64), 1e-12, None)
    total = float(clipped.sum())
    return clipped / total if total > 0 else np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)


class MFGPersonaEquilibrium:
    def coefficients(self, disagreement: float, entropy: float) -> tuple[float, float]:
        g1_t = float(np.clip(0.65 + (disagreement * 400.0), 0.35, 1.60))
        g2_t = float(np.clip(-(0.30 + (entropy * 0.18)), -1.20, -0.10))
        return g1_t, g2_t

    def optimal_trading_rate(self, alpha_k: float, inventory_k: float, g1_t: float, g2_t: float) -> float:
        return float((g1_t * alpha_k) + (g2_t * inventory_k))


@dataclass
class PersonaBelief:
    name: str
    belief: np.ndarray = field(default_factory=lambda: np.array([0.33, 0.34, 0.33], dtype=np.float64))
    inventory: float = 0.0
    risk_aversion: float = 0.25
    impact: float = 1.0

    def update(self, observed_return: float, sigma: float = SIGMA) -> None:
        likelihoods = np.exp(-0.5 * ((float(observed_return) - DRIFT_STATES) / max(float(sigma), 1e-9)) ** 2)
        posterior = _normalized(self.belief * likelihoods)
        transition = expm(GENERATOR * (5.0 / (252.0 * 8.0 * 60.0)))
        self.belief = _normalized(np.real_if_close(posterior @ transition).astype(np.float64))
        self.inventory = float(np.clip((0.92 * self.inventory) + (observed_return * 220.0), -1.5, 1.5))

    @property
    def expected_drift(self) -> float:
        return float(self.belief @ DRIFT_STATES)

    @property
    def entropy(self) -> float:
        p = np.clip(self.belief, 1e-12, 1.0)
        return float(-np.sum(p * np.log(p)))

    @property
    def dominant_state(self) -> str:
        index = int(np.argmax(self.belief))
        return ["bullish_drift", "neutral_drift", "bearish_drift"][index]

    def summary(self, equilibrium: MFGPersonaEquilibrium | None = None, disagreement: float = 0.0) -> dict[str, Any]:
        eq = equilibrium or MFGPersonaEquilibrium()
        g1_t, g2_t = eq.coefficients(disagreement=disagreement, entropy=self.entropy)
        rate = eq.optimal_trading_rate(self.expected_drift, self.inventory, g1_t, g2_t)
        return {
            "belief": [round(float(value), 6) for value in self.belief.tolist()],
            "expected_drift": round(float(self.expected_drift), 8),
            "entropy": round(float(self.entropy), 6),
            "dominant_state": self.dominant_state,
            "inventory": round(float(self.inventory), 6),
            "equilibrium_rate": round(float(rate), 8),
            "risk_aversion": round(float(self.risk_aversion), 4),
            "impact": round(float(self.impact), 4),
        }


class MFGBeliefState:
    def __init__(self) -> None:
        self.personas = {
            name: PersonaBelief(name=name, belief=prior.copy())
            for name, prior in PERSONA_PRIORS.items()
        }
        self.equilibrium = MFGPersonaEquilibrium()

    def update(self, observed_return: float) -> None:
        for persona in self.personas.values():
            persona.update(observed_return)

    def update_from_bars(self, bars: list[dict[str, Any]]) -> None:
        for index in range(1, len(bars)):
            prev = float((bars[index - 1] or {}).get("close", 0.0) or 0.0)
            curr = float((bars[index] or {}).get("close", 0.0) or 0.0)
            if prev <= 0.0 or curr <= 0.0:
                continue
            self.update((curr - prev) / prev)

    @property
    def disagreement(self) -> float:
        drifts = np.asarray([persona.expected_drift for persona in self.personas.values()], dtype=np.float64)
        return float(np.std(drifts))

    @property
    def consensus_drift(self) -> float:
        weights = {"retail": 0.30, "institutional": 0.35, "algo": 0.20, "whale": 0.10, "noise": 0.05}
        return float(sum(weights[name] * persona.expected_drift for name, persona in self.personas.items()))

    def summary(self) -> dict[str, Any]:
        disagreement = self.disagreement
        personas = {
            name: persona.summary(self.equilibrium, disagreement=disagreement)
            for name, persona in self.personas.items()
        }
        rates = [float(item["equilibrium_rate"]) for item in personas.values()]
        return {
            "disagreement": round(float(disagreement), 8),
            "consensus_drift": round(float(self.consensus_drift), 8),
            "mean_equilibrium_rate": round(float(np.mean(rates) if rates else 0.0), 8),
            "personas": personas,
        }
