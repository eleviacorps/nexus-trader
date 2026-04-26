"""V27 Short Horizon 15-Minute Trade Predictor.

Uses existing Phase 1 generator to predict next 15 minutes of trade opportunities.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch import Tensor


@dataclass
class PredictionResult:
    """Result of a 15-minute trade prediction."""

    decision: str  # BUY, SELL, or HOLD
    confidence: float
    expected_duration_min: float
    expected_return: float
    stop_loss: float
    take_profit: float
    entry_price: float
    scenario_breakdown: dict
    expiry_timestamp: float
    generated_at: float
    valid_horizon_min: int = 15

    def is_expired(self) -> bool:
        """Check if prediction has expired."""
        return time.time() > self.expiry_timestamp

    def is_valid(self) -> bool:
        """Check if prediction is still valid."""
        return not self.is_expired() and self.decision != "HOLD"

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "confidence": float(self.confidence),
            "expected_duration_min": float(self.expected_duration_min),
            "expected_return": float(self.expected_return),
            "stop_loss": float(self.stop_loss),
            "take_profit": float(self.take_profit),
            "entry_price": float(self.entry_price),
            "scenario_breakdown": self.scenario_breakdown,
            "expiry_timestamp": self.expiry_timestamp,
            "generated_at": self.generated_at,
            "valid_horizon_min": self.valid_horizon_min,
            "is_expired": self.is_expired(),
        }


class ShortHorizonPredictor:
    """Predictor for 15-minute trade horizons.

    Generates high-quality trade predictions valid for exactly 15 minutes.
    Uses scenario clustering and confidence scoring.
    """

    def __init__(
        self,
        base_generator,
        num_futures: int = 64,
        num_clusters: int = 5,
        confidence_threshold: float = 0.60,
        validity_minutes: int = 15,
        device: str = "cpu",
    ):
        self.base_generator = base_generator
        self.num_futures = min(num_futures, 16)  # Cap at 16 for speed
        self.num_clusters = num_clusters
        self.confidence_threshold = confidence_threshold
        self.validity_minutes = validity_minutes
        self.device = torch.device(device)

    @torch.no_grad()
    def predict_15min_trade(
        self,
        past_context: Tensor,
        regime_probs: Tensor,
        current_price: float = 0.0,
        steps: int = 20,
    ) -> PredictionResult:
        """Generate a 15-minute trade prediction.

        Args:
            past_context: (T, C) or (1, T, C) historical context
            regime_probs: (9,) or (1, 9) regime probabilities
            current_price: Current price for entry calculation
            steps: Diffusion steps

        Returns:
            PredictionResult with trade decision
        """
        if past_context is not None:
            past_context = past_context.to(self.device)
        regime_probs = regime_probs.to(self.device)

        # Handle batch vs single
        single = regime_probs.dim() == 1
        if single:
            regime_probs = regime_probs.unsqueeze(0)
        if past_context is not None and past_context.dim() == 2:
            past_context = past_context.unsqueeze(0)

        now = time.time()
        expiry = now + self.validity_minutes * 60

        # Generate futures
        futures = []
        for i in range(self.num_futures):
            rp = regime_probs[0:1]
            ctx = past_context[0:1] if past_context is not None else None

            paths = self.base_generator.generate_paths(
                world_state=None,
                regime_probs=rp,
                num_paths=1,
                past_context=ctx,
                steps=steps,
            )

            if paths:
                data = paths[0]["data"]
                if isinstance(data, list):
                    data = data[0]
                future = np.array(data)
                futures.append(future)

        if not futures:
            return PredictionResult(
                decision="HOLD",
                confidence=0.0,
                expected_duration_min=0.0,
                expected_return=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                entry_price=current_price,
                scenario_breakdown={},
                expiry_timestamp=expiry,
                generated_at=now,
            )

        futures = np.array(futures)

        # Cluster futures into scenarios
        scenarios = self._cluster_scenarios(futures)

        # Determine best scenario
        best_name = max(scenarios.keys(), key=lambda k: scenarios[k]["count"])
        best_count = scenarios[best_name]["count"]
        best_return = scenarios[best_name]["avg_return"]

        confidence = best_count / self.num_futures

        # Decision
        if confidence < self.confidence_threshold:
            decision = "HOLD"
            tp = sl = entry = expected_return = 0.0
            duration = 0.0
        elif best_return > 0.001:
            decision = "BUY"
            entry = current_price if current_price else futures[0].mean()
            tp = entry * (1 + abs(best_return) * 3)
            sl = entry * (1 - abs(best_return))
            expected_return = best_return
            duration = min(15, max(1, int(abs(best_return) * 100)))
        elif best_return < -0.001:
            decision = "SELL"
            entry = current_price if current_price else futures[0].mean()
            tp = entry * (1 - abs(best_return) * 3)
            sl = entry * (1 + abs(best_return))
            expected_return = abs(best_return)
            duration = min(15, max(1, int(abs(best_return) * 100)))
        else:
            decision = "HOLD"
            tp = sl = entry = expected_return = 0.0
            duration = 0.0

        return PredictionResult(
            decision=decision,
            confidence=confidence,
            expected_duration_min=duration,
            expected_return=expected_return,
            stop_loss=sl,
            take_profit=tp,
            entry_price=entry,
            scenario_breakdown={
                name: {"count": scenarios[name]["count"], "avg_return": scenarios[name]["avg_return"]}
                for name in scenarios
            },
            expiry_timestamp=expiry,
            generated_at=now,
        )

    def _cluster_scenarios(self, futures: np.ndarray) -> dict:
        """Cluster futures into directional scenarios."""
        # Compute returns
        returns = np.diff(futures, axis=1)
        total_returns = returns.sum(axis=1)

        scenarios = {
            "bullish": [],
            "bearish": [],
            "sideways": [],
            "breakout_up": [],
            "breakout_down": [],
        }

        for i, ret in enumerate(total_returns):
            if ret > 0.005:
                scenarios["bullish"].append(i)
            elif ret < -0.005:
                scenarios["bearish"].append(i)
            elif abs(ret) < 0.001:
                scenarios["sideways"].append(i)
            elif ret > 0.001:
                scenarios["breakout_up"].append(i)
            else:
                scenarios["breakout_down"].append(i)

        # Return as dict with avg_return precomputed
        result = {}
        for name, indices in scenarios.items():
            if indices:
                scenario_returns = futures[indices].mean(axis=0)
                avg_return = float(np.diff(scenario_returns).mean())
                result[name] = {"indices": indices, "avg_return": avg_return, "count": len(indices)}
            else:
                result[name] = {"indices": [], "avg_return": 0.0, "count": 0}

        return result


def create_short_horizon_predictor(base_generator, device="cpu") -> ShortHorizonPredictor:
    """Create V27 short horizon predictor."""
    return ShortHorizonPredictor(
        base_generator=base_generator,
        num_futures=64,
        num_clusters=5,
        confidence_threshold=0.60,
        validity_minutes=15,
        device=device,
    )