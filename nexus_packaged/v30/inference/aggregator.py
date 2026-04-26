"""V30 Aggregator - converts weighted paths to trading signals."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from typing import TypedDict


class TradingSignal(TypedDict):
    """Trading signal output from aggregator."""
    prob_up: float
    prob_down: float
    expected_return: float
    uncertainty: float
    confidence: float
    decision: str  # BUY, SELL, HOLD
    weights: list[float]


class Aggregator:
    """Converts weighted diffusion paths into trading signals."""

    def __init__(
        self,
        confidence_threshold: float = 0.55,
        min_ev_threshold: float = 0.0001,
    ):
        self.confidence_threshold = confidence_threshold
        self.min_ev_threshold = min_ev_threshold

    def aggregate(
        self,
        paths: np.ndarray,
        weights: np.ndarray,
    ) -> TradingSignal:
        """
        Convert weighted paths to trading signal.
        
        Args:
            paths: [num_paths, horizon] - diffusion paths (price values)
            weights: [num_paths] - evaluator weights
        
        Returns:
            TradingSignal with probabilities, EV, uncertainty, decision
        """
        # Compute returns for each path
        returns = (paths[:, -1] - paths[:, 0]) / (paths[:, 0] + 1e-8)
        
        # Weighted statistics
        weights_normalized = weights / (weights.sum() + 1e-8)
        
        # Probability up = sum of weights where return > 0
        prob_up = float((weights_normalized[returns > 0]).sum())
        prob_down = float((weights_normalized[returns < 0]).sum())
        
        # Expected return = weighted mean return
        expected_return = float((weights_normalized * returns).sum())
        
        # Uncertainty = weighted standard deviation
        variance = ((weights_normalized * (returns - expected_return) ** 2).sum())
        uncertainty = float(np.sqrt(variance))
        
        # IMPROVED confidence: edge / risk ratio
        # This measures: how strong is the expected return relative to uncertainty
        # Higher = more confident in the signal
        if uncertainty > 1e-8:
            confidence = abs(expected_return) / uncertainty
            confidence = float(np.clip(confidence, 0, 1))  # Scale to [0, 1]
        else:
            confidence = 0.5  # Neutral if no uncertainty
        
        # IMPROVED decision logic: use score-based approach
        # score = expected_return / uncertainty (edge/risk)
        # Positive score = BUY, Negative = SELL, Near zero = HOLD
        score = expected_return / (uncertainty + 1e-8)
        
        # Use symmetric thresholds for decision
        decision_threshold = self.confidence_threshold * 0.1  # Scale threshold appropriately
        if score > decision_threshold:
            decision = "BUY"
        elif score < -decision_threshold:
            decision = "SELL"
        else:
            decision = "HOLD"
        
        return TradingSignal(
            prob_up=prob_up,
            prob_down=prob_down,
            expected_return=expected_return,
            uncertainty=uncertainty,
            confidence=confidence,
            decision=decision,
            weights=weights.tolist(),
        )

    def aggregate_batch(
        self,
        paths_batch: np.ndarray,  # [B, num_paths, horizon]
        weights_batch: np.ndarray,  # [B, num_paths]
    ) -> list[TradingSignal]:
        """Process batch of paths."""
        signals = []
        for paths, weights in zip(paths_batch, weights_batch):
            signals.append(self.aggregate(paths, weights))
        return signals


def torch_aggregate(
    paths: Tensor,  # [B, num_paths, horizon]
    weights: Tensor,  # [B, num_paths]
    confidence_threshold: float = 0.55,
    min_ev_threshold: float = 0.0001,
) -> dict[str, Tensor]:
    """PyTorch version for batched inference.
    
    Returns dict of tensors for efficient processing.
    """
    # Compute returns
    returns = (paths[:, :, -1] - paths[:, :, 0]) / (paths[:, :, 0] + 1e-8)  # [B, num_paths]
    
    # Normalize weights
    weights_norm = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)  # [B, num_paths]
    
    # Prob up
    prob_up = (weights_norm * (returns > 0).float()).sum(dim=-1)  # [B]
    
    # Expected return
    expected_return = (weights_norm * returns).sum(dim=-1)  # [B]
    
    # Uncertainty
    variance = (weights_norm * (returns - expected_return.unsqueeze(-1)) ** 2).sum(dim=-1)
    uncertainty = torch.sqrt(variance + 1e-8)  # [B]
    
    # IMPROVED confidence: edge / risk ratio
    confidence = torch.where(
        uncertainty > 1e-8,
        (expected_return.abs() / uncertainty).clamp(0, 1),
        torch.full_like(expected_return, 0.5)
    )
    
    # IMPROVED decision: score-based (edge/risk ratio)
    # Positive score = BUY, Negative = SELL, Near zero = HOLD
    score = expected_return / (uncertainty + 1e-8)
    threshold = confidence_threshold * 0.1  # Scale threshold
    
    decision = torch.where(
        score > threshold,
        torch.ones_like(expected_return, dtype=torch.long),  # BUY = 1
        torch.where(
            score < -threshold,
            torch.zeros_like(expected_return, dtype=torch.long),  # SELL = 0
            2 * torch.ones_like(expected_return, dtype=torch.long)  # HOLD = 2
        )
    )
    
    return {
        "prob_up": prob_up,
        "expected_return": expected_return,
        "uncertainty": uncertainty,
        "confidence": confidence,
        "decision": decision,
    }