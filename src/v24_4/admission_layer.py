"""
V24.4 Adaptive Admission Layer
Controls trade admission based on multi-factor scoring system.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from enum import Enum


class MarketRegime(Enum):
    """Market regime types."""
    TREND = "trend"
    BREAKOUT = "breakout"
    LIQUIDITY_SWEEP = "liquidity_sweep"
    MEAN_REVERSION = "mean_reversion"


class AdaptiveAdmissionLayer:
    """Adaptive admission layer for V24.4 tactical trading system."""

    def __init__(self):
        self.regime_thresholds = {
            MarketRegime.TREND: 0.62,
            MarketRegime.BREAKOUT: 0.68,
            MarketRegime.LIQUIDITY_SWEEP: 0.65,
            MarketRegime.MEAN_REVERSION: 0.75
        }
        self.participation_history = []
        self.admission_scores = []

    def calculate_admission_score(self, tactical_trade_probability: float,
                                tactical_tradeability: float,
                                execution_quality: float,
                                strategic_confidence: float,
                                regime_specialist_confidence: float) -> float:
        """
        Calculate multi-factor admission score for a trade.

        Args:
            tactical_trade_probability: Probability from tactical analysis (0.0-1.0)
            tactical_tradeability: Tradeability score (0.0-1.0)
            execution_quality: Execution quality score (0.0-1.0)
            strategic_confidence: Confidence from strategic engine (0.0-1.0)
            regime_specialist_confidence: Confidence from regime specialist (0.0-1.0)

        Returns:
            float: Admission score (0.0-1.0)
        """
        # Weighted admission score calculation
        admission_score = (
            0.30 * tactical_trade_probability +
            0.25 * tactical_tradeability +
            0.20 * execution_quality +
            0.15 * strategic_confidence +
            0.10 * regime_specialist_confidence
        )

        return min(1.0, max(0.0, admission_score))

    def should_admit_trade(self, regime: MarketRegime, admission_score: float) -> bool:
        """
        Determine if a trade should be admitted based on admission score and regime thresholds.

        Args:
            regime: Current market regime
            admission_score: Calculated admission score

        Returns:
            bool: Whether trade should be admitted
        """
        threshold = self.regime_thresholds.get(regime, 0.7)
        return admission_score >= threshold

    def get_adaptive_threshold(self, regime: MarketRegime) -> float:
        """
        Get adaptive threshold for specific regime.

        Args:
            regime: Market regime

        Returns:
            float: Adaptive threshold value
        """
        return self.regime_thresholds.get(regime, 0.7)

    def update_regime_threshold(self, regime: MarketRegime, adjustment: float):
        """
        Update regime threshold based on market conditions.

        Args:
            regime: Market regime
            adjustment: Threshold adjustment value
        """
        if regime in self.regime_thresholds:
            current_threshold = self.regime_thresholds[regime]
            self.regime_thresholds[regime] = max(0.60, min(0.80, current_threshold + adjustment))

    def get_participation_rate(self) -> float:
        """
        Calculate current participation rate.

        Returns:
            float: Current participation rate
        """
        if not self.participation_history:
            return 0.0

        admitted_trades = sum(1 for admitted in self.participation_history if admitted)
        return admitted_trades / len(self.participation_history) if self.participation_history else 0.0


def main():
    """Example usage of the adaptive admission layer."""
    admission_layer = AdaptiveAdmissionLayer()

    # Example admission score calculation
    score = admission_layer.calculate_admission_score(
        tactical_trade_probability=0.85,
        tactical_tradeability=0.90,
        execution_quality=0.75,
        strategic_confidence=0.80,
        regime_specialist_confidence=0.70
    )

    print(f"Admission score: {score}")

    # Example regime-based threshold
    regime = MarketRegime.TREND
    threshold = admission_layer.get_adaptive_threshold(regime)
    print(f"Threshold for {regime.value}: {threshold}")

    # Example admission decision
    should_admit = admission_layer.should_admit_trade(regime, score)
    print(f"Trade admission decision: {should_admit}")


if __name__ == "__main__":
    main()