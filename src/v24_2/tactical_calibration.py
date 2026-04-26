"""
V24.2 Tactical Calibration Model

This module implements the tactical calibration model for V24.2.
"""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from datetime import datetime


@dataclass
class TacticalCalibrationModel:
    """Tactical calibration model for V24.2."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_trade_probability = self.config.get('min_trade_probability', 0.75)

    def calculate_tactical_trade_probability(self, inputs: Dict[str, Any]) -> float:
        """
        Calculate tactical trade probability based on inputs.

        Args:
            inputs: Dictionary of calibration inputs

        Returns:
            Tactical trade probability (0.0 to 1.0)
        """
        # Extract input features
        tactical_tradeability = inputs.get('tactical_tradeability', 0.5)
        spread = inputs.get('spread', 0.001)
        liquidity_sweep_prob = inputs.get('liquidity_sweep_probability', 0.3)
        branch_disagreement = inputs.get('branch_disagreement', 0.3)
        strategic_confidence = inputs.get('strategic_confidence', 0.7)
        recent_tactical_win_rate = inputs.get('recent_tactical_win_rate', 0.6)

        # Calculate weighted probability
        # Weight different factors based on their importance
        probability = (
            tactical_tradeability * 0.3 +
            (1.0 - spread * 1000) * 0.2 +  # Inverse of spread impact
            liquidity_sweep_prob * 0.15 +
            (1.0 - branch_disagreement) * 0.15 +  # Inverse of disagreement
            strategic_confidence * 0.1 +
            recent_tactical_win_rate * 0.1
        )

        # Ensure probability is between 0 and 1
        return max(0.0, min(1.0, probability))

    def should_trade_tactically(self, probability: float, tactical_tradeability: float) -> bool:
        """
        Determine if tactical trade should be executed.

        Args:
            probability: Calculated tactical trade probability
            tactical_tradeability: Tactical tradeability score

        Returns:
            Boolean indicating if tactical trade should be executed
        """
        return (
            probability > self.min_trade_probability and
            tactical_tradeability > 0.35
        )

    def evaluate_tactical_conditions(self, market_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate all tactical trading conditions.

        Args:
            market_state: Current market state

        Returns:
            Evaluation results
        """
        # Get tactical tradeability from tactical CABR
        tactical_cabr_results = market_state.get('tactical_cabr_results', {})
        tactical_tradeability = tactical_cabr_results.get('short_tradeability', 0.0)

        # Calculate tactical trade probability
        inputs = {
            'tactical_tradeability': tactical_tradeability,
            'spread': market_state.get('spread', 0.001),
            'liquidity_sweep_probability': market_state.get('liquidity_sweep_probability', 0.3),
            'branch_disagreement': market_state.get('branch_disagreement', 0.3),
            'strategic_confidence': market_state.get('strategic_confidence', 0.7),
            'recent_tactical_win_rate': market_state.get('recent_tactical_win_rate', 0.6)
        }

        tactical_probability = self.calculate_tactical_trade_probability(inputs)
        should_trade = self.should_trade_tactically(tactical_probability, tactical_tradeability)

        return {
            'tactical_trade_probability': tactical_probability,
            'should_trade_tactically': should_trade,
            'tactical_tradeability': tactical_tradeability,
            'evaluation_timestamp': datetime.now().isoformat()
        }


def create_tactical_calibration_model():
    """Create and test tactical calibration model."""
    print("V24.2 Tactical Calibration Model")
    print("=" * 35)

    # Initialize calibration model
    model = TacticalCalibrationModel({
        'min_trade_probability': 0.75
    })

    # Create sample market state
    sample_market_state = {
        'spread': 0.0008,
        'liquidity_sweep_probability': 0.4,
        'branch_disagreement': 0.25,
        'strategic_confidence': 0.75,
        'recent_tactical_win_rate': 0.65,
        'tactical_cabr_results': {
            'short_tradeability': 0.55
        }
    }

    # Evaluate tactical conditions
    results = model.evaluate_tactical_conditions(sample_market_state)

    print("Tactical Calibration Results:")
    print(f"  Tactical Trade Probability: {results['tactical_trade_probability']:.4f}")
    print(f"  Should Trade Tactically: {results['should_trade_tactically']}")
    print(f"  Tactical Tradeability: {results['tactical_tradeability']:.4f}")

    return results


if __name__ == "__main__":
    # Run the tactical calibration model
    create_tactical_calibration_model()