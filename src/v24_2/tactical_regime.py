"""
V24.2 Tactical Regime Detector

This module implements the tactical regime detection for V24.2.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
from datetime import datetime


@dataclass
class TacticalRegimeDetector:
    """Class to detect tactical trading regimes."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_regime_confidence = self.config.get('min_regime_confidence', 0.75)

    def detect_regime(self, market_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect the current tactical regime.

        Args:
            market_state: Current market state data

        Returns:
            Regime detection results
        """
        # Extract relevant features
        close_price = market_state.get('close', 0)
        high = market_state.get('high', 0)
        low = market_state.get('low', 0)
        open_price = market_state.get('open', 0)
        volume = market_state.get('volume', 0)
        spread = market_state.get('spread', 0)

        # Calculate regime features
        price_range = high - low
        body_range = abs(close_price - open_price)
        wick_size = price_range - body_range

        # Determine regime type based on price action
        if self._is_trend_continuation(market_state):
            regime_type = "trend_continuation"
            regime_confidence = 0.85
        elif self._is_breakout(market_state):
            regime_type = "breakout"
            regime_confidence = 0.80
        elif self._is_mean_reversion(market_state):
            regime_type = "mean_reversion"
            regime_confidence = 0.75
        elif self._is_liquidity_sweep(market_state):
            regime_type = "liquidity_sweep_reversal"
            regime_confidence = 0.70
        else:
            regime_type = "chop_no_trade"
            regime_confidence = 0.30

        # Determine if tactical trade is allowed
        allow_tactical_trade = (
            regime_confidence >= self.min_regime_confidence and
            self._is_favorable_market_conditions(market_state)
        )

        return {
            'regime_type': regime_type,
            'regime_confidence': regime_confidence,
            'allow_tactical_trade': allow_tactical_trade,
            'detection_timestamp': datetime.now().isoformat()
        }

    def _is_trend_continuation(self, market_state: Dict[str, Any]) -> bool:
        """Check if current state shows trend continuation pattern."""
        # This would implement actual trend detection logic
        # For now, return a simple heuristic
        return market_state.get('trend_strength', 0) > 0.7

    def _is_breakout(self, market_state: Dict[str, Any]) -> bool:
        """Check if current state shows breakout pattern."""
        volume = market_state.get('volume', 0)
        atr = market_state.get('atr', 0)
        # Simple breakout detection
        return volume > market_state.get('avg_volume', 0) * 1.5 and atr > market_state.get('avg_atr', 0) * 1.2

    def _is_mean_reversion(self, market_state: Dict[str, Any]) -> bool:
        """Check if current state shows mean reversion pattern."""
        # Simple mean reversion detection
        return abs(market_state.get('z_score', 0)) > 2.0

    def _is_liquidity_sweep(self, market_state: Dict[str, Any]) -> bool:
        """Check if current state shows liquidity sweep pattern."""
        # Simple liquidity sweep detection
        return market_state.get('wick_imbalance', 0) > 0.7

    def _is_favorable_market_conditions(self, market_state: Dict[str, Any]) -> bool:
        """Check if market conditions are favorable for tactical trading."""
        spread = market_state.get('spread', 0)
        volatility = market_state.get('volatility', 0)

        # Check spread is not too wide
        max_spread = market_state.get('max_spread', 0.001)
        if spread > max_spread:
            return False

        # Check for major news events
        if market_state.get('news_event', False):
            return False

        return True


def create_tactical_regime_detector():
    """Create and test tactical regime detector."""
    print("V24.2 Tactical Regime Detector")
    print("=" * 35)

    # Initialize detector
    detector = TacticalRegimeDetector()

    # Create sample market state
    sample_market_state = {
        'timestamp': '2026-04-12T10:00:00Z',
        'symbol': 'XAUUSD',
        'close': 2350.50,
        'high': 2355.25,
        'low': 2345.75,
        'open': 2348.00,
        'volume': 1000,
        'spread': 0.0008,
        'trend_strength': 0.8,
        'atr': 5.0,
        'avg_volume': 800,
        'avg_atr': 4.0,
        'z_score': 2.5,
        'wick_imbalance': 0.8,
        'volatility': 0.0015,
        'max_spread': 0.0015,
        'news_event': False
    }

    # Detect regime
    results = detector.detect_regime(sample_market_state)

    print("Tactical Regime Detection Results:")
    print(f"  Regime Type: {results['regime_type']}")
    print(f"  Regime Confidence: {results['regime_confidence']:.4f}")
    print(f"  Allow Tactical Trade: {results['allow_tactical_trade']}")

    return results


if __name__ == "__main__":
    # Run the tactical regime detection
    create_tactical_regime_detector()