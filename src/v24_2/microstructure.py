"""
V24.2 Microstructure Layer

This module implements microstructure analysis for tactical trading.
"""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from datetime import datetime


@dataclass
class MicrostructureAnalyzer:
    """Microstructure analyzer for tactical trading."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_execution_quality = self.config.get('min_execution_quality', 0.0)

    def analyze_microstructure(self, market_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze microstructure factors for tactical trading.

        Args:
            market_state: Current market state data

        Returns:
            Microstructure analysis results
        """
        # Calculate microstructure metrics
        spread_regime = self._calculate_spread_regime(market_state)
        spread_expansion_rate = self._calculate_spread_expansion_rate(market_state)
        wick_imbalance = self._calculate_wick_imbalance(market_state)
        liquidity_sweep_probability = self._calculate_liquidity_sweep_probability(market_state)
        estimated_slippage = self._estimate_slippage(market_state)
        execution_quality = self._calculate_execution_quality(
            market_state, estimated_slippage
        )

        return {
            'spread_regime': spread_regime,
            'spread_expansion_rate': spread_expansion_rate,
            'wick_imbalance': wick_imbalance,
            'liquidity_sweep_probability': liquidity_sweep_probability,
            'estimated_slippage': estimated_slippage,
            'execution_quality': execution_quality,
            'tradeable': execution_quality > self.min_execution_quality,
            'analysis_timestamp': datetime.now().isoformat()
        }

    def _calculate_spread_regime(self, market_state: Dict[str, Any]) -> str:
        """Calculate current spread regime."""
        spread = market_state.get('spread', 0)
        avg_spread = market_state.get('avg_spread', 0.001)

        if spread > avg_spread * 2:
            return "wide"
        elif spread > avg_spread * 1.5:
            return "moderate"
        else:
            return "tight"

    def _calculate_spread_expansion_rate(self, market_state: Dict[str, Any]) -> float:
        """Calculate rate of spread expansion."""
        current_spread = market_state.get('spread', 0)
        previous_spread = market_state.get('previous_spread', current_spread)

        if previous_spread == 0:
            return 0.0

        return (current_spread - previous_spread) / previous_spread

    def _calculate_wick_imbalance(self, market_state: Dict[str, Any]) -> float:
        """Calculate wick imbalance."""
        high = market_state.get('high', 0)
        low = market_state.get('low', 0)
        open_price = market_state.get('open', 0)
        close = market_state.get('close', 0)

        upper_wick = high - max(open_price, close)
        lower_wick = min(open_price, close) - low

        if upper_wick + lower_wick == 0:
            return 0.0

        return (upper_wick - lower_wick) / (upper_wick + lower_wick)

    def _calculate_liquidity_sweep_probability(self, market_state: Dict[str, Any]) -> float:
        """Calculate probability of liquidity sweep."""
        # Simple liquidity sweep detection
        wick_imbalance = abs(self._calculate_wick_imbalance(market_state))
        volume_imbalance = market_state.get('volume_imbalance', 0)

        # Higher wick imbalance and volume imbalance suggest higher probability
        return min(1.0, (wick_imbalance + abs(volume_imbalance)) / 2)

    def _estimate_slippage(self, market_state: Dict[str, Any]) -> float:
        """Estimate slippage cost."""
        spread = market_state.get('spread', 0)
        volatility = market_state.get('volatility', 0)
        volume = market_state.get('volume', 0)
        avg_volume = market_state.get('avg_volume', 1000)

        # Slippage increases with spread, volatility, and low volume
        slippage_factor = (spread * 0.5) + (volatility * 0.3) + (max(0, 1 - volume/avg_volume) * 0.2)
        return slippage_factor

    def _calculate_execution_quality(self, market_state: Dict[str, Any], slippage: float) -> float:
        """Calculate execution quality score."""
        expected_profit = market_state.get('expected_profit', 0.01)

        # Calculate total costs
        spread_cost = market_state.get('spread', 0) * 2  # Bid-ask spread cost
        slippage_cost = slippage

        # Execution quality is expected profit minus costs
        execution_quality = expected_profit - spread_cost - slippage_cost
        return execution_quality

    def should_trade_microstructure(self, analysis: Dict[str, Any]) -> bool:
        """Determine if trade should proceed based on microstructure analysis."""
        execution_quality = analysis.get('execution_quality', 0)
        return execution_quality > self.min_execution_quality


def create_microstructure_analyzer():
    """Create and test microstructure analyzer."""
    print("V24.2 Microstructure Analyzer")
    print("=" * 30)

    # Initialize analyzer
    analyzer = MicrostructureAnalyzer({
        'min_execution_quality': 0.0
    })

    # Create sample market state
    sample_market_state = {
        'timestamp': '2026-04-12T10:00:00Z',
        'symbol': 'XAUUSD',
        'close': 2350.50,
        'high': 2355.25,
        'low': 2345.75,
        'open': 2348.00,
        'spread': 0.0008,
        'previous_spread': 0.0007,
        'avg_spread': 0.001,
        'volatility': 0.0012,
        'volume': 1000,
        'avg_volume': 800,
        'volume_imbalance': 0.1,
        'expected_profit': 0.015
    }

    # Analyze microstructure
    results = analyzer.analyze_microstructure(sample_market_state)

    print("Microstructure Analysis Results:")
    print(f"  Spread Regime: {results['spread_regime']}")
    print(f"  Spread Expansion Rate: {results['spread_expansion_rate']:.6f}")
    print(f"  Wick Imbalance: {results['wick_imbalance']:.4f}")
    print(f"  Liquidity Sweep Probability: {results['liquidity_sweep_probability']:.4f}")
    print(f"  Estimated Slippage: {results['estimated_slippage']:.6f}")
    print(f"  Execution Quality: {results['execution_quality']:.6f}")
    print(f"  Tradeable: {results['tradeable']}")

    return results


if __name__ == "__main__":
    # Run the microstructure analyzer
    create_microstructure_analyzer()