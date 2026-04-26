"""
V24.4 Regime Profitability Analysis
Tracks and analyzes profitability by market regime.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from collections import defaultdict


class RegimeProfitability:
    """Track and analyze profitability by market regime."""

    def __init__(self):
        self.regime_performance = defaultdict(list)
        self.regime_metrics = defaultdict(dict)
        self.trade_history = []

    def track_trade_performance(self, regime: str, trade_result: Dict[str, Any]):
        """
        Track performance of a trade by regime.

        Args:
            regime: Market regime identifier
            trade_result: Dictionary containing trade results
        """
        # Record trade in history
        trade_record = {
            'regime': regime,
            'timestamp': trade_result.get('timestamp'),
            'pnl': trade_result.get('pnl', 0),
            'win': trade_result.get('win', False),
            'trade_data': trade_result
        }
        self.trade_history.append(trade_record)

        # Track performance metrics
        self.regime_performance[regime].append(trade_result)

    def calculate_regime_metrics(self, regime: str) -> Dict[str, float]:
        """
        Calculate performance metrics for a specific regime.

        Args:
            regime: Market regime identifier

        Returns:
            Dict containing regime metrics
        """
        regime_trades = self.regime_performance[regime]
        if not regime_trades:
            return {}

        # Calculate win rate
        wins = sum(1 for trade in regime_trades if trade.get('win', False))
        win_rate = wins / len(regime_trades) if regime_trades else 0

        # Calculate average expectancy
        total_pnl = sum(trade.get('pnl', 0) for trade in regime_trades)
        avg_pnl = total_pnl / len(regime_trades) if regime_trades else 0

        # Calculate profit factor
        winning_trades = [trade for trade in regime_trades if trade.get('win', False)]
        losing_trades = [trade for trade in regime_trades if not trade.get('win', False)]

        winning_pnl = abs(sum(trade.get('pnl', 0) for trade in winning_trades))
        losing_pnl = abs(sum(trade.get('pnl', 0) for trade in losing_trades))

        profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else float('inf')

        return {
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'profit_factor': profit_factor,
            'trade_count': len(regime_trades)
        }

    def adjust_regime_weights(self, regime: str) -> float:
        """
        Adjust regime weights based on profitability.

        Args:
            regime: Market regime identifier

        Returns:
            float: Adjusted weight for the regime
        """
        metrics = self.calculate_regime_metrics(regime)

        # If regime has negative expectancy, reduce weight
        if metrics.get('avg_pnl', 0) < 0:
            return 0.5  # Reduced weight for poor performing regimes
        elif metrics.get('win_rate', 0) < 0.5:
            return 0.7  # Reduced weight for low win rate regimes
        else:
            return 1.0  # Full weight for good performance

    def get_regime_expectancy(self, regime: str) -> float:
        """
        Get expectancy for a specific regime.

        Args:
            regime: Market regime identifier

        Returns:
            float: Regime expectancy
        """
        metrics = self.calculate_regime_metrics(regime)
        return metrics.get('avg_pnl', 0)

    def should_reduce_regime_weight(self, regime: str) -> bool:
        """
        Determine if regime weight should be reduced based on performance.

        Args:
            regime: Market regime identifier

        Returns:
            bool: Whether regime weight should be reduced
        """
        metrics = self.calculate_regime_metrics(regime)
        return metrics.get('avg_pnl', 0) < -0.05 or metrics.get('win_rate', 1.0) < 0.4


def main():
    """Example usage of regime profitability analysis."""
    # This would typically be integrated with the main trading system
    print("Regime Profitability Analysis")
    print("This component tracks performance by market regime and adjusts weights accordingly.")


if __name__ == "__main__":
    main()