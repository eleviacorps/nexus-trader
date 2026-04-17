"""
V24.4 Tactical Router with Adaptive Admission Layer
Routes trades through appropriate regime specialists with V24.4 enhancements.
"""
import pandas as pd
from typing import Dict, Any
from src.v24_4.admission_layer import AdaptiveAdmissionLayer, MarketRegime
from src.v24_4.position_sizer import PositionSizer
from src.v24_4.cooldown_manager import CooldownManager
from src.v24_4.trade_cluster_filter import TradeClusterFilter
from src.v24_4.regime_profitability import RegimeProfitability


class V24_4TacticalRouter:
    """Enhanced tactical router with V24.4 adaptive admission layer."""

    def __init__(self):
        self.admission_layer = AdaptiveAdmissionLayer()
        self.position_sizer = PositionSizer()
        self.cooldown_manager = CooldownManager()
        self.trade_cluster_filter = TradeClusterFilter()
        self.regime_profitability = RegimeProfitability()
        self.routing_history = []

    def route_trade_with_admission(self, market_data: pd.DataFrame,
                                 strategic_signal: Dict[str, Any],
                                 tactical_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a trade with adaptive admission layer.

        Args:
            market_data: Current market data
            strategic_signal: Signal from strategic engine
            tactical_data: Tactical analysis data

        Returns:
            Dict containing routing decision
        """
        # Calculate admission score
        admission_score = self.admission_layer.calculate_admission_score(
            tactical_trade_probability=tactical_data.get('tactical_probability', 0.5),
            tactical_tradeability=tactical_data.get('tradeability', 0.5),
            execution_quality=tactical_data.get('execution_quality', 0.5),
            strategic_confidence=strategic_signal.get('confidence', 0.5),
            regime_specialist_confidence=tactical_data.get('regime_confidence', 0.5)
        )

        # Determine market regime
        regime = self._detect_regime(market_data)

        # Check if trade should be admitted
        should_admit = self.admission_layer.should_admit_trade(regime, admission_score)

        if not should_admit:
            return {
                'should_trade': False,
                'reason': 'Trade not admitted by admission layer',
                'admission_score': admission_score
            }

        # Calculate position size
        position_size = self.position_sizer.calculate_position_size(admission_score)

        # Apply cooldown if needed
        current_threshold = self.cooldown_manager.get_current_threshold(
            self.admission_layer.get_adaptive_threshold(regime)
        )

        return {
            'should_trade': True,
            'position_size': position_size,
            'admission_score': admission_score,
            'current_threshold': current_threshold,
            'regime': regime.value
        }

    def _detect_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """
        Detect current market regime.

        Args:
            market_data: Market data

        Returns:
            MarketRegime: Detected regime
        """
        # Simple regime detection based on price action and volatility
        recent_returns = market_data['close'].pct_change().dropna()
        volatility = recent_returns.std()
        trend_strength = abs(recent_returns.mean() / volatility if volatility > 0 else 0)

        if trend_strength > 0.5:
            return MarketRegime.TREND
        elif volatility > recent_returns.std() * 1.5:
            return MarketRegime.BREAKOUT
        elif trend_strength < 0.1 and volatility > recent_returns.std():
            return MarketRegime.LIQUIDITY_SWEEP
        else:
            return MarketRegime.MEAN_REVERSION


def main():
    """Example usage of V24.4 tactical router."""
    print("V24.4 Tactical Router")
    print("This enhanced router incorporates adaptive admission layer and V24.4 components.")


if __name__ == "__main__":
    main()