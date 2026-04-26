"""
V24.3 Tactical Router
Routes trades to appropriate regime specialists.
"""
import pandas as pd
from typing import Dict, Any
from src.v24_3.regime_specialist import RegimeSpecialist, MarketRegime
from src.v24_3.execution_simulator import ExecutionSimulator


class TacticalRouter:
    """Routes trades to appropriate regime specialists."""

    def __init__(self):
        self.regime_specialist = RegimeSpecialist()
        self.execution_simulator = ExecutionSimulator()
        self.routing_history = []

    def route_trade(self, market_data: pd.DataFrame, strategic_signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a trade through the appropriate specialist based on market regime.

        Args:
            market_data (pd.DataFrame): Current market data
            strategic_signal (dict): Signal from strategic engine

        Returns:
            dict: Routed trade decision with execution analysis
        """
        # Detect current market regime
        regime = self.regime_specialist.detect_regime(market_data)

        # Log routing decision
        routing_entry = {
            'timestamp': pd.Timestamp.now(),
            'regime': regime.value,
            'strategic_signal': strategic_signal
        }
        self.routing_history.append(routing_entry)

        # Select appropriate specialist
        specialist = self.regime_specialist.select_specialist(regime)

        # Get regime-specific recommendation
        specialist_recommendation = self.regime_specialist.get_specialist_recommendation(regime, market_data)

        # Get regime-specific parameters
        regime_params = self.regime_specialist.get_regime_specific_parameters(regime)

        # Combine strategic signal with specialist recommendation
        combined_signal = self.combine_signals(strategic_signal, specialist_recommendation, regime_params)

        # Simulate execution costs
        execution_analysis = self.analyze_execution(combined_signal, market_data)

        # Final decision
        final_decision = self.make_final_decision(combined_signal, execution_analysis)

        return {
            'regime': regime.value,
            'specialist_recommendation': specialist_recommendation,
            'regime_parameters': regime_params,
            'execution_analysis': execution_analysis,
            'final_decision': final_decision
        }

    def combine_signals(self, strategic_signal: Dict[str, Any],
                       specialist_recommendation: Dict[str, Any],
                       regime_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine strategic and tactical signals.

        Args:
            strategic_signal (dict): Signal from strategic engine
            specialist_recommendation (dict): Recommendation from regime specialist
            regime_params (dict): Regime-specific parameters

        Returns:
            dict: Combined signal
        """
        # Simple combination logic - in practice this would be more sophisticated
        combined_confidence = (
            strategic_signal.get('confidence', 0.5) * 0.6 +
            specialist_recommendation.get('confidence', 0.5) * 0.4
        )

        return {
            'signal': specialist_recommendation.get('signal', 'hold'),
            'confidence': combined_confidence,
            'entry_price': specialist_recommendation.get('entry_price'),
            'tp': specialist_recommendation.get('tp'),
            'sl': specialist_recommendation.get('sl'),
            'tp_sl_ratio': regime_params.get('tp_sl_ratio', 2.0)
        }

    def analyze_execution(self, combined_signal: Dict[str, Any], market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze execution costs for the combined signal.

        Args:
            combined_signal (dict): Combined strategic and tactical signal
            market_data (pd.DataFrame): Market data

        Returns:
            dict: Execution analysis
        """
        # Prepare trade parameters for execution simulation
        trade_params = {
            'volume': 1.0,  # Standard lot size
            'slippage_risk': 0.3,  # Default slippage risk
            'market_volatility': market_data['close'].pct_change().std() if len(market_data) > 1 else 0.01,
            'execution_delay_ms': 100  # Default execution delay
        }

        # Calculate execution costs
        execution_costs = self.execution_simulator.calculate_execution_costs(trade_params)

        # Calculate net expectancy
        raw_expectancy = combined_signal.get('confidence', 0.1)  # Simplified expectancy
        net_expectancy = raw_expectancy - execution_costs['total_cost']

        # Determine if trade is viable
        is_viable = self.execution_simulator.evaluate_trade_viability(raw_expectancy, execution_costs)

        return {
            'execution_costs': execution_costs,
            'raw_expectancy': raw_expectancy,
            'net_expectancy': net_expectancy,
            'is_viable': is_viable
        }

    def make_final_decision(self, combined_signal: Dict[str, Any], execution_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make final trading decision based on all factors.

        Args:
            combined_signal (dict): Combined strategic and tactical signal
            execution_analysis (dict): Execution analysis results

        Returns:
            dict: Final trading decision
        """
        # Only trade if both strategic+tactical alignment exists and execution is viable
        should_trade = (
            combined_signal['signal'] != 'hold' and
            execution_analysis['is_viable'] and
            execution_analysis['net_expectancy'] > 0
        )

        return {
            'should_trade': should_trade,
            'signal': combined_signal['signal'] if should_trade else 'hold',
            'confidence': combined_signal['confidence'],
            'net_expectancy': execution_analysis['net_expectancy'],
            'execution_cost': execution_analysis['execution_costs']['total_cost']
        }

    def get_routing_statistics(self) -> Dict[str, Any]:
        """
        Get statistics on routing decisions.

        Returns:
            dict: Routing statistics
        """
        if not self.routing_history:
            return {'message': 'No routing history available'}

        # Count regime occurrences
        regime_counts = {}
        for entry in self.routing_history:
            regime = entry['regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        return {
            'total_routed_trades': len(self.routing_history),
            'regime_distribution': regime_counts,
            'most_common_regime': max(regime_counts, key=regime_counts.get) if regime_counts else None
        }


def main():
    """Example usage of the tactical router."""
    router = TacticalRouter()

    # Create sample market data
    dates = pd.date_range('2026-01-01', periods=100, freq='1min')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.random(100) * 100 + 1800,
        'high': np.random.random(100) * 100 + 1850,
        'low': np.random.random(100) * 100 + 1750,
        'close': np.random.random(100) * 100 + 1800,
        'volume': np.random.random(100) * 1000000
    })

    # Sample strategic signal
    strategic_signal = {
        'signal': 'buy',
        'confidence': 0.75,
        'reason': 'Strong bullish momentum detected'
    }

    # Route a trade
    decision = router.route_trade(sample_data.tail(50), strategic_signal)

    print("Tactical Routing Decision:")
    print(f"  Regime: {decision['regime']}")
    print(f"  Should Trade: {decision['final_decision']['should_trade']}")
    print(f"  Signal: {decision['final_decision']['signal']}")
    print(f"  Net Expectancy: {decision['final_decision']['net_expectancy']:.6f}")

    # Get routing statistics
    stats = router.get_routing_statistics()
    print(f"\nRouting Statistics: {stats}")


if __name__ == "__main__":
    main()