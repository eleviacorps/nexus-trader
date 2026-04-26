"""
V24.2 Integrated Engine

This module implements the integration between strategic and tactical engines.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
from datetime import datetime

from src.v24_2.tactical_regime import TacticalRegimeDetector
from src.v24_2.tactical_generator import TacticalGenerator
from src.v24_2.tactical_cabr import TacticalCABR
from src.v24_2.microstructure import MicrostructureAnalyzer
from src.v24_2.tactical_calibration import TacticalCalibrationModel


@dataclass
class IntegratedEngine:
    """Integrated engine for V24.2 strategic + tactical system."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.strategic_supervisor_active = self.config.get('strategic_supervisor_active', True)

        # Initialize all components
        self.tactical_regime_detector = TacticalRegimeDetector()
        self.tactical_generator = TacticalGenerator()
        self.tactical_cabr = TacticalCABR()
        self.microstructure_analyzer = MicrostructureAnalyzer()
        self.tactical_calibration = TacticalCalibrationModel()

    def make_trading_decision(self, market_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a complete trading decision using both strategic and tactical engines.

        Args:
            market_state: Current market state

        Returns:
            Complete trading decision
        """
        decision = {
            'timestamp': datetime.now().isoformat(),
            'strategic_direction': market_state.get('strategic_direction', 0),
            'strategic_confidence': market_state.get('strategic_confidence', 0.0),
            'tactical_enabled': False,
            'tactical_direction': 0,
            'tactical_confidence': 0.0,
            'should_trade': False,
            'trade_type': 'none',
            'position_size': 0.0,
            'reason': ''
        }

        # Step 1: Detect tactical regime
        regime_result = self.tactical_regime_detector.detect_regime(market_state)
        if not regime_result.get('allow_tactical_trade', False):
            decision['reason'] = 'Tactical trading not allowed in current regime'
            return decision

        # Step 2: Generate tactical paths
        tactical_paths = self.tactical_generator.generate_tactical_paths(market_state)
        if not tactical_paths:
            decision['reason'] = 'No tactical paths generated'
            return decision

        # Step 3: Analyze microstructure
        microstructure_result = self.microstructure_analyzer.analyze_microstructure(market_state)
        if not microstructure_result.get('tradeable', False):
            decision['reason'] = 'Microstructure conditions not favorable'
            return decision

        # Step 4: Calculate tactical tradeability
        tactical_cabr_result = self.tactical_cabr.calculate_tactical_tradeability(
            tactical_paths, market_state
        )

        # Step 5: Check strategic alignment
        strategic_direction = market_state.get('strategic_direction', 0)
        tactical_direction = self._get_tactical_direction(tactical_paths)

        if not self._directions_aligned(strategic_direction, tactical_direction):
            decision['reason'] = 'Tactical direction does not align with strategic direction'
            return decision

        # Step 6: Calculate tactical trade probability
        tactical_inputs = {
            'tactical_tradeability': tactical_cabr_result.get('short_tradeability', 0.0),
            'spread': market_state.get('spread', 0.001),
            'liquidity_sweep_probability': market_state.get('liquidity_sweep_probability', 0.3),
            'branch_disagreement': market_state.get('branch_disagreement', 0.3),
            'strategic_confidence': market_state.get('strategic_confidence', 0.7),
            'recent_tactical_win_rate': market_state.get('recent_tactical_win_rate', 0.6)
        }

        tactical_probability = self.tactical_calibration.calculate_tactical_trade_probability(
            tactical_inputs
        )

        # Step 7: Final decision
        tactical_tradeability = tactical_cabr_result.get('short_tradeability', 0.0)
        should_trade = self.tactical_calibration.should_trade_tactically(
            tactical_probability, tactical_tradeability
        )

        # Update decision with results
        decision.update({
            'tactical_enabled': True,
            'tactical_direction': tactical_direction,
            'tactical_confidence': tactical_probability,
            'should_trade': should_trade,
            'trade_type': 'tactical' if should_trade else 'strategic',
            'position_size': 0.1 if should_trade else 0.0,
            'reason': 'Tactical trade approved' if should_trade else 'Falling back to strategic'
        })

        return decision

    def _get_tactical_direction(self, tactical_paths: list) -> int:
        """Get the overall tactical direction from paths."""
        if not tactical_paths:
            return 0

        # Simple direction calculation from first path
        first_path = tactical_paths[0]
        expected_move = first_path.get('expected_move', 0)
        return 1 if expected_move > 0 else -1 if expected_move < 0 else 0

    def _directions_aligned(self, strategic_direction: int, tactical_direction: int) -> bool:
        """Check if strategic and tactical directions are aligned."""
        # Same direction or strategic is neutral (0)
        return strategic_direction == tactical_direction or strategic_direction == 0

    def evaluate_system_performance(self, historical_data: list) -> Dict[str, Any]:
        """
        Evaluate the performance of the integrated system.

        Args:
            historical_data: Historical trading data

        Returns:
            Performance evaluation results
        """
        # This would implement actual performance evaluation
        # For now, return placeholder results
        return {
            'total_return': 0.15,  # 15% return
            'win_rate': 0.65,       # 65% win rate
            'max_drawdown': 0.12,   # 12% drawdown
            'sharpe_ratio': 2.1,
            'evaluation_timestamp': datetime.now().isoformat()
        }

    def stress_test_system(self, test_scenarios: list) -> Dict[str, Any]:
        """
        Run stress tests on the system.

        Args:
            test_scenarios: List of market condition scenarios

        Returns:
            Stress test results
        """
        # This would implement actual stress testing
        # For now, return placeholder results
        return {
            'stress_tests_passed': len(test_scenarios),
            'stress_tests_total': len(test_scenarios),
            'worst_case_performance': -0.08,  # 8% worst case drawdown
            'average_performance': 0.12,       # 12% average return
            'test_timestamp': datetime.now().isoformat()
        }


def create_integrated_engine():
    """Create and test integrated engine."""
    print("V24.2 Integrated Engine")
    print("=" * 25)

    # Initialize integrated engine
    engine = IntegratedEngine()

    # Create sample market state
    sample_market_state = {
        'timestamp': '2026-04-12T10:00:00Z',
        'symbol': 'XAUUSD',
        'close': 2350.50,
        'high': 2355.25,
        'low': 2345.75,
        'open': 2348.00,
        'spread': 0.0008,
        'volatility': 0.0012,
        'strategic_direction': 1,
        'strategic_confidence': 0.8,
        'liquidity_sweep_probability': 0.4,
        'branch_disagreement': 0.25,
        'recent_tactical_win_rate': 0.65
    }

    # Make trading decision
    decision = engine.make_trading_decision(sample_market_state)

    print("Integrated Engine Decision:")
    print(f"  Should Trade: {decision['should_trade']}")
    print(f"  Trade Type: {decision['trade_type']}")
    print(f"  Tactical Enabled: {decision['tactical_enabled']}")
    print(f"  Reason: {decision['reason']}")

    return decision


if __name__ == "__main__":
    # Run the integrated engine
    create_integrated_engine()