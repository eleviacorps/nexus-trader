"""
V24.3 Test Suite
Test suite to verify V24.3 components are working correctly.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.v24_3.execution_dataset import ExecutionDataset
from src.v24_3.execution_simulator import ExecutionSimulator
from src.v24_3.regime_specialist import RegimeSpecialist, MarketRegime
from src.v24_3.tactical_router import TacticalRouter
import pandas as pd
import numpy as np


def test_execution_dataset():
    """Test the execution dataset creation."""
    print("Testing Execution Dataset...")

    dataset = ExecutionDataset()
    data = dataset.create_dataset("XAUUSD", "2026-01-01", "2026-01-02")

    # Check that we have data
    assert len(data) > 0, "Dataset should not be empty"
    assert 'net_trade_outcome_after_execution' in data.columns, "Dataset should have net outcome column"

    print("[PASS] Execution Dataset test passed")
    return True


def test_execution_simulator():
    """Test the execution simulator."""
    print("Testing Execution Simulator...")

    simulator = ExecutionSimulator()

    # Test cost calculation
    trade_params = {
        'volume': 1.0,
        'slippage_risk': 0.3,
        'market_volatility': 1.2,
        'execution_delay_ms': 150
    }

    costs = simulator.calculate_execution_costs(trade_params)
    assert 'total_cost' in costs, "Costs should include total_cost"
    assert costs['total_cost'] >= 0, "Total cost should be non-negative"

    # Test trade viability
    is_viable = simulator.evaluate_trade_viability(0.15, costs)
    assert isinstance(is_viable, bool), "evaluate_trade_viability should return boolean"

    print("[PASS] Execution Simulator test passed")
    return True


def test_regime_specialist():
    """Test the regime specialist."""
    print("Testing Regime Specialist...")

    specialist = RegimeSpecialist()

    # Create sample data for regime detection
    dates = pd.date_range('2026-01-01', periods=100, freq='1min')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.random(100) * 100 + 1800,
        'high': np.random.random(100) * 100 + 1850,
        'low': np.random.random(100) * 100 + 1750,
        'close': np.random.random(100) * 100 + 1800,
        'volume': np.random.random(100) * 1000000
    })

    # Test regime detection
    regime = specialist.detect_regime(sample_data)
    assert isinstance(regime, MarketRegime), "Regime should be a MarketRegime enum"

    # Test regime parameters
    params = specialist.get_regime_specific_parameters(regime)
    assert isinstance(params, dict), "Regime parameters should be a dictionary"

    print("[PASS] Regime Specialist test passed")
    return True


def test_tactical_router():
    """Test the tactical router."""
    print("Testing Tactical Router...")

    # Create sample data
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
        'reason': 'Test signal'
    }

    # Test routing
    router = TacticalRouter()
    decision = router.route_trade(sample_data.tail(50), strategic_signal)

    assert 'regime' in decision, "Decision should include regime"
    assert 'final_decision' in decision, "Decision should include final_decision"

    print("[PASS] Tactical Router test passed")
    return True


def run_all_tests():
    """Run all V24.3 tests."""
    print("Running V24.3 Test Suite")
    print("=" * 30)

    try:
        test_execution_dataset()
        test_execution_simulator()
        test_regime_specialist()
        test_tactical_router()

        print("\n" + "=" * 30)
        print("All V24.3 tests passed! [PASS]")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        return False


if __name__ == "__main__":
    run_all_tests()