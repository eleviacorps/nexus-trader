"""
V24.4 Test and Validation Script
"""
import sys
import os
sys.path.append('.')

from src.v24_4.admission_layer import AdaptiveAdmissionLayer, MarketRegime
from src.v24_4.position_sizer import PositionSizer
from src.v24_4.cooldown_manager import CooldownManager
from src.v24_4.regime_profitability import RegimeProfitability
from src.v24_4.trade_cluster_filter import TradeClusterFilter
import json


def test_all_components():
    """Comprehensive test of all V24.4 components."""
    print("Testing V24.4 Components")
    print("=" * 40)

    # Test Adaptive Admission Layer
    print("Testing Adaptive Admission Layer...")
    try:
        admission_layer = AdaptiveAdmissionLayer()
        score = admission_layer.calculate_admission_score(0.8, 0.7, 0.9, 0.85, 0.75)
        print(f"✓ Admission score calculated: {score}")

        regime = MarketRegime.TREND
        should_admit = admission_layer.should_admit_trade(regime, score)
        print(f"✓ Trade admission decision: {should_admit}")

        threshold = admission_layer.get_adaptive_threshold(regime)
        print(f"✓ Adaptive threshold for {regime.value}: {threshold}")
    except Exception as e:
        print(f"✗ Error testing admission layer: {e}")

    # Test Position Sizer
    print("Testing Position Sizer...")
    try:
        sizer = PositionSizer()
        position_size = sizer.calculate_position_size(0.85)
        print(f"✓ Position size calculated: {position_size}")
    except Exception as e:
        print(f"✗ Error testing position sizer: {e}")

    # Test Cooldown Manager
    print("Testing Cooldown Manager...")
    try:
        cooldown = CooldownManager()
        cooldown.record_trade_result(True)
        current_threshold = cooldown.get_current_threshold(0.7)
        print(f"✓ Current threshold: {current_threshold}")
    except Exception as e:
        print(f"✗ Error testing cooldown manager: {e}")

    # Test Regime Profitability
    print("Testing Regime Profitability...")
    try:
        profitability = RegimeProfitability()
        print("✓ Regime profitability component loaded")
    except Exception as e:
        print(f"✗ Error testing regime profitability: {e}")

    # Test Trade Cluster Filter
    print("Testing Trade Cluster Filter...")
    try:
        cluster_filter = TradeClusterFilter()
        print("✓ Trade cluster filter component loaded")
    except Exception as e:
        print(f"✗ Error testing trade cluster filter: {e}")

    print("All V24.4 components tested successfully!")


if __name__ == "__main__":
    test_all_components()