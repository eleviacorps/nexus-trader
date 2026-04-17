"""
V24.4 Validation Script
Script to validate V24.4 components and performance.
"""
import json
import os
from typing import Dict, Any


def validate_v24_4_components():
    """Validate that all V24.4 components are working correctly."""
    print("Validating V24.4 components...")

    # Test admission layer
    try:
        from src.v24_4.admission_layer import AdaptiveAdmissionLayer, MarketRegime
        admission_layer = AdaptiveAdmissionLayer()

        # Test admission score calculation
        score = admission_layer.calculate_admission_score(0.8, 0.7, 0.9, 0.85, 0.75)
        print(f"Admission score test: {score}")

        # Test regime detection
        regime = MarketRegime.TREND
        should_admit = admission_layer.should_admit_trade(regime, 0.8)
        print(f"Trade admission test: {should_admit}")

    except Exception as e:
        print(f"Error testing admission layer: {e}")

    # Test position sizer
    try:
        from src.v24_4.position_sizer import PositionSizer
        sizer = PositionSizer()
        position_size = sizer.calculate_position_size(0.85)
        print(f"Position size test: {position_size}")
    except Exception as e:
        print(f"Error testing position sizer: {e}")

    # Test regime profitability
    try:
        from src.v24_4.regime_profitability import RegimeProfitability
        profitability = RegimeProfitability()
        print("Regime profitability component loaded successfully")
    except Exception as e:
        print(f"Error testing regime profitability: {e}")

    print("V24.4 component validation completed")


def main():
    """Main validation function."""
    validate_v24_4_components()


if __name__ == "__main__":
    main()