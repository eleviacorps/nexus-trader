"""
V24.1 System Integration Test

This script tests the complete V24.1 implementation.
"""

import sys
import os

def test_v24_1_implementation():
    """Test the complete V24.1 implementation."""
    print("V24.1 Implementation Test")
    print("=" * 30)

    # Add current directory to Python path
    sys.path.insert(0, '.')

    # Test each module
    try:
        # Test branch realism module
        from src.v24_1.branch_realism import BranchRealismEvaluator
        evaluator = BranchRealismEvaluator()
        print("+ Branch realism module loaded successfully")

        # Test validation dataset module
        from src.v24_1.validation_dataset import ValidationDataset
        validator = ValidationDataset()
        print("+ Validation dataset module loaded successfully")

        # Test generator tournament module
        from src.v24_1.generator_tournament import GeneratorTournament
        tournament = GeneratorTournament()
        print("+ Generator tournament module loaded successfully")

        # Test CABR tradeability module
        from src.v24_1.cabr_tradeability import DangerousBranchCABR
        cabr = DangerousBranchCABR()
        print("+ CABR tradeability module loaded successfully")

        # Test evolution runner module
        from src.v24_1.evolution_runner import EvolutionRunner
        evolution_runner = EvolutionRunner()
        print("+ Evolution runner module loaded successfully")

        # Test calibration model module
        from src.v24_1.calibration_model import CalibrationModel
        calibration = CalibrationModel()
        print("+ Calibration model module loaded successfully")

        print("\nAll V24.1 modules loaded successfully!")
        return True

    except Exception as e:
        print(f"Error loading modules: {e}")
        return False

if __name__ == "__main__":
    success = test_v24_1_implementation()
    if success:
        print("V24.1 implementation test completed successfully!")
    else:
        print("V24.1 implementation test failed!")