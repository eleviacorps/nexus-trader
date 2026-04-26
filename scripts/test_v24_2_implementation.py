"""
V24.2 System Integration Test

This script tests the complete V24.2 tactical mode implementation.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v24_2.tactical_regime import create_tactical_regime_detector
from src.v24_2.tactical_generator import create_tactical_generator
from src.v24_2.tactical_cabr import create_tactical_cabr
from src.v24_2.microstructure import create_microstructure_analyzer
from src.v24_2.tactical_calibration import create_tactical_calibration_model
from src.v24_2.integrated_engine import create_integrated_engine


def test_v24_2_implementation():
    """Test the complete V24.2 implementation."""
    print("V24.2 Implementation Test")
    print("=" * 30)

    # Test each component
    print("Testing V24.2 Tactical Mode Components...")

    # Test tactical regime detector
    print("\n1. Testing Tactical Regime Detector...")
    try:
        create_tactical_regime_detector()
        print("   [OK] Tactical regime detector working")
    except Exception as e:
        print(f"   [FAIL] Tactical regime detector failed: {e}")
        return False

    # Test tactical generator
    print("\n2. Testing Tactical Generator...")
    try:
        create_tactical_generator()
        print("   [OK] Tactical generator working")
    except Exception as e:
        print(f"   [FAIL] Tactical generator failed: {e}")
        return False

    # Test tactical CABR
    print("\n3. Testing Tactical CABR...")
    try:
        create_tactical_cabr()
        print("   [OK] Tactical CABR working")
    except Exception as e:
        print(f"   [FAIL] Tactical CABR failed: {e}")
        return False

    # Test microstructure analyzer
    print("\n4. Testing Microstructure Analyzer...")
    try:
        create_microstructure_analyzer()
        print("   [OK] Microstructure analyzer working")
    except Exception as e:
        print(f"   [FAIL] Microstructure analyzer failed: {e}")
        return False

    # Test tactical calibration
    print("\n5. Testing Tactical Calibration...")
    try:
        create_tactical_calibration_model()
        print("   [OK] Tactical calibration working")
    except Exception as e:
        print(f"   [FAIL] Tactical calibration failed: {e}")
        return False

    # Test integrated engine
    print("\n6. Testing Integrated Engine...")
    try:
        create_integrated_engine()
        print("   [OK] Integrated engine working")
    except Exception as e:
        print(f"   [FAIL] Integrated engine failed: {e}")
        return False

    print("\nV24.2 Implementation Test Complete!")
    print("All components loaded successfully.")
    return True


if __name__ == "__main__":
    success = test_v24_2_implementation()
    if success:
        print("\n[OK] V24.2 implementation test completed successfully!")
    else:
        print("\n[FAIL] V24.2 implementation test failed!")
        sys.exit(1)