"""
V24 CABR System Simple Test

This script tests the basic functionality of the CABR V24 system.
"""

import sys
import torch

# Add the project directory to the Python path
sys.path.append('.')

def test_cabr_basic():
    """Simple test to verify the CABR V24 system implementation."""
    print("Testing V24 CABR System")
    print("=" * 30)

    try:
        # Import the required modules
        from src.v24.cabr_v24 import CABRConfig, CABRRanker, MarketBranch
        from src.v24.world_state import WorldState

        print("SUCCESS: Modules imported successfully")

        # Test configuration creation
        config = CABRConfig()
        print("SUCCESS: CABRConfig created successfully")

        # Test CABR ranker creation
        cabr = CABRRanker(config)
        print("SUCCESS: CABRRanker created successfully")

        # Test that model has required attributes
        assert hasattr(cabr, 'confidence_weight'), "CABR should have confidence_weight parameter"
        print("SUCCESS: CABR model parameters verified")

        # Test basic branch creation
        branch = MarketBranch(
            branch_id="test_branch",
            path=torch.randn(10, 5).numpy(),  # Simple 10x5 path
            confidence=0.75,
            quality_score=0.82,
            uncertainty=0.15,
            timestamp="2026-04-12T10:00:00Z",
            metadata={"test": "data"}
        )
        print("SUCCESS: MarketBranch created successfully")

        # Test basic world state creation
        world_state = WorldState(
            timestamp="2026-04-12T10:00:00Z",
            symbol="XAUUSD",
            direction="BUY",
            market_structure={"close": 2350.50, "atr_pct": 0.0015, "vol_regime": 2},
            nexus_features={"cabr_score": 0.75, "confidence_score": 0.82},
            quant_models={"hmm_confidence": 0.66, "hmm_persistence_count": 3, "macro_vol_regime_class": 2},
            runtime_state={"rolling_win_rate_10": 0.55, "consecutive_losses": 0, "daily_drawdown_pct": 0.0},
            execution_context={"v22_risk_score": 0.25, "v22_meta_label_prob": 0.65, "v22_agreement_rate": 0.75}
        )
        print("SUCCESS: WorldState created successfully")

        print("\nAll basic tests passed!")
        return True

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_cabr_basic()