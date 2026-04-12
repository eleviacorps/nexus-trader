"""
V24 CABR System Demo

This script demonstrates the CABR V24 system in action.
"""

import sys
import torch
import numpy as np

# Add the project directory to the Python path
sys.path.append('.')

def demo_cabr_system():
    """Demo script to show the CABR V24 system in action."""
    print("V24 CABR (Confidence-Aware Branch Ranking) System Demo")
    print("=" * 55)

    try:
        # Import the required modules
        from src.v24.cabr_v24 import CABRConfig, CABRRanker, MarketBranch
        from src.v24.world_state import WorldState

        print("Creating CABR system...")
        # Create configuration
        config = CABRConfig()
        cabr = CABRRanker(config)
        print("SUCCESS: CABR system created")

        print("\nCreating sample market branches...")
        # Create sample branches
        sample_branches = [
            MarketBranch(
                branch_id=f"branch_{i}",
                path=np.random.randn(30, 36),  # 30 time steps, 36 features
                confidence=0.6 + 0.1 * i,
                quality_score=0.4 + 0.1 * i,
                uncertainty=0.2 - 0.03 * i,
                timestamp="2026-04-12T10:00:00Z",
                metadata={"source": "diffusion_model", "branch_index": i}
            )
            for i in range(5)
        ]
        print(f"SUCCESS: Created {len(sample_branches)} sample branches")

        print("\nCreating market state...")
        # Create sample world state
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
        print("SUCCESS: Market state created")

        print("\nRanking branches...")
        # Rank the branches
        ranked_branches = cabr.rank_branches(sample_branches, world_state)
        print(f"SUCCESS: Ranked {len(ranked_branches)} branches")

        print("\nTop branches:")
        for i, branch in enumerate(ranked_branches[:3]):  # Show top 3
            print(f"  {i+1}. Branch {branch.branch_id}:")
            print(f"     Confidence: {branch.confidence:.3f}")
            print(f"     Quality Score: {branch.quality_score:.3f}")
            print(f"     Uncertainty: {branch.uncertainty:.3f}")

        print("\nSelecting best branches...")
        # Select top branches
        best_branches = cabr.select_best_branches(sample_branches, world_state, num_branches=3)
        print(f"SUCCESS: Selected {len(best_branches)} best branches")

        print("\nDemo completed successfully!")
        return True

    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    demo_cabr_system()