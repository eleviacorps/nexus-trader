"""
V24 CABR (Confidence-Aware Branch Ranking) Tests

This module contains tests for the CABR V24 implementation.
"""

import unittest
from typing import Any, Dict

import numpy as np
import torch

from src.v24.cabr_v24 import CABRConfig, CABRRanker, MarketBranch
from src.v24.world_state import WorldState


class TestCABRV24(unittest.TestCase):
    """Test cases for the CABR V24 system."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = CABRConfig()
        self.cabr = CABRRanker(self.config)

    def test_cabr_initialization(self):
        """Test that CABR system initializes correctly."""
        self.assertIsInstance(self.cabr, CABRRanker)
        self.assertIsInstance(self.cabr.config, CABRConfig)

    def test_market_branch_creation(self):
        """Test MarketBranch creation."""
        branch = MarketBranch(
            branch_id="test_branch",
            path=np.random.randn(30, 36),
            confidence=0.75,
            quality_score=0.82,
            uncertainty=0.15,
            timestamp="2026-04-12T10:00:00Z",
            metadata={"test": "data"}
        )

        self.assertEqual(branch.branch_id, "test_branch")
        self.assertEqual(branch.confidence, 0.75)
        self.assertEqual(branch.quality_score, 0.82)
        self.assertEqual(branch.uncertainty, 0.15)

    def test_cabr_ranking_basic(self):
        """Test basic branch ranking functionality."""
        # Create sample branches
        branches = [
            MarketBranch(
                branch_id=f"branch_{i}",
                path=np.random.randn(30, 36),
                confidence=0.7 + 0.05 * i,
                quality_score=0.5 + 0.1 * i,
                uncertainty=0.1 * i,
                timestamp="2026-04-12T10:00:00Z",
                metadata={}
            )
            for i in range(3)
        ]

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

        # Test ranking
        ranked_branches = self.cabr.rank_branches(branches, world_state)

        self.assertIsInstance(ranked_branches, list)
        self.assertLessEqual(len(ranked_branches), len(branches))

    def test_cabr_model_parameters(self):
        """Test that CABR model has required parameters."""
        self.assertTrue(hasattr(self.cabr, 'confidence_weight'))
        self.assertTrue(hasattr(self.cabr, 'diversity_weight'))
        self.assertTrue(hasattr(self.cabr, 'quality_weight'))

    def test_diversity_scoring(self):
        """Test diversity scoring functionality."""
        # Create a branch with some path data
        branch = MarketBranch(
            branch_id="diversity_test",
            path=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),  # Simple 2D path
            confidence=0.75,
            quality_score=0.82,
            uncertainty=0.15,
            timestamp="2026-04-12T10:00:00Z",
            metadata={}
        )

        # Test that diversity scoring works
        diversity_score = self.cabr._calculate_diversity_score(branch)
        self.assertIsInstance(diversity_score, float)
        self.assertGreaterEqual(diversity_score, 0.0)
        self.assertLessEqual(diversity_score, 1.0)

    def test_regime_adjustment(self):
        """Test regime adjustment calculation."""
        # Create world state with different regime classes
        world_state_low_vol = WorldState(
            timestamp="2026-04-12T10:00:00Z",
            symbol="XAUUSD",
            direction="BUY",
            market_structure={"close": 2350.50},
            nexus_features={"cabr_score": 0.75},
            quant_models={"macro_vol_regime_class": 1},  # Low volatility
            runtime_state={},
            execution_context={}
        )

        world_state_high_vol = WorldState(
            timestamp="2026-04-12T10:00:00Z",
            symbol="XAUUSD",
            direction="BUY",
            market_structure={"close": 2350.50},
            nexus_features={"cabr_score": 0.75},
            quant_models={"macro_vol_regime_class": 3},  # High volatility
            runtime_state={},
            execution_context={}
        )

        # Test adjustments
        adjustment_low = self.cabr._calculate_regime_adjustment(world_state_low_vol)
        adjustment_high = self.cabr._calculate_regime_adjustment(world_state_high_vol)

        self.assertIsInstance(adjustment_low, float)
        self.assertIsInstance(adjustment_high, float)

    def test_branch_selection(self):
        """Test branch selection functionality."""
        # Create sample branches
        branches = [
            MarketBranch(
                branch_id=f"select_branch_{i}",
                path=np.random.randn(30, 36),
                confidence=0.7 + 0.05 * i,
                quality_score=0.5 + 0.1 * i,
                uncertainty=0.1 * i,
                timestamp="2026-04-12T10:00:00Z",
                metadata={}
            )
            for i in range(5)
        ]

        # Create sample world state
        world_state = WorldState(
            timestamp="2026-04-12T10:00:00Z",
            symbol="XAUUSD",
            direction="BUY",
            market_structure={"close": 2350.50},
            nexus_features={"cabr_score": 0.75},
            quant_models={"macro_vol_regime_class": 2},
            runtime_state={},
            execution_context={}
        )

        # Test branch selection
        selected_branches = self.cabr.select_best_branches(branches, world_state, num_branches=3)

        self.assertIsInstance(selected_branches, list)
        self.assertLessEqual(len(selected_branches), 3)


class TestCABREnsemble(unittest.TestCase):
    """Test cases for CABR ensemble functionality."""

    def test_ensemble_creation(self):
        """Test that CABR ensemble can be created."""
        configs = [CABRConfig(), CABRConfig(confidence_threshold=0.7)]
        from src.v24.cabr_v24 import CABREnsemble
        ensemble = CABREnsemble(configs)

        self.assertIsInstance(ensemble, CABREnsemble)
        self.assertEqual(len(ensemble.rankers), 2)


def run_all_tests():
    """Run all CABR tests."""
    print("Running CABR V24 Tests")
    print("=" * 30)

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print test results
    print(f"\nTest Results:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Success: {result.wasSuccessful()}")

    return result.wasSuccessful()


if __name__ == "__main__":
    run_all_tests()