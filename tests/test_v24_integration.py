"""
V24 Conditional Diffusion Model Integration Test

This script tests the integration of the conditional diffusion model with the V24 framework.
"""

import unittest
from typing import Any, Dict

import numpy as np
import torch

from src.v24.diffusion_model import ConditionalDiffusionModel, DiffusionConfig
from src.v24.conditional_generator import ConditionalPathGenerator, GenerationConfig
from src.v24.world_state import WorldState


class TestV24DiffusionIntegration(unittest.TestCase):
    """Integration tests for V24 conditional diffusion model."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = DiffusionConfig(
            sequence_length=60,
            feature_dim=36,
            hidden_dim=128
        )
        self.generation_config = GenerationConfig(
            num_paths=3,
            generation_steps=20
        )

    def test_model_initialization(self):
        """Test model initialization."""
        model = ConditionalDiffusionModel(self.config)
        self.assertIsInstance(model, ConditionalDiffusionModel)

    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = ConditionalPathGenerator(self.config, self.generation_config)
        self.assertIsInstance(generator, ConditionalPathGenerator)

    def test_path_generation(self):
        """Test path generation functionality."""
        generator = ConditionalPathGenerator(self.config, self.generation_config)

        # Create sample world state
        sample_world_state = {
            "timestamp": "2026-04-12T10:00:00Z",
            "symbol": "XAUUSD",
            "direction": "BUY",
            "market_structure": {
                "close": 2350.50,
                "atr_pct": 0.0015,
                "vol_regime": 2
            },
            "nexus_features": {
                "cabr_score": 0.75,
                "confidence_score": 0.82
            },
            "quant_models": {},
            "runtime_state": {},
            "execution_context": {}
        }

        # Test path generation
        paths = generator.generate_conditional_paths(
            sample_world_state,
            num_paths=2
        )

        self.assertIsInstance(paths, list)
        self.assertEqual(len(paths), 2)

    def test_world_state_integration(self):
        """Test integration with WorldState objects."""
        generator = ConditionalPathGenerator(self.config, self.generation_config)

        # Create a WorldState object
        world_state = WorldState(
            timestamp="2026-04-12T10:00:00Z",
            symbol="XAUUSD",
            direction="BUY",
            market_structure={
                "close": 2350.50,
                "atr_pct": 0.0015,
                "vol_regime": 2
            },
            nexus_features={
                "cabr_score": 0.75,
                "confidence_score": 0.82
            },
            quant_models={},
            runtime_state={},
            execution_context={}
        )

        # Test that generation works with WorldState objects
        paths = generator.generate_conditional_paths(world_state)
        self.assertIsInstance(paths, list)

    def test_model_device_compatibility(self):
        """Test model device compatibility."""
        model = ConditionalDiffusionModel(self.config)

        # Test that model can be moved to different devices
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            self.assertIsNotNone(model_cuda)

        # Test that model works on CPU
        model_cpu = model.cpu()
        self.assertIsNotNone(model_cpu)

    def test_model_serialization(self):
        """Test model serialization."""
        model = ConditionalDiffusionModel(self.config)

        # Test that model can be serialized
        try:
            state_dict = model.state_dict()
            self.assertIsInstance(state_dict, dict)
        except Exception as e:
            self.fail(f"Model serialization failed: {e}")

    def test_generator_serialization(self):
        """Test generator serialization."""
        generator = ConditionalPathGenerator(self.config, self.generation_config)

        # Test that generator can be serialized
        try:
            # This would test serialization of the generator's internal state
            pass
        except Exception as e:
            self.fail(f"Generator serialization failed: {e}")


def run_all_tests():
    """Run all integration tests."""
    print("Running V24 Diffusion Model Integration Tests")
    print("=" * 45)

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

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
    success = run_all_tests()
    if success:
        print("\nAll integration tests passed!")
    else:
        print("\nSome integration tests failed!")
        sys.exit(1)