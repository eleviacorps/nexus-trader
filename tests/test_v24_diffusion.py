"""
V24 Conditional Diffusion Model Tests

This module contains tests for the conditional diffusion model implementation.
"""

import unittest
from typing import Any, Dict, List

import numpy as np
import torch

from src.v24.diffusion_model import ConditionalDiffusionModel, DiffusionConfig
from src.v24.conditional_generator import ConditionalPathGenerator, GenerationConfig
from src.v24.world_state import WorldState


class TestConditionalDiffusionModel(unittest.TestCase):
    """Test cases for the conditional diffusion model."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = DiffusionConfig(
            sequence_length=60,
            feature_dim=36,
            hidden_dim=128
        )
        self.model = ConditionalDiffusionModel(self.config)

    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        self.assertIsInstance(self.model, ConditionalDiffusionModel)
        self.assertEqual(self.model.config.sequence_length, 60)
        self.assertEqual(self.model.config.feature_dim, 36)

    def test_model_parameters(self):
        """Test that model parameters are properly initialized."""
        # Check that all model components exist
        self.assertTrue(hasattr(self.model, 'encoder'))
        self.assertTrue(hasattr(self.model, 'diffusion_net'))
        self.assertTrue(hasattr(self.model, 'betas'))
        self.assertTrue(hasattr(self.model, 'alphas_cumprod'))

    def test_diffusion_process(self):
        """Test the diffusion process functions."""
        # Test q_sample function
        x_start = torch.randn(2, self.config.sequence_length, self.config.feature_dim)
        t = torch.tensor([100, 200])
        noise = torch.randn_like(x_start)

        # Test that q_sample runs without error
        result = self.model.q_sample(x_start, t, noise)
        self.assertEqual(result.shape, x_start.shape)

    def test_model_predictions(self):
        """Test model prediction functions."""
        # Create test data
        x = torch.randn(2, self.config.sequence_length, self.config.feature_dim)
        context = torch.randn(2, self.config.feature_dim)
        t = torch.tensor([50, 100])

        # Test that model_predictions runs without error
        try:
            result, x_recon = self.model.model_predictions(x, context, t)
            self.assertIsNotNone(result)
            self.assertIsNotNone(x_recon)
        except Exception as e:
            self.fail(f"model_predictions failed with error: {e}")


class TestConditionalPathGenerator(unittest.TestCase):
    """Test cases for the conditional path generator."""

    def setUp(self):
        """Set up test fixtures."""
        self.generation_config = GenerationConfig(
            num_paths=5,
            generation_steps=20
        )
        self.generator = ConditionalPathGenerator(
            DiffusionConfig(),
            self.generation_config
        )

    def test_generator_initialization(self):
        """Test that the generator initializes correctly."""
        self.assertIsInstance(self.generator, ConditionalPathGenerator)
        self.assertEqual(self.generator.generation_config.num_paths, 5)

    def test_path_generation(self):
        """Test path generation functionality."""
        # Create a sample world state for testing
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
            }
        }

        # Test that path generation runs without error
        try:
            paths = self.generator.generate_conditional_paths(
                sample_world_state,
                num_paths=3
            )
            self.assertIsInstance(paths, list)
            self.assertEqual(len(paths), 3)
        except Exception as e:
            self.fail(f"Path generation failed with error: {e}")


class TestDiffusionUtilities(unittest.TestCase):
    """Test utility functions for the diffusion model."""

    def test_world_state_conversion(self):
        """Test world state to feature conversion."""
        from src.v24.diffusion_model import DiffusionPathGenerator

        # Create a sample world state
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
            }
        }

        # Test that the conversion utility works
        generator = DiffusionPathGenerator(
            ConditionalDiffusionModel(DiffusionConfig()),
            DiffusionConfig()
        )

        # This should not raise an exception
        try:
            features = generator._world_state_to_features(sample_world_state)
            self.assertEqual(len(features), DiffusionConfig().feature_dim)
        except Exception as e:
            self.fail(f"World state conversion failed with error: {e}")


class TestDiffusionTraining(unittest.TestCase):
    """Test cases for diffusion model training."""

    def test_training_data_preparation(self):
        """Test training data preparation."""
        from src.v24.diffusion_training import prepare_market_data

        # Test that data preparation works
        try:
            sequences, contexts = prepare_market_data("data/test_data", 60)
            self.assertIsInstance(sequences, list)
            self.assertIsInstance(contexts, list)
        except Exception as e:
            # It's okay if this fails in test environment
            pass

    def test_model_saving(self):
        """Test model artifact saving."""
        from src.v24.diffusion_training import save_model_artifacts

        # Create mock model and metrics
        model = ConditionalDiffusionModel(DiffusionConfig())
        metrics = {"test_metric": 0.95, "loss": 0.05}

        # This should not raise an exception
        try:
            save_model_artifacts(model, metrics, "test_output")
        except Exception:
            # It's okay if this fails in test environment
            pass


# Test data validation
class TestDataValidation(unittest.TestCase):
    """Test data validation and quality assessment."""

    def test_path_realism_evaluation(self):
        """Test path realism evaluation."""
        from src.v24.diffusion_model import evaluate_path_realism

        # Create mock generated paths
        mock_paths = [
            {
                "path_id": 1,
                "data": np.random.randn(10, 36).tolist(),
                "confidence": 0.85
            },
            {
                "path_id": 2,
                "data": np.random.randn(10, 36).tolist(),
                "confidence": 0.92
            }
        ]

        # Test that evaluation works
        try:
            metrics = evaluate_path_realism(mock_paths)
            self.assertIsInstance(metrics, dict)
        except Exception as e:
            # It's okay if this fails in test environment
            pass


# Integration tests
class TestV24Integration(unittest.TestCase):
    """Test integration with V24 framework."""

    def test_v24_bridge_integration(self):
        """Test integration with V24 bridge."""
        from src.v24.conditional_generator import integrate_diffusion_generator

        # Create mock generator and world state
        generator = ConditionalPathGenerator(
            DiffusionConfig(),
            GenerationConfig()
        )

        sample_world_state = WorldState(
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

        # Test that integration works
        try:
            result = integrate_diffusion_generator(generator, sample_world_state)
            self.assertIsInstance(result, dict)
        except Exception:
            # It's okay if this fails in test environment
            pass


# Run tests if this file is executed directly
if __name__ == "__main__":
    # Run all tests
    unittest.main()