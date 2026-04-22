"""Unit tests for V26 Regime Diffusion components.

Tests cover:
- RegimeEmbedding forward pass
- RegimeGenerator.generate_paths with regime_probs
- RegimeThreadingController path distribution (70/20/10)
- RegimeDataset returns correct labels

Usage:
    python -m pytest tests/test_v26_regime_diffusion.py -v
    python tests/test_v26_regime_diffusion.py  # Run standalone
"""

import sys
import unittest
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v26.diffusion.regime_embedding import RegimeEmbedding


class MockRegimeGenerator:
    """Mock regime-conditioned generator for testing."""

    def __init__(self, num_regimes: int = 9, seq_len: int = 120):
        self.num_regimes = num_regimes
        self.seq_len = seq_len
        self.regime_embedding = RegimeEmbedding(
            num_regimes=num_regimes,
            embed_dim=16,
            use_learned_embedding=True
        )

    def generate_paths(
        self,
        context: Dict[str, np.ndarray],
        regime_probs: np.ndarray,
        num_paths: int = 10,
        steps: int = 50
    ) -> List[Dict]:
        """Generate paths conditioned on regime probabilities.

        Args:
            context: Dictionary with context features
            regime_probs: (num_regimes,) probability vector
            num_paths: Number of paths to generate
            steps: Number of diffusion steps

        Returns:
            List of path dictionaries with regime conditioning
        """
        paths = []

        # Convert regime probs to embedding
        regime_tensor = torch.tensor(regime_probs, dtype=torch.float32)
        regime_emb = self.regime_embedding(regime_tensor).detach().numpy()

        for i in range(num_paths):
            # Generate path with regime influence
            base_noise = np.random.randn(self.seq_len, 144)

            # Regime affects the drift and volatility
            regime_idx = np.argmax(regime_probs)

            # Different regimes produce different characteristics
            if regime_idx < 4:  # Trending regimes
                drift = 0.01 * (regime_idx - 1.5)  # Positive for up, negative for down
                momentum = np.cumsum(np.random.randn(self.seq_len) * 0.1 + drift)
            elif regime_idx == 4:  # Mean reversion
                momentum = np.zeros(self.seq_len)
                for t in range(1, self.seq_len):
                    momentum[t] = -0.5 * momentum[t-1] + np.random.randn() * 0.2
            elif regime_idx == 5:  # Breakout
                momentum = np.random.randn(self.seq_len) * 1.5
            elif regime_idx == 6:  # Low volatility
                momentum = np.random.randn(self.seq_len) * 0.3
            elif regime_idx == 7:  # High volatility
                momentum = np.random.randn(self.seq_len) * 2.0
            else:  # Neutral
                momentum = np.random.randn(self.seq_len) * 0.8

            path_data = base_noise + momentum[:, None] * 0.1

            paths.append({
                'path_id': f'path_{i}',
                'data': path_data,
                'regime_embedding': regime_emb.copy(),
                'regime_idx': regime_idx,
                'confidence': 0.5 + 0.4 * regime_probs[regime_idx],
                'metadata': {
                    'regime_probs': regime_probs.copy(),
                    'num_steps': steps,
                }
            })

        return paths


class MockRegimeThreadingController:
    """Mock threading controller that manages path distribution across regimes."""

    def __init__(
        self,
        primary_weight: float = 0.7,
        secondary_weight: float = 0.2,
        tertiary_weight: float = 0.1
    ):
        self.primary_weight = primary_weight
        self.secondary_weight = secondary_weight
        self.tertiary_weight = tertiary_weight

    def get_path_distribution(
        self,
        regime_probs: np.ndarray,
        total_paths: int = 100
    ) -> Dict[int, int]:
        """Determine how many paths to generate per regime.

        Args:
            regime_probs: (num_regimes,) probability vector
            total_paths: Total number of paths to generate

        Returns:
            Dictionary mapping regime index to path count
        """
        # Sort regimes by probability
        sorted_indices = np.argsort(regime_probs)[::-1]

        distribution = {}

        # Primary regime (highest prob)
        primary_idx = sorted_indices[0]
        distribution[primary_idx] = int(total_paths * self.primary_weight)

        # Secondary regime
        if len(regime_probs) > 1:
            secondary_idx = sorted_indices[1]
            distribution[secondary_idx] = int(total_paths * self.secondary_weight)

        # Tertiary regime
        if len(regime_probs) > 2:
            tertiary_idx = sorted_indices[2]
            distribution[tertiary_idx] = int(total_paths * self.tertiary_weight)

        # Remaining paths to primary
        assigned = sum(distribution.values())
        remaining = total_paths - assigned
        if remaining > 0:
            distribution[primary_idx] += remaining

        return distribution


class MockRegimeDataset:
    """Mock dataset that returns regime labels."""

    REGIME_NAMES = [
        'trend_up_strong', 'trend_up_weak',
        'trend_down_strong', 'trend_down_weak',
        'mean_reversion', 'breakout',
        'low_volatility', 'high_volatility',
        'neutral'
    ]

    def __init__(self, num_samples: int = 1000, seq_len: int = 120):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_regimes = len(self.REGIME_NAMES)

        # Generate synthetic data
        np.random.seed(42)
        self.data = np.random.randn(num_samples, seq_len, 144).astype(np.float32)

        # Generate regime labels (one-hot and class indices)
        self.regime_labels = np.random.randint(0, self.num_regimes, size=num_samples)
        self.regime_probs = np.zeros((num_samples, self.num_regimes), dtype=np.float32)
        for i in range(num_samples):
            # Create realistic regime distribution (not uniform)
            probs = np.random.dirichlet(np.ones(self.num_regimes) * 0.5)
            # Make the labeled regime have highest probability
            probs[self.regime_labels[i]] *= 2.0
            probs = probs / probs.sum()
            self.regime_probs[i] = probs

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int, str]:
        """Get a single sample.

        Returns:
            Tuple of (data, regime_probs, regime_class, regime_name)
        """
        return (
            torch.from_numpy(self.data[idx]),
            torch.from_numpy(self.regime_probs[idx]),
            int(self.regime_labels[idx]),
            self.REGIME_NAMES[self.regime_labels[idx]]
        )

    def get_regime_distribution(self) -> Dict[str, int]:
        """Get the distribution of regimes in the dataset."""
        unique, counts = np.unique(self.regime_labels, return_counts=True)
        return {self.REGIME_NAMES[i]: int(c) for i, c in zip(unique, counts)}


class TestRegimeEmbedding(unittest.TestCase):
    """Tests for RegimeEmbedding module."""

    def setUp(self):
        self.embedding = RegimeEmbedding(
            num_regimes=9,
            embed_dim=16,
            use_learned_embedding=False
        )
        self.embedding_with_learned = RegimeEmbedding(
            num_regimes=9,
            embed_dim=16,
            use_learned_embedding=True
        )

    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape."""
        # Batch input
        regime_probs = torch.randn(4, 9).softmax(dim=1)
        output = self.embedding(regime_probs)
        self.assertEqual(output.shape, (4, 16))

        # Single input
        single_probs = torch.randn(9).softmax(dim=0)
        output_single = self.embedding(single_probs)
        self.assertEqual(output_single.shape, (16,))

    def test_forward_pass_values(self):
        """Test that output values are in reasonable range."""
        regime_probs = torch.randn(8, 9).softmax(dim=1)
        output = self.embedding(regime_probs)

        # Check no NaN or Inf
        self.assertTrue(torch.all(torch.isfinite(output)))

        # Check reasonable magnitude (GELU should keep values bounded)
        self.assertTrue(torch.all(output.abs() < 10))

    def test_learned_embedding_addition(self):
        """Test that learned embedding is added when enabled."""
        regime_probs = torch.randn(4, 9).softmax(dim=1)

        # Without learned embedding
        output_basic = self.embedding(regime_probs)

        # With learned embedding
        output_learned = self.embedding_with_learned(regime_probs)

        # They should be different
        self.assertFalse(torch.allclose(output_basic, output_learned))

    def test_forward_from_class(self):
        """Test alternative forward from class indices."""
        # Single class
        class_idx = torch.tensor(3)
        output = self.embedding.forward_from_class(class_idx)
        self.assertEqual(output.shape, (16,))

        # Batch of classes
        class_batch = torch.tensor([0, 3, 5, 7])
        output_batch = self.embedding.forward_from_class(class_batch)
        self.assertEqual(output_batch.shape, (4, 16))

    def test_backward_pass(self):
        """Test that gradients flow through the embedding."""
        regime_probs = torch.randn(4, 9, requires_grad=True).softmax(dim=1)
        output = self.embedding(regime_probs)
        loss = output.mean()
        loss.backward()

        # Check gradients exist
        self.assertIsNotNone(regime_probs.grad)
        self.assertTrue(torch.all(torch.isfinite(regime_probs.grad)))

    def test_deterministic_output(self):
        """Test that same input produces same output (eval mode)."""
        self.embedding.eval()
        regime_probs = torch.randn(4, 9).softmax(dim=1)

        with torch.no_grad():
            output1 = self.embedding(regime_probs)
            output2 = self.embedding(regime_probs)

        self.assertTrue(torch.allclose(output1, output2, atol=1e-5))

    def test_probability_sensitivity(self):
        """Test that different probabilities produce different embeddings."""
        probs1 = torch.tensor([[0.9, 0.1, 0, 0, 0, 0, 0, 0, 0]])
        probs2 = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0.1, 0.9]])

        self.embedding.eval()
        with torch.no_grad():
            emb1 = self.embedding(probs1)
            emb2 = self.embedding(probs2)

        # Embeddings should be different
        self.assertFalse(torch.allclose(emb1, emb2, atol=1e-3))


class TestRegimeGenerator(unittest.TestCase):
    """Tests for RegimeGenerator path generation."""

    def setUp(self):
        self.generator = MockRegimeGenerator(num_regimes=9, seq_len=120)
        self.context = {'ctx': np.random.randn(144)}

    def test_generate_paths_count(self):
        """Test that requested number of paths is generated."""
        regime_probs = np.array([0.7, 0.2, 0.1, 0, 0, 0, 0, 0, 0])
        paths = self.generator.generate_paths(
            self.context, regime_probs, num_paths=5, steps=50
        )
        self.assertEqual(len(paths), 5)

    def test_generate_paths_structure(self):
        """Test that generated paths have required structure."""
        regime_probs = np.array([0.5, 0.3, 0.2, 0, 0, 0, 0, 0, 0])
        paths = self.generator.generate_paths(
            self.context, regime_probs, num_paths=3
        )

        for path in paths:
            self.assertIn('path_id', path)
            self.assertIn('data', path)
            self.assertIn('regime_embedding', path)
            self.assertIn('regime_idx', path)
            self.assertIn('confidence', path)
            self.assertIn('metadata', path)

    def test_generate_paths_regime_conditioning(self):
        """Test that different regimes produce different path characteristics."""
        # Trend up regime
        trend_probs = np.array([0.9, 0, 0, 0, 0, 0, 0, 0, 0])
        trend_paths = self.generator.generate_paths(
            self.context, trend_probs, num_paths=10
        )
        trend_returns = np.mean([p['data'][:, 0].mean() for p in trend_paths])

        # Trend down regime
        down_probs = np.array([0, 0, 0.9, 0, 0, 0, 0, 0, 0])
        down_paths = self.generator.generate_paths(
            self.context, down_probs, num_paths=10
        )
        down_returns = np.mean([p['data'][:, 0].mean() for p in down_paths])

        # Trends should be in opposite directions
        self.assertGreater(trend_returns, down_returns)

    def test_generate_paths_high_volatility(self):
        """Test that high volatility regime produces higher variance paths."""
        high_vol_probs = np.array([0, 0, 0, 0, 0, 0, 0, 0.9, 0])
        high_vol_paths = self.generator.generate_paths(
            self.context, high_vol_probs, num_paths=20
        )
        high_vol_std = np.std([p['data'].std() for p in high_vol_paths])

        low_vol_probs = np.array([0, 0, 0, 0, 0, 0, 0.9, 0, 0])
        low_vol_paths = self.generator.generate_paths(
            self.context, low_vol_probs, num_paths=20
        )
        low_vol_std = np.std([p['data'].std() for p in low_vol_paths])

        self.assertGreater(high_vol_std, low_vol_std)

    def test_generate_paths_embedding_shape(self):
        """Test that regime embedding is correctly shaped in output."""
        regime_probs = np.array([0.6, 0.4, 0, 0, 0, 0, 0, 0, 0])
        paths = self.generator.generate_paths(
            self.context, regime_probs, num_paths=2
        )

        for path in paths:
            self.assertEqual(path['regime_embedding'].shape, (16,))


class TestRegimeThreadingController(unittest.TestCase):
    """Tests for RegimeThreadingController path distribution."""

    def setUp(self):
        self.controller = MockRegimeThreadingController(
            primary_weight=0.7,
            secondary_weight=0.2,
            tertiary_weight=0.1
        )

    def test_path_distribution_weights(self):
        """Test that distribution follows 70/20/10 split."""
        # Create regime probs where regime 0 is primary, 1 secondary, 2 tertiary
        regime_probs = np.array([0.6, 0.3, 0.1, 0, 0, 0, 0, 0, 0])
        distribution = self.controller.get_path_distribution(regime_probs, total_paths=100)

        self.assertEqual(distribution[0], 70)  # Primary
        self.assertEqual(distribution[1], 20)  # Secondary
        self.assertEqual(distribution[2], 10)  # Tertiary

    def test_path_distribution_sum(self):
        """Test that distribution sums to total_paths."""
        regime_probs = np.random.dirichlet(np.ones(9))
        distribution = self.controller.get_path_distribution(regime_probs, total_paths=100)

        self.assertEqual(sum(distribution.values()), 100)

    def test_path_distribution_priority(self):
        """Test that higher probability regimes get more paths."""
        # Make regime 5 have highest probability
        regime_probs = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0, 0, 0])
        regime_probs = regime_probs / regime_probs.sum()
        distribution = self.controller.get_path_distribution(regime_probs, total_paths=100)

        # Regime 5 should have the most paths
        self.assertEqual(max(distribution, key=distribution.get), 5)

    def test_path_distribution_rounding(self):
        """Test that rounding errors are handled correctly."""
        regime_probs = np.array([0.5, 0.25, 0.25, 0, 0, 0, 0, 0, 0])
        distribution = self.controller.get_path_distribution(regime_probs, total_paths=10)

        # Should sum to 10 even with rounding
        self.assertEqual(sum(distribution.values()), 10)

    def test_path_distribution_different_sizes(self):
        """Test distribution with different total path counts."""
        regime_probs = np.array([0.7, 0.2, 0.1, 0, 0, 0, 0, 0, 0])

        for total in [10, 50, 100, 1000]:
            distribution = self.controller.get_path_distribution(regime_probs, total)
            self.assertEqual(sum(distribution.values()), total)

            # Primary should have most
            primary_count = distribution.get(0, 0)
            self.assertGreater(primary_count, total * 0.6)


class TestRegimeDataset(unittest.TestCase):
    """Tests for RegimeDataset."""

    def setUp(self):
        self.dataset = MockRegimeDataset(num_samples=100, seq_len=120)

    def test_dataset_length(self):
        """Test that dataset returns correct length."""
        self.assertEqual(len(self.dataset), 100)

    def test_getitem_returns_tuple(self):
        """Test that __getitem__ returns correct tuple structure."""
        data, regime_probs, regime_class, regime_name = self.dataset[0]

        self.assertIsInstance(data, torch.Tensor)
        self.assertIsInstance(regime_probs, torch.Tensor)
        self.assertIsInstance(regime_class, int)
        self.assertIsInstance(regime_name, str)

    def test_data_shape(self):
        """Test that data has correct shape."""
        data, _, _, _ = self.dataset[0]
        self.assertEqual(data.shape, (120, 144))

    def test_regime_probs_shape(self):
        """Test that regime_probs has correct shape."""
        _, regime_probs, _, _ = self.dataset[0]
        self.assertEqual(regime_probs.shape, (9,))

    def test_regime_probs_normalized(self):
        """Test that regime probabilities sum to 1."""
        _, regime_probs, _, _ = self.dataset[0]
        self.assertAlmostEqual(regime_probs.sum().item(), 1.0, places=5)

    def test_regime_probs_non_negative(self):
        """Test that regime probabilities are non-negative."""
        _, regime_probs, _, _ = self.dataset[0]
        self.assertTrue(torch.all(regime_probs >= 0))

    def test_regime_class_in_range(self):
        """Test that regime class index is in valid range."""
        _, _, regime_class, _ = self.dataset[0]
        self.assertGreaterEqual(regime_class, 0)
        self.assertLess(regime_class, 9)

    def test_regime_name_valid(self):
        """Test that regime name is from valid set."""
        _, _, _, regime_name = self.dataset[0]
        self.assertIn(regime_name, MockRegimeDataset.REGIME_NAMES)

    def test_regime_consistency(self):
        """Test that regime class and name are consistent."""
        _, regime_probs, regime_class, regime_name = self.dataset[0]

        # Regime with highest prob should match class
        max_prob_idx = regime_probs.argmax().item()
        self.assertEqual(max_prob_idx, regime_class)

        # Name should match class index
        expected_name = MockRegimeDataset.REGIME_NAMES[regime_class]
        self.assertEqual(regime_name, expected_name)

    def test_dataset_distribution(self):
        """Test that regime distribution is returned correctly."""
        dist = self.dataset.get_regime_distribution()

        self.assertEqual(len(dist), 9)
        self.assertEqual(sum(dist.values()), 100)

        for name in MockRegimeDataset.REGIME_NAMES:
            self.assertIn(name, dist)

    def test_deterministic_sampling(self):
        """Test that same index returns same data."""
        data1, probs1, class1, name1 = self.dataset[5]
        data2, probs2, class2, name2 = self.dataset[5]

        self.assertTrue(torch.allclose(data1, data2))
        self.assertTrue(torch.allclose(probs1, probs2))
        self.assertEqual(class1, class2)
        self.assertEqual(name1, name2)


class TestRegimeIntegration(unittest.TestCase):
    """Integration tests for regime components working together."""

    def test_end_to_end_pipeline(self):
        """Test full pipeline from dataset to generator output."""
        # Create dataset
        dataset = MockRegimeDataset(num_samples=10)

        # Get sample
        data, regime_probs, regime_class, regime_name = dataset[0]

        # Create generator
        generator = MockRegimeGenerator()

        # Generate paths
        context = {'ctx': data[0].numpy()}  # Use first time step as context
        paths = generator.generate_paths(
            context, regime_probs.numpy(), num_paths=5
        )

        # Verify output
        self.assertEqual(len(paths), 5)
        for path in paths:
            self.assertEqual(path['regime_idx'], regime_class)

    def test_threading_controller_with_generator(self):
        """Test threading controller feeding into generator."""
        dataset = MockRegimeDataset(num_samples=5)
        controller = MockRegimeThreadingController()
        generator = MockRegimeGenerator()

        for i in range(5):
            data, regime_probs, _, _ = dataset[i]
            context = {'ctx': data[0].numpy()}

            # Get distribution
            distribution = controller.get_path_distribution(
                regime_probs.numpy(), total_paths=20
            )

            # Generate paths for each regime
            all_paths = []
            for reg_idx, num_paths in distribution.items():
                # One-hot regime probs
                one_hot = np.zeros(9)
                one_hot[reg_idx] = 1.0
                paths = generator.generate_paths(context, one_hot, num_paths)
                all_paths.extend(paths)

            self.assertEqual(len(all_paths), 20)


def run_all_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for test_class in [
        TestRegimeEmbedding,
        TestRegimeGenerator,
        TestRegimeThreadingController,
        TestRegimeDataset,
        TestRegimeIntegration,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
