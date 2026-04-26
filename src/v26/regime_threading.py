"""
V26 Phase 1, Agent 2: Regime Threading Controller

This module implements the regime-threaded path generation system that generates
paths using top-3 regime weighting to improve path diversity and regime coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Tuple, Union

import numpy as np
import torch
from torch import Tensor


# 9-dim regime vector indices (from v6/regime_detection.py)
REGIME_INDICES: Dict[str, int] = {
    "trend_up_strong": 0,
    "trend_up_weak": 1,
    "range": 2,
    "mean_reversion": 3,
    "breakout": 4,
    "panic_news_shock": 5,
    "trend_down_weak": 6,
    "trend_down_strong": 7,
    "low_volatility": 8,
}

# Reverse mapping
REGIME_NAMES: Dict[int, str] = {v: k for k, v in REGIME_INDICES.items()}


class RegimeConditionedGenerator(Protocol):
    """Protocol for regime-conditioned generators."""

    def generate_paths(
        self,
        world_state: Any,
        regime_probs: Tensor,
        num_paths: int,
        past_context: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Generate paths conditioned on regime probabilities.

        Args:
            world_state: Current market state
            regime_probs: One-hot or soft regime probability distribution (9,)
            num_paths: Number of paths to generate
            past_context: Optional past context for conditioning

        Returns:
            Generated paths tensor
        """
        ...


@dataclass(frozen=True)
class ThreadedPathResult:
    """Result container for regime-threaded path generation."""

    paths: Tensor  # Combined paths from all regimes
    regime_distribution: List[float]  # Original regime probability distribution
    top_3_regimes: List[Dict[str, Any]]  # Top 3 regimes with probabilities and counts
    paths_per_regime: Dict[str, int]  # Count of paths per regime name
    regime_indices: List[int]  # Indices of top 3 regimes
    path_weights: Dict[str, float]  # Weight factors for each regime (for downstream use)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


@dataclass
class RegimeThreadingConfig:
    """Configuration for regime threading controller."""

    num_paths: int = 20
    top_1_weight: float = 0.70
    top_2_weight: float = 0.20
    top_3_weight: float = 0.10
    min_paths_per_regime: int = 1
    device: str = "cpu"
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        # Validate weights sum to approximately 1.0
        total = self.top_1_weight + self.top_2_weight + self.top_3_weight
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"Regime weights must sum to 1.0, got {total}")


class RegimeThreadingController:
    """
    Regime Threading Controller for V26.

    Generates paths using top-3 regime weighting:
    - Top-1 regime: 70% of paths
    - Top-2 regime: 20% of paths
    - Top-3 regime: 10% of paths

    This approach ensures diverse path coverage across the most likely
    market regimes while maintaining focus on the dominant regime.
    """

    def __init__(
        self,
        generator: RegimeConditionedGenerator,
        config: Optional[RegimeThreadingConfig] = None,
    ) -> None:
        """
        Initialize the regime threading controller.

        Args:
            generator: Regime-conditioned path generator
            config: Threading configuration (uses defaults if None)
        """
        self.generator = generator
        self.config = config or RegimeThreadingConfig()
        self.device = torch.device(self.config.device)

    def generate_regime_threaded_paths(
        self,
        world_state: Any,
        regime_probs: Tensor,
        past_context: Optional[Tensor] = None,
    ) -> ThreadedPathResult:
        """
        Generate regime-threaded paths using top-3 weighting.

        Args:
            world_state: Current market state (WorldState or compatible)
            regime_probs: 9-dim regime probability distribution from v6/regime_detection.py
                [trend_up_strong, trend_up_weak, range, mean_reversion, breakout,
                 panic_news_shock, trend_down_weak, trend_down_strong, low_volatility]
            past_context: Optional past context tensor for conditioning

        Returns:
            ThreadedPathResult containing combined paths and metadata
        """
        # Validate input
        if regime_probs.shape != (9,):
            raise ValueError(f"regime_probs must be shape (9,), got {regime_probs.shape}")

        # Ensure tensor is on correct device
        regime_probs = regime_probs.to(self.device)

        # 1. Get top-3 regimes
        top_k = torch.topk(regime_probs, k=3)
        top_indices = top_k.indices.tolist()
        top_values = top_k.values.tolist()

        # 2. Calculate path counts
        n1, n2, n3 = self._calculate_path_counts(self.config.num_paths)

        # 3. Generate paths for each regime
        all_paths: List[Tensor] = []
        paths_per_regime: Dict[str, int] = {}

        # Top-1 regime (70% of paths)
        if n1 > 0:
            regime1_one_hot = self._one_hot_encode(top_indices[0], 9)
            paths1 = self.generator.generate_paths(
                world_state,
                regime_probs=regime1_one_hot.to(self.device),
                num_paths=n1,
                past_context=past_context,
            )
            all_paths.append(paths1)
            regime1_name = REGIME_NAMES.get(top_indices[0], f"regime_{top_indices[0]}")
            paths_per_regime[regime1_name] = n1

        # Top-2 regime (20% of paths)
        if n2 > 0:
            regime2_one_hot = self._one_hot_encode(top_indices[1], 9)
            paths2 = self.generator.generate_paths(
                world_state,
                regime_probs=regime2_one_hot.to(self.device),
                num_paths=n2,
                past_context=past_context,
            )
            all_paths.append(paths2)
            regime2_name = REGIME_NAMES.get(top_indices[1], f"regime_{top_indices[1]}")
            paths_per_regime[regime2_name] = n2

        # Top-3 regime (10% of paths)
        if n3 > 0:
            regime3_one_hot = self._one_hot_encode(top_indices[2], 9)
            paths3 = self.generator.generate_paths(
                world_state,
                regime_probs=regime3_one_hot.to(self.device),
                num_paths=n3,
                past_context=past_context,
            )
            all_paths.append(paths3)
            regime3_name = REGIME_NAMES.get(top_indices[2], f"regime_{top_indices[2]}")
            paths_per_regime[regime3_name] = n3

        # 4. Combine paths
        combined_paths = torch.cat(all_paths, dim=0) if len(all_paths) > 0 else torch.tensor([])

        # 5. Prepare top-3 regime info
        top_3_regimes = []
        path_counts = [n1, n2, n3]
        for i, (idx, prob) in enumerate(zip(top_indices, top_values)):
            regime_name = REGIME_NAMES.get(idx, f"regime_{idx}")
            top_3_regimes.append({
                "rank": i + 1,
                "index": idx,
                "name": regime_name,
                "probability": round(prob, 6),
                "path_count": path_counts[i],
                "weight": [self.config.top_1_weight, self.config.top_2_weight, self.config.top_3_weight][i],
            })

        # 6. Calculate path weights (for downstream ensemble weighting)
        path_weights = self._get_path_weights(regime_probs)

        # 7. Return combined result
        return ThreadedPathResult(
            paths=combined_paths,
            regime_distribution=regime_probs.cpu().tolist(),
            top_3_regimes=top_3_regimes,
            paths_per_regime=paths_per_regime,
            regime_indices=top_indices,
            path_weights=path_weights,
            metadata={
                "total_paths": self.config.num_paths,
                "device": str(self.device),
                "dtype": str(self.config.dtype),
            },
        )

    def _calculate_path_counts(self, total_paths: int) -> Tuple[int, int, int]:
        """
        Calculate path counts for top-3 regimes.

        Args:
            total_paths: Total number of paths to generate

        Returns:
            Tuple of (n1, n2, n3) path counts for top-1, top-2, top-3 regimes
        """
        n1 = int(self.config.top_1_weight * total_paths)
        n2 = int(self.config.top_2_weight * total_paths)
        n3 = total_paths - n1 - n2

        # Ensure minimum paths per regime
        if n1 < self.config.min_paths_per_regime and n1 > 0:
            n1 = self.config.min_paths_per_regime
        if n2 < self.config.min_paths_per_regime and n2 > 0:
            n2 = self.config.min_paths_per_regime

        # Recalculate n3 to ensure total matches
        n3 = total_paths - n1 - n2

        return n1, n2, n3

    def _one_hot_encode(self, index: int, num_classes: int) -> Tensor:
        """
        Create one-hot encoded tensor.

        Args:
            index: Class index to set to 1
            num_classes: Total number of classes

        Returns:
            One-hot encoded tensor
        """
        tensor = torch.zeros(num_classes, dtype=self.config.dtype)
        tensor[index] = 1.0
        return tensor

    def _get_path_weights(self, regime_probs: Tensor) -> Dict[str, float]:
        """
        Calculate path weights for downstream ensemble weighting.

        Args:
            regime_probs: Regime probability distribution

        Returns:
            Dictionary mapping regime names to their weights
        """
        top_3 = torch.topk(regime_probs, k=3)
        weights = {}

        for i, idx in enumerate(top_3.indices.tolist()):
            regime_name = REGIME_NAMES.get(idx, f"regime_{idx}")
            base_weight = [self.config.top_1_weight, self.config.top_2_weight, self.config.top_3_weight][i]
            # Normalize by regime probability
            prob = regime_probs[idx].item()
            weights[regime_name] = base_weight * prob

        return weights


def regime_probs_to_embedding(regime_probs: Tensor) -> Tensor:
    """
    Convert regime probabilities to a regime embedding.

    Args:
        regime_probs: 9-dim regime probability distribution

    Returns:
        Regime embedding tensor
    """
    if regime_probs.shape != (9,):
        raise ValueError(f"regime_probs must be shape (9,), got {regime_probs.shape}")

    # Soft embedding: keep as probabilities (allows for mixed regimes)
    # This can be used for soft-conditioning in the generator
    return regime_probs


def get_path_weights(
    regime_probs: Tensor,
    top_1_weight: float = 0.70,
    top_2_weight: float = 0.20,
    top_3_weight: float = 0.10,
) -> Dict[str, int]:
    """
    Get path counts per regime for the top-3 regimes.

    This is a convenience function for calculating path distribution
    without instantiating the full controller.

    Args:
        regime_probs: 9-dim regime probability distribution
        top_1_weight: Weight for top-1 regime
        top_2_weight: Weight for top-2 regime
        top_3_weight: Weight for top-3 regime

    Returns:
        Dictionary mapping regime names to path counts
    """
    if regime_probs.shape != (9,):
        raise ValueError(f"regime_probs must be shape (9,), got {regime_probs.shape}")

    # Validate weights
    total = top_1_weight + top_2_weight + top_3_weight
    if not 0.99 <= total <= 1.01:
        raise ValueError(f"Weights must sum to 1.0, got {total}")

    # Get top-3
    top_k = torch.topk(regime_probs, k=3)
    top_indices = top_k.indices.tolist()

    # Default num_paths for calculation
    num_paths = 20
    n1 = int(top_1_weight * num_paths)
    n2 = int(top_2_weight * num_paths)
    n3 = num_paths - n1 - n2

    path_counts = [n1, n2, n3]
    weights = {}

    for i, idx in enumerate(top_indices):
        regime_name = REGIME_NAMES.get(idx, f"regime_{idx}")
        weights[regime_name] = path_counts[i]

    return weights


# Integration helper for V24/V25 compatibility
def create_regime_threading_controller(
    generator: RegimeConditionedGenerator,
    num_paths: int = 20,
    device: str = "cpu",
) -> RegimeThreadingController:
    """
    Factory function to create a RegimeThreadingController.

    Args:
        generator: Regime-conditioned generator instance
        num_paths: Number of paths to generate
        device: Device to use ("cpu" or "cuda")

    Returns:
        Configured RegimeThreadingController instance
    """
    config = RegimeThreadingConfig(num_paths=num_paths, device=device)
    return RegimeThreadingController(generator, config)


# Mock generator for testing
class MockRegimeConditionedGenerator:
    """Mock generator for testing the RegimeThreadingController."""

    def __init__(self, device: str = "cpu", path_length: int = 20):
        self.device = torch.device(device)
        self.path_length = path_length
        self.call_count = 0
        self.call_log: List[Dict[str, Any]] = []

    def generate_paths(
        self,
        world_state: Any,
        regime_probs: Tensor,
        num_paths: int,
        past_context: Optional[Tensor] = None,
    ) -> Tensor:
        """Generate mock paths."""
        self.call_count += 1

        # Log the call
        regime_idx = torch.argmax(regime_probs).item()
        self.call_log.append({
            "call_id": self.call_count,
            "num_paths": num_paths,
            "regime_idx": regime_idx,
            "regime_name": REGIME_NAMES.get(regime_idx, f"regime_{regime_idx}"),
            "world_state_type": type(world_state).__name__,
        })

        # Generate mock paths: (num_paths, path_length, 1)
        paths = torch.randn(
            num_paths,
            self.path_length,
            1,
            device=self.device,
            dtype=torch.float32,
        )

        # Add regime-specific bias to make paths distinguishable
        paths += regime_idx * 0.1

        return paths


def test_regime_threading_controller():
    """
    Test suite for RegimeThreadingController.

    Returns:
        Dict containing test results
    """
    print("=" * 60)
    print("RegimeThreadingController Test Suite")
    print("=" * 60)

    results = {
        "tests_run": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "errors": [],
    }

    # Test 1: Top-3 selection works correctly
    print("\n[Test 1] Verify top-3 selection works...")
    results["tests_run"] += 1
    try:
        mock_gen = MockRegimeConditionedGenerator()
        controller = RegimeThreadingController(mock_gen, RegimeThreadingConfig(num_paths=20))

        # Create regime probabilities with known ordering
        regime_probs = torch.tensor([
            0.40,  # trend_up_strong (0) - highest
            0.25,  # trend_up_weak (1) - second
            0.15,  # range (2) - third
            0.05,  # mean_reversion (3)
            0.03,  # breakout (4)
            0.04,  # panic_news_shock (5)
            0.03,  # trend_down_weak (6)
            0.02,  # trend_down_strong (7)
            0.03,  # low_volatility (8)
        ])

        result = controller.generate_regime_threaded_paths(
            world_state={"test": "data"},
            regime_probs=regime_probs,
        )

        expected_top_3 = [0, 1, 2]  # Indices of top 3
        actual_top_3 = result.regime_indices

        assert actual_top_3 == expected_top_3, f"Expected {expected_top_3}, got {actual_top_3}"
        print(f"  [PASS] Top-3 selection: {actual_top_3} (correct)")
        results["tests_passed"] += 1
    except Exception as e:
        results["tests_failed"] += 1
        results["errors"].append(f"Test 1 failed: {e}")
        print(f"  [FAIL] Failed: {e}")

    # Test 2: Path counts sum correctly
    print("\n[Test 2] Verify path counts sum correctly...")
    results["tests_run"] += 1
    try:
        total_paths = sum(result.paths_per_regime.values())
        expected_total = 20

        assert total_paths == expected_total, f"Expected {expected_total}, got {total_paths}"

        # Verify 70/20/10 split
        top_1_count = result.paths_per_regime.get("trend_up_strong", 0)
        top_2_count = result.paths_per_regime.get("trend_up_weak", 0)
        top_3_count = result.paths_per_regime.get("range", 0)

        expected_n1 = int(0.70 * 20)  # 14
        expected_n2 = int(0.20 * 20)  # 4
        expected_n3 = 20 - expected_n1 - expected_n2  # 2

        assert top_1_count == expected_n1, f"Top-1 count: expected {expected_n1}, got {top_1_count}"
        assert top_2_count == expected_n2, f"Top-2 count: expected {expected_n2}, got {top_2_count}"
        assert top_3_count == expected_n3, f"Top-3 count: expected {expected_n3}, got {top_3_count}"

        print(f"  [PASS] Path counts: n1={top_1_count}, n2={top_2_count}, n3={top_3_count} (total={total_paths})")
        results["tests_passed"] += 1
    except Exception as e:
        results["tests_failed"] += 1
        results["errors"].append(f"Test 2 failed: {e}")
        print(f"  [FAIL] Failed: {e}")

    # Test 3: Regime metadata is attached
    print("\n[Test 3] Verify regime metadata is attached...")
    results["tests_run"] += 1
    try:
        assert len(result.regime_distribution) == 9, "Regime distribution should have 9 elements"
        assert len(result.top_3_regimes) == 3, "Top-3 regimes should have 3 entries"
        assert len(result.paths_per_regime) == 3, "Paths per regime should have 3 entries"
        assert len(result.regime_indices) == 3, "Regime indices should have 3 entries"

        # Check metadata structure
        for regime_info in result.top_3_regimes:
            assert "rank" in regime_info, "Regime info should have rank"
            assert "index" in regime_info, "Regime info should have index"
            assert "name" in regime_info, "Regime info should have name"
            assert "probability" in regime_info, "Regime info should have probability"
            assert "path_count" in regime_info, "Regime info should have path_count"
            assert "weight" in regime_info, "Regime info should have weight"

        print(f"  [PASS] Metadata attached: {len(result.top_3_regimes)} regimes with full info")
        results["tests_passed"] += 1
    except Exception as e:
        results["tests_failed"] += 1
        results["errors"].append(f"Test 3 failed: {e}")
        print(f"  [FAIL] Failed: {e}")

    # Test 4: Generator called for each regime
    print("\n[Test 4] Verify generator called for each regime...")
    results["tests_run"] += 1
    try:
        assert mock_gen.call_count == 3, f"Generator should be called 3 times, got {mock_gen.call_count}"

        # Verify call log
        call_regimes = [call["regime_idx"] for call in mock_gen.call_log]
        assert call_regimes == [0, 1, 2], f"Expected calls for regimes [0, 1, 2], got {call_regimes}"

        print(f"  [PASS] Generator called {mock_gen.call_count} times for top-3 regimes")
        results["tests_passed"] += 1
    except Exception as e:
        results["tests_failed"] += 1
        results["errors"].append(f"Test 4 failed: {e}")
        print(f"  [FAIL] Failed: {e}")

    # Test 5: Path weights calculation
    print("\n[Test 5] Verify path weights calculation...")
    results["tests_run"] += 1
    try:
        weights = controller._get_path_weights(regime_probs)
        assert len(weights) == 3, f"Should have 3 path weights, got {len(weights)}"

        # Check that weights contains expected regimes
        assert "trend_up_strong" in weights, "Should have trend_up_strong"
        assert "trend_up_weak" in weights, "Should have trend_up_weak"
        assert "range" in weights, "Should have range"

        print(f"  [PASS] Path weights: {weights}")
        results["tests_passed"] += 1
    except Exception as e:
        results["tests_failed"] += 1
        results["errors"].append(f"Test 5 failed: {e}")
        print(f"  [FAIL] Failed: {e}")

    # Test 6: Different num_paths configuration
    print("\n[Test 6] Verify different num_paths configuration...")
    results["tests_run"] += 1
    try:
        mock_gen2 = MockRegimeConditionedGenerator()
        controller2 = RegimeThreadingController(mock_gen2, RegimeThreadingConfig(num_paths=50))

        result2 = controller2.generate_regime_threaded_paths(
            world_state={"test": "data"},
            regime_probs=regime_probs,
        )

        total = sum(result2.paths_per_regime.values())
        assert total == 50, f"Expected 50 paths, got {total}"

        expected_n1 = int(0.70 * 50)  # 35
        expected_n2 = int(0.20 * 50)  # 10
        expected_n3 = 50 - expected_n1 - expected_n2  # 5

        top_1_count = result2.paths_per_regime.get("trend_up_strong", 0)
        top_2_count = result2.paths_per_regime.get("trend_up_weak", 0)
        top_3_count = result2.paths_per_regime.get("range", 0)

        assert top_1_count == expected_n1, f"Top-1 count: expected {expected_n1}, got {top_1_count}"
        assert top_2_count == expected_n2, f"Top-2 count: expected {expected_n2}, got {top_2_count}"
        assert top_3_count == expected_n3, f"Top-3 count: expected {expected_n3}, got {top_3_count}"

        print(f"  [PASS] num_paths=50: n1={top_1_count}, n2={top_2_count}, n3={top_3_count}")
        results["tests_passed"] += 1
    except Exception as e:
        results["tests_failed"] += 1
        results["errors"].append(f"Test 6 failed: {e}")
        print(f"  [FAIL] Failed: {e}")

    # Test 7: Invalid regime probability shape
    print("\n[Test 7] Verify invalid regime probability shape handling...")
    results["tests_run"] += 1
    try:
        mock_gen3 = MockRegimeConditionedGenerator()
        controller3 = RegimeThreadingController(mock_gen3)

        invalid_probs = torch.tensor([0.5, 0.5])  # Wrong shape

        try:
            controller3.generate_regime_threaded_paths(
                world_state={"test": "data"},
                regime_probs=invalid_probs,
            )
            results["tests_failed"] += 1
            results["errors"].append("Test 7 failed: Should have raised ValueError")
            print(f"  [FAIL] Failed: Should have raised ValueError for invalid shape")
        except ValueError as e:
            print(f"  [PASS] Correctly raised ValueError for invalid shape: {e}")
            results["tests_passed"] += 1
    except Exception as e:
        results["tests_failed"] += 1
        results["errors"].append(f"Test 7 failed: {e}")
        print(f"  [FAIL] Failed: {e}")

    # Test 8: regime_probs_to_embedding helper function
    print("\n[Test 8] Verify regime_probs_to_embedding helper...")
    results["tests_run"] += 1
    try:
        probs = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2])
        embedding = regime_probs_to_embedding(probs)

        assert torch.allclose(embedding, probs), "Embedding should equal input probabilities"
        assert embedding.shape == (9,), "Embedding should have shape (9,)"

        print(f"  [PASS] Embedding shape: {embedding.shape}, preserves soft probabilities")
        results["tests_passed"] += 1
    except Exception as e:
        results["tests_failed"] += 1
        results["errors"].append(f"Test 8 failed: {e}")
        print(f"  [FAIL] Failed: {e}")

    # Test 9: get_path_weights helper function
    print("\n[Test 9] Verify get_path_weights helper...")
    results["tests_run"] += 1
    try:
        probs = torch.tensor([0.40, 0.25, 0.15, 0.05, 0.03, 0.04, 0.03, 0.02, 0.03])
        path_counts = get_path_weights(probs)

        assert len(path_counts) == 3, f"Should return 3 regimes, got {len(path_counts)}"

        total = sum(path_counts.values())
        expected = 20  # Default num_paths
        assert total == expected, f"Path counts should sum to {expected}, got {total}"

        # Verify 70/20/10 split (default)
        assert path_counts["trend_up_strong"] == 14, f"Expected 14 for top-1, got {path_counts['trend_up_strong']}"
        assert path_counts["trend_up_weak"] == 4, f"Expected 4 for top-2, got {path_counts['trend_up_weak']}"
        assert path_counts["range"] == 2, f"Expected 2 for top-3, got {path_counts['range']}"

        print(f"  [PASS] Path weights helper: {path_counts}")
        results["tests_passed"] += 1
    except Exception as e:
        results["tests_failed"] += 1
        results["errors"].append(f"Test 9 failed: {e}")
        print(f"  [FAIL] Failed: {e}")

    # Test 10: Config validation
    print("\n[Test 10] Verify config weight validation...")
    results["tests_run"] += 1
    try:
        try:
            # Weights that don't sum to 1.0 should raise error
            RegimeThreadingConfig(top_1_weight=0.5, top_2_weight=0.3, top_3_weight=0.1)
            results["tests_failed"] += 1
            results["errors"].append("Test 10 failed: Should have raised ValueError")
            print(f"  [FAIL] Failed: Should have raised ValueError for invalid weights")
        except ValueError as e:
            print(f"  [PASS] Correctly raised ValueError for invalid weights: {e}")
            results["tests_passed"] += 1
    except Exception as e:
        results["tests_failed"] += 1
        results["errors"].append(f"Test 10 failed: {e}")
        print(f"  [FAIL] Failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Tests Run:    {results['tests_run']}")
    print(f"Tests Passed: {results['tests_passed']}")
    print(f"Tests Failed: {results['tests_failed']}")

    if results["errors"]:
        print("\nErrors:")
        for error in results["errors"]:
            print(f"  - {error}")

    print("=" * 60)

    return results


if __name__ == "__main__":
    # Run the test suite
    test_results = test_regime_threading_controller()

    # Exit with appropriate code
    import sys
    sys.exit(0 if test_results["tests_failed"] == 0 else 1)
