"""
V24 CABR (Confidence-Aware Branch Ranking) Implementation

This module implements the Confidence-Aware Branch Ranking system for V24 Phase 4.
The system ranks generated market branches based on confidence metrics and quality scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn

from src.v24.world_state import WorldState


@dataclass
class CABRConfig:
    """Configuration for the CABR system."""
    confidence_threshold: float = 0.65
    diversity_weight: float = 0.3
    quality_weight: float = 0.7
    min_branches: int = 5
    max_branches: int = 20
    uncertainty_penalty: float = 0.2


@dataclass
class MarketBranch:
    """Represents a single market branch/path prediction."""
    branch_id: str
    path: np.ndarray  # Shape: (time_steps, features)
    confidence: float
    quality_score: float
    uncertainty: float
    timestamp: str
    metadata: Dict[str, Any]


class CABRRanker(nn.Module):
    """
    Confidence-Aware Branch Ranking system for V24.

    This system ranks market branches based on confidence metrics and quality scores
    to determine the most promising trading opportunities.
    """

    def __init__(self, config: Optional[CABRConfig] = None) -> None:
        super().__init__()
        self.config = config or CABRConfig()

        # Scoring components
        self.confidence_weight = nn.Parameter(torch.tensor(0.5))
        self.diversity_weight = nn.Parameter(torch.tensor(0.3))
        self.quality_weight = nn.Parameter(torch.tensor(0.7))

    def rank_branches(
        self,
        branches: List[MarketBranch],
        world_state: WorldState
    ) -> List[MarketBranch]:
        """
        Rank branches based on confidence and quality metrics.

        Args:
            branches: List of market branches to rank
            world_state: Current market state for context

        Returns:
            Ranked list of branches
        """
        if not branches:
            return []

        # Calculate scores for each branch
        scored_branches = []
        for branch in branches:
            score = self._calculate_branch_score(branch, world_state)
            scored_branches.append((branch, score))

        # Sort by score (highest first)
        scored_branches.sort(key=lambda x: x[1], reverse=True)

        # Return top branches
        return [branch for branch, _ in scored_branches[:self.config.max_branches]]

    def _calculate_branch_score(
        self,
        branch: MarketBranch,
        world_state: WorldState
    ) -> float:
        """
        Calculate comprehensive score for a branch.

        Args:
            branch: Market branch to score
            world_state: Current market state

        Returns:
            Combined score for the branch
        """
        # Base confidence score
        confidence_component = branch.confidence * (1.0 - branch.uncertainty)

        # Quality component
        quality_component = branch.quality_score

        # Diversity component (based on how different this branch is from others)
        diversity_component = self._calculate_diversity_score(branch)

        # Market regime adjustment
        regime_adjustment = self._calculate_regime_adjustment(world_state)

        # Combine components
        combined_score = (
            self.confidence_weight * confidence_component +
            self.quality_weight * quality_component +
            self.diversity_weight * diversity_component
        ) * regime_adjustment

        return float(combined_score)

    def _calculate_diversity_score(self, branch: MarketBranch) -> float:
        """
        Calculate diversity score based on how unique this branch is.

        Args:
            branch: Market branch to evaluate

        Returns:
            Diversity score (0.0 to 1.0)
        """
        # In a real implementation, this would compare the branch
        # to other branches to measure uniqueness
        # For now, we'll use a simple heuristic
        path_variance = np.var(branch.path, axis=0).mean() if branch.path.size > 0 else 0.0
        return float(np.clip(path_variance * 10.0, 0.0, 1.0))

    def _calculate_regime_adjustment(self, world_state: WorldState) -> float:
        """
        Calculate regime-based adjustment factor.

        Args:
            world_state: Current market state

        Returns:
            Adjustment factor based on market regime
        """
        features = world_state.to_flat_features()
        regime_class = int(features.get("quant_models_macro_vol_regime_class", 0))

        # Adjust based on regime (e.g., higher volatility regimes get penalty)
        if regime_class >= 3:  # High volatility regime
            return 0.8
        elif regime_class >= 2:  # Medium volatility regime
            return 0.9
        else:  # Low volatility regime
            return 1.0

    def select_best_branches(
        self,
        branches: List[MarketBranch],
        world_state: WorldState,
        num_branches: int = 5
    ) -> List[MarketBranch]:
        """
        Select the best branches based on ranking.

        Args:
            branches: List of branches to select from
            world_state: Current market state
            num_branches: Number of branches to select

        Returns:
            Top N branches
        """
        ranked_branches = self.rank_branches(branches, world_state)
        return ranked_branches[:num_branches]


class CABREnsemble:
    """
    Ensemble of CABR rankers for robust branch selection.
    """

    def __init__(self, configs: List[CABRConfig]) -> None:
        self.rankers = [CABRRanker(config) for config in configs]

    def ensemble_rank(
        self,
        branches: List[MarketBranch],
        world_state: WorldState
    ) -> List[MarketBranch]:
        """
        Use ensemble approach to rank branches.

        Args:
            branches: List of branches to rank
            world_state: Current market state

        Returns:
            Ensemble-ranked branches
        """
        # Get rankings from all rankers
        all_rankings = []
        for ranker in self.rankers:
            ranked = ranker.rank_branches(branches, world_state)
            all_rankings.append(ranked)

        # Combine rankings (simple average for now)
        return self._combine_rankings(all_rankings)

    def _combine_rankings(
        self,
        rankings: List[List[MarketBranch]]
    ) -> List[MarketBranch]:
        """
        Combine multiple rankings into a single ranking.

        Args:
            rankings: List of rankings from different rankers

        Returns:
            Combined ranking
        """
        if not rankings:
            return []

        # Simple voting approach
        branch_scores: Dict[str, float] = {}
        branch_count = {}

        for ranking in rankings:
            for i, branch in enumerate(ranking):
                score = len(ranking) - i  # Higher rank = higher score
                branch_id = branch.branch_id
                if branch_id not in branch_scores:
                    branch_scores[branch_id] = 0
                    branch_count[branch_id] = 0
                branch_scores[branch_id] += score
                branch_count[branch_id] = branch_count.get(branch_id, 0) + 1

        # Sort by average score
        sorted_branches = sorted(
            branch_scores.items(),
            key=lambda x: x[1] / branch_count[x[0]],
            reverse=True
        )

        # Return branches in order (this would need mapping back to actual branch objects)
        return [branch for branch, _ in sorted_branches]


def create_cabr_system(config: Optional[CABRConfig] = None) -> CABRRanker:
    """
    Factory function to create a CABR system.

    Args:
        config: Configuration for the CABR system

    Returns:
        Configured CABR ranker
    """
    return CABRRanker(config)


def evaluate_cabr_performance(
    ranker: CABRRanker,
    test_branches: List[MarketBranch],
    world_state: WorldState
) -> Dict[str, float]:
    """
    Evaluate CABR system performance.

    Args:
        ranker: CABR ranker to evaluate
        test_branches: Test branches
        world_state: Current market state

    Returns:
        Performance metrics
    """
    # Rank the branches
    ranked_branches = ranker.rank_branches(test_branches, world_state)

    # Calculate metrics
    metrics = {
        "branches_ranked": len(ranked_branches),
        "avg_confidence": float(np.mean([b.confidence for b in ranked_branches])) if ranked_branches else 0.0,
        "avg_quality": float(np.mean([b.quality_score for b in ranked_branches])) if ranked_branches else 0.0,
        "confidence_std": float(np.std([b.confidence for b in ranked_branches])) if ranked_branches else 0.0,
    }

    return metrics


# Example usage
def main():
    """Example usage of the CABR system."""
    # Create configuration
    config = CABRConfig()

    # Create CABR system
    cabr = CABRRanker(config)

    # Create sample branches (in practice, these would come from the diffusion model)
    sample_branches = [
        MarketBranch(
            branch_id=f"branch_{i}",
            path=np.random.randn(30, 36),  # 30 time steps, 36 features
            confidence=0.7 + 0.1 * i,
            quality_score=0.5 + 0.1 * i,
            uncertainty=0.1 * i,
            timestamp="2026-04-12T10:00:00Z",
            metadata={"source": "diffusion_model"}
        )
        for i in range(5)
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

    # Rank branches
    ranked_branches = cabr.rank_branches(sample_branches, world_state)

    print(f"Ranked {len(ranked_branches)} branches")

    # Evaluate performance
    metrics = evaluate_cabr_performance(cabr, sample_branches, world_state)
    print(f"Performance metrics: {metrics}")


if __name__ == "__main__":
    main()