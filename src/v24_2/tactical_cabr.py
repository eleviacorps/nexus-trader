"""
V24.2 Tactical CABR System

This module implements the tactical Confidence-Aware Branch Ranking system.
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
from datetime import datetime


@dataclass
class TacticalCABR:
    """Tactical Confidence-Aware Branch Ranking system."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_tradeability = self.config.get('min_tradeability', 0.35)

    def calculate_tactical_tradeability(self,
                                      tactical_paths: List[Dict[str, Any]],
                                      strategic_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate tactical tradeability based on paths and strategic context.

        Args:
            tactical_paths: List of tactical paths
            strategic_state: Current strategic engine state

        Returns:
            Tradeability assessment results
        """
        if not tactical_paths:
            return {
                'best_short_branch': None,
                'dangerous_short_branch': None,
                'short_tradeability': 0.0,
                'should_trade': False,
                'reason': 'No tactical paths provided'
            }

        # Find best and dangerous branches
        best_branch = self._find_best_branch(tactical_paths)
        dangerous_branch = self._find_dangerous_branch(tactical_paths)

        # Calculate tradeability score
        short_tradeability = self._calculate_tradeability_score(
            best_branch, dangerous_branch, strategic_state
        )

        # Check if tactical trade is allowed
        should_trade = self._should_allow_tactical_trade(
            short_tradeability, strategic_state
        )

        return {
            'best_short_branch': best_branch,
            'dangerous_short_branch': dangerous_branch,
            'short_tradeability': short_tradeability,
            'should_trade': should_trade,
            'calculation_timestamp': datetime.now().isoformat()
        }

    def _find_best_branch(self, tactical_paths: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find the best tactical branch based on quality metrics."""
        if not tactical_paths:
            return {}

        # Sort paths by quality score
        sorted_paths = sorted(
            tactical_paths,
            key=lambda x: x.get('branch_probability', 0),
            reverse=True
        )

        return sorted_paths[0] if sorted_paths else {}

    def _find_dangerous_branch(self, tactical_paths: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find the most dangerous tactical branch."""
        if not tactical_paths:
            return {}

        # Sort paths by risk (lowest probability first)
        sorted_paths = sorted(
            tactical_paths,
            key=lambda x: x.get('branch_probability', 1.0)
        )

        return sorted_paths[0] if sorted_paths else {}

    def _calculate_tradeability_score(self,
                                     best_branch: Dict[str, Any],
                                     dangerous_branch: Dict[str, Any],
                                     strategic_state: Dict[str, Any]) -> float:
        """Calculate tactical tradeability score."""
        if not best_branch or not dangerous_branch:
            return 0.0

        best_score = best_branch.get('branch_probability', 0.5)
        dangerous_score = dangerous_branch.get('branch_probability', 0.3)

        # Calculate tradeability as difference between best and dangerous
        tradeability = best_score - dangerous_score

        # Adjust based on strategic alignment
        strategic_direction = strategic_state.get('direction', 0)
        tactical_direction = best_branch.get('expected_direction', 0)

        # Reduce tradeability if not aligned with strategic direction
        if strategic_direction * tactical_direction < 0:  # Opposite directions
            tradeability *= 0.5  # Penalty for misalignment

        return max(0.0, min(1.0, tradeability))

    def _should_allow_tactical_trade(self,
                                   tradeability: float,
                                   strategic_state: Dict[str, Any]) -> bool:
        """Determine if tactical trade should be allowed."""
        # Check minimum tradeability threshold
        if tradeability < self.min_tradeability:
            return False

        # Check strategic alignment
        strategic_confidence = strategic_state.get('confidence', 0)
        if strategic_confidence < 0.7:
            return False

        # Check regime conditions
        regime_quality = strategic_state.get('regime_quality', 0)
        if regime_quality < 0.6:
            return False

        return True


def create_tactical_cabr():
    """Create and test tactical CABR system."""
    print("V24.2 Tactical CABR System")
    print("=" * 25)

    # Initialize tactical CABR
    tactical_cabr = TacticalCABR({
        'min_tradeability': 0.35
    })

    # Create sample tactical paths
    sample_paths = [
        {
            'path_id': i,
            'branch_probability': 0.6 + np.random.random() * 0.4,
            'expected_direction': 1 if i % 2 == 0 else -1,
            'risk_score': np.random.random() * 0.5
        }
        for i in range(5)
    ]

    # Create sample strategic state
    sample_strategic_state = {
        'direction': 1,
        'confidence': 0.8,
        'regime_quality': 0.7,
        'tradeability': 0.65
    }

    # Calculate tradeability
    results = tactical_cabr.calculate_tactical_tradeability(
        sample_paths, sample_strategic_state
    )

    print("Tactical CABR Results:")
    print(f"  Short Tradeability: {results['short_tradeability']:.4f}")
    print(f"  Should Trade: {results['should_trade']}")
    print(f"  Best Branch ID: {results['best_short_branch'].get('path_id', 'N/A') if results['best_short_branch'] else 'None'}")

    return results


if __name__ == "__main__":
    # Run the tactical CABR system
    create_tactical_cabr()