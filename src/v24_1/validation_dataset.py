"""
V24.1 Validation Dataset Creation Module

This module creates comprehensive validation datasets for V24.1 system validation.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import json
from datetime import datetime
import os


@dataclass
class ValidationDataset:
    """Class to manage validation dataset creation and management."""

    def __init__(self):
        self.data = []
        self.metadata = {}

    def create_validation_dataset(self,
                                 world_states: List[Dict[str, Any]],
                                 branches: List[List[np.ndarray]],
                                 cabr_scores: List[Dict[str, Any]],
                                 outcomes: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Create comprehensive validation dataset.

        Args:
            world_states: List of world state dictionaries
            branches: List of generated market branches
            cabr_scores: List of CABR score dictionaries
            outcomes: List of realized outcomes

        Returns:
            DataFrame containing validation dataset
        """
        # Create validation dataset with all required fields
        dataset = []

        for i, (world_state, branch_set, cabr_score, outcome) in enumerate(
            zip(world_states, branches, cabr_scores, outcomes)
        ):
            record = {
                'timestamp': world_state.get('timestamp', datetime.now().isoformat()),
                'world_state': world_state,
                'generated_branches': branch_set,
                'cabr_scores': cabr_score,
                'realized_outcomes': outcome,
                'macro_regime': world_state.get('macro_regime', {}),
                'branch_realism_metrics': self._calculate_branch_realism(branch_set)
            }
            dataset.append(record)

        return pd.DataFrame(dataset)

    def _calculate_branch_realism(self, branches: List[np.ndarray]) -> Dict[str, float]:
        """
        Calculate branch realism metrics for validation.

        Args:
            branches: List of generated market branches

        Returns:
            Dictionary of realism metrics
        """
        if not branches:
            return {
                'volatility_realism': 0.0,
                'analog_similarity': 0.0,
                'regime_consistency': 0.0,
                'path_plausibility': 0.0,
                'minority_usefulness': 0.0
            }

        # Calculate various realism metrics
        # This is a simplified implementation - in practice this would be more complex
        return {
            'volatility_realism': np.mean([np.std(branch) for branch in branches]) if branches else 0.0,
            'analog_similarity': 0.85,  # Placeholder value
            'regime_consistency': 0.92,  # Placeholder value
            'path_plausibility': 0.78,  # Placeholder value
            'minority_usefulness': 0.73  # Placeholder value
        }

    def save_validation_dataset(self, data: pd.DataFrame, filepath: str) -> bool:
        """
        Save validation dataset to parquet file.

        Args:
            data: DataFrame containing validation data
            filepath: Path to save the file

        Returns:
            True if successful, False otherwise
        """
        try:
            data.to_parquet(filepath, index=False)
            return True
        except Exception as e:
            print(f"Error saving validation dataset: {e}")
            return False


def create_validation_dataset_cli():
    """Create CLI interface for validation dataset creation."""
    print("V24.1 Validation Dataset Creation")
    print("=" * 35)

    # Initialize validation dataset
    validator = ValidationDataset()

    # Create sample data for demonstration
    sample_world_states = [
        {
            'timestamp': '2026-04-12T10:00:00Z',
            'symbol': 'XAUUSD',
            'direction': 'BUY',
            'market_structure': {'close': 2350.50, 'atr_pct': 0.0015, 'vol_regime': 2},
            'nexus_features': {'cabr_score': 0.75, 'confidence_score': 0.82},
            'quant_models': {'hmm_confidence': 0.66, 'hmm_persistence_count': 3},
            'runtime_state': {'rolling_win_rate_10': 0.55, 'consecutive_losses': 0},
            'execution_context': {'v22_risk_score': 0.25, 'v22_meta_label_prob': 0.65}
        }
    ]

    # Create validation dataset
    dataset = validator.create_validation_dataset(
        world_states=sample_world_states,
        branches=[[]],  # Empty branches for demo
        cabr_scores=[{'score': 0.75}],  # Sample CABR score
        outcomes=[{'profit': 10.5, 'loss': -5.2}]  # Sample outcome
    )

    # Save dataset
    success = validator.save_validation_dataset(
        dataset,
        'outputs/v24_1/validation_dataset.parquet'
    )

    if success:
        print("Validation dataset created successfully")
    else:
        print("Failed to create validation dataset")

    return success


if __name__ == "__main__":
    # Run the validation dataset creation
    create_validation_dataset_cli()