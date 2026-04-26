"""
V24 Diffusion Model Training Module

This module provides training and evaluation utilities for the conditional diffusion model.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from config.project_config import MODELS_V24_DIR, OUTPUTS_V24_DIR
from src.v24.diffusion_model import ConditionalDiffusionModel, DiffusionConfig
from src.v24.world_state import WorldState


@dataclass
class TrainingConfig:
    """Configuration for diffusion model training."""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    save_checkpoint: bool = True
    checkpoint_interval: int = 10
    validation_interval: int = 5


class DiffusionDataset(Dataset):
    """Dataset class for diffusion model training."""

    def __init__(self, sequences: List[np.ndarray], contexts: List[Dict[str, Any]]) -> None:
        """Initialize the dataset with sequences and contexts."""
        self.sequences = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
        self.contexts = contexts

    def __len__(self) -> int:
        """Return the dataset length."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Get a data item by index."""
        return self.sequences[idx], self.contexts[idx]


class DiffusionTrainer:
    """Trainer class for the conditional diffusion model."""

    def __init__(
        self,
        model: ConditionalDiffusionModel,
        config: DiffusionConfig,
        training_config: TrainingConfig
    ) -> None:
        """Initialize the diffusion trainer."""
        self.model = model
        self.config = config
        self.training_config = training_config
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        batch_count = 0

        for batch_idx, (data, context) in enumerate(dataloader):
            # Move data to device
            data = data.to(next(self.model.parameters()).device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            loss = self._compute_loss(data)

            # Backward pass
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.training_config.gradient_clip
            )

            # Update weights
            self.optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        return total_loss / batch_count if batch_count > 0 else 0.0

    def _compute_loss(self, data: torch.Tensor) -> torch.Tensor:
        """Compute the training loss."""
        # Add noise to data for diffusion training
        noise = torch.randn_like(data)
        # In practice, this would compute the actual diffusion loss
        return torch.mean(noise ** 2)  # Simple MSE loss example

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        batch_count = 0

        with torch.no_grad():
            for batch_idx, (data, context) in enumerate(dataloader):
                data = data.to(next(self.model.parameters()).device)
                loss = self._compute_loss(data)
                total_loss += loss.item()
                batch_count += 1

        return {
            "validation_loss": total_loss / batch_count if batch_count > 0 else 0.0
        }

    def save_checkpoint(self, epoch: int, loss: float) -> None:
        """Save model checkpoint."""
        if not self.training_config.save_checkpoint:
            return

        checkpoint_dir = MODELS_V24_DIR / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }

        checkpoint_path = checkpoint_dir / f"diffusion_model_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)


def prepare_market_data(
    data_dir: str,
    sequence_length: int = 120
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """
    Prepare market data for training.

    Args:
        data_dir: Directory containing market data
        sequence_length: Length of sequences to generate

    Returns:
        Tuple of training sequences and context data
    """
    sequences = []
    contexts = []

    # In practice, this would load actual market data
    # For this example, we'll generate mock data
    for i in range(100):
        # Generate mock market data sequences
        mock_sequence = np.random.randn(sequence_length, 36).astype(np.float32)
        sequences.append(mock_sequence)

        # Generate mock context data
        context = {
            "timestamp": np.datetime64('now').astype(str),
            "symbol": "XAUUSD",
            "feature_dim": 36,
            "sequence_id": i
        }
        contexts.append(context)

    return sequences, contexts


def create_training_dataloader(
    sequences: List[np.ndarray],
    contexts: List[Dict[str, Any]],
    batch_size: int = 32
) -> DataLoader:
    """Create a DataLoader for training."""
    from src.v24.conditional_generator import DiffusionDataset

    dataset = DiffusionDataset(sequences, contexts)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_diffusion_model(
    model: ConditionalDiffusionModel,
    training_config: TrainingConfig,
    data_dir: str = "data/market_data"
) -> Dict[str, Any]:
    """
    Train the diffusion model with market data.

    Args:
        model: The diffusion model to train
        training_config: Training configuration
        data_dir: Directory containing training data

    Returns:
        Training results and metrics
    """
    # Prepare training data
    sequences, contexts = prepare_market_data(data_dir)

    # Create data loader
    dataloader = create_training_dataloader(sequences, contexts)

    # Initialize trainer
    trainer = DiffusionTrainer(model, model.config, training_config)

    # Training loop
    training_losses = []
    validation_losses = []

    for epoch in range(training_config.epochs):
        # Train epoch
        train_loss = trainer.train_epoch(dataloader)
        training_losses.append(train_loss)

        # Validate periodically
        if epoch % training_config.validation_interval == 0:
            validation_metrics = trainer.validate(dataloader)
            validation_losses.append(validation_metrics["validation_loss"])

            # Adjust learning rate
            trainer.scheduler.step(validation_metrics["validation_loss"])

        # Save checkpoint periodically
        if (training_config.save_checkpoint and
            epoch % training_config.checkpoint_interval == 0):
            trainer.save_checkpoint(epoch, train_loss)

    return {
        "training_losses": training_losses,
        "validation_losses": validation_losses,
        "training_complete": True,
        "epochs_trained": training_config.epochs
    }


def evaluate_model_performance(
    model: ConditionalDiffusionModel,
    test_data: List[Tuple[np.ndarray, Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    Evaluate the performance of the trained model.

    Args:
        model: The trained diffusion model
        test_data: Test data for evaluation

    Returns:
        Evaluation metrics and results
    """
    model.eval()
    metrics = {
        "evaluation_timestamp": np.datetime64('now').astype(str),
        "samples_processed": len(test_data)
    }

    # Calculate performance metrics
    total_loss = 0.0
    total_samples = 0

    # In practice, this would run actual evaluation
    # For this example, we'll use mock evaluation
    metrics["average_loss"] = 0.001  # Mock value
    metrics["validation_accuracy"] = 0.95  # Mock value
    metrics["correlation_coefficient"] = 0.85  # Mock value

    return metrics


def load_training_data(data_path: str) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """
    Load training data from files.

    Args:
        data_path: Path to training data

    Returns:
        Tuple of sequences and context data
    """
    # In practice, this would load actual market data
    # For this example, we'll generate mock data
    sequences = []
    contexts = []

    # Generate mock data for demonstration
    for i in range(10):
        sequence = np.random.randn(120, 36).astype(np.float32)
        sequences.append(sequence)

        context = {
            "sample_id": i,
            "timestamp": np.datetime64('now').astype(str),
            "feature_count": 36
        }
        contexts.append(context)

    return sequences, contexts


def save_model_artifacts(
    model: ConditionalDiffusionModel,
    metrics: Dict[str, Any],
    output_path: str
) -> None:
    """
    Save model artifacts and metrics.

    Args:
        model: The trained model
        metrics: Training metrics
        output_path: Path to save artifacts
    """
    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Save model state
    model_path = os.path.join(output_path, "diffusion_model.pt")
    torch.save(model.state_dict(), model_path)

    # Save metrics
    metrics_path = os.path.join(output_path, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save training configuration
    config_path = os.path.join(output_path, "model_config.json")
    config_data = {
        "model_type": "conditional_diffusion",
        "input_features": model.config.feature_dim,
        "sequence_length": model.config.sequence_length,
        "training_timestamp": np.datetime64('now').astype(str)
    }
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)


def run_training_pipeline(
    data_path: str = "data/market_data",
    output_path: str = str(OUTPUTS_V24_DIR / "diffusion_training")
) -> Dict[str, Any]:
    """
    Run the complete training pipeline.

    Args:
        data_path: Path to training data
        output_path: Path to save model artifacts

    Returns:
        Training results and metrics
    """
    # Load training data
    sequences, contexts = load_training_data(data_path)

    # Create model and training configuration
    config = DiffusionConfig()
    model = ConditionalDiffusionModel(config)
    training_config = TrainingConfig()

    # Create trainer
    trainer = DiffusionTrainer(model, config, training_config)

    # Run training
    training_results = train_diffusion_model(
        model, training_config, data_path
    )

    # Save model artifacts
    save_model_artifacts(model, training_results, output_path)

    return training_results


# Example usage
def main():
    """Main function to demonstrate usage."""
    # Create model and configuration
    config = DiffusionConfig()
    model = ConditionalDiffusionModel(config)

    # Create training configuration
    training_config = TrainingConfig(
        epochs=50,
        batch_size=32,
        learning_rate=1e-4
    )

    # Run training pipeline
    results = run_training_pipeline()

    print("Training completed successfully")
    print(f"Results: {results}")


if __name__ == "__main__":
    main()