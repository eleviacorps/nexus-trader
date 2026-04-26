"""V30 Training script for path evaluator."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
from typing import Generator

# Add parent to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from v30.models.evaluator.evaluator import PathEvaluator
from v30.models.evaluator.loss import WeightedPathLoss, CalibrationLoss
from v30.models.evaluator.structure_similarity import compute_path_rewards
from v30.training.dataloader import load_alignment_data


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """Setup logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    
    # Console
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    # File
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class DiffusionPathGenerator:
    """Wrapper to generate diffusion paths from frozen model."""
    
    def __init__(self, model, config: dict):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def __call__(self, context_window: np.ndarray) -> np.ndarray:
        """Generate paths from context.
        
        Args:
            context_window: [lookback, feature_dim]
        
        Returns:
            paths: [num_paths, horizon]
        """
        self.model.eval()
        
        # Handle both (T, D) and (B, T, D) inputs
        if context_window.ndim == 2:
            context = torch.from_numpy(context_window).float().unsqueeze(0).to(self.device)
        else:
            context = torch.from_numpy(context_window).float().to(self.device)
        
        with torch.no_grad():
            paths = self.model(context)
        
        # Handle different output shapes
        if paths.ndim == 3:
            paths = paths[0]  # [num_paths, horizon]
        
        return paths.cpu().numpy()


def setup_generator(config: dict) -> DiffusionPathGenerator:
    """Load frozen diffusion model."""
    from core.diffusion_loader import DiffusionModelLoader
    from protection.encryptor import derive_key_from_env
    
    logger = logging.getLogger("v30.train")
    
    # Get model path
    model_path = config.get("model", {}).get("generator", {}).get("path", 
        "nexus_packaged/protection/model_enc.bin")
    
    # Derive key
    key = derive_key_from_env(
        env_var="NEXUS_MODEL_KEY",
        salt="nexus_trader_salt_2024",
    )
    
    # Load model
    loader = DiffusionModelLoader(model_path, key, settings=config)
    loader.load()
    loader.warm_up()
    
    logger.info(f"Loaded diffusion model from {model_path}")
    
    return DiffusionPathGenerator(loader, config)


def train_epoch(
    evaluator: nn.Module,
    dataloader: DataLoader,
    path_generator: DiffusionPathGenerator | None,
    optimizer: optim.Optimizer,
    loss_fn: WeightedPathLoss,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    config: dict,
) -> dict[str, float]:
    """Train one epoch."""
    evaluator.train()
    
    loss_weights = config.get("training", {}).get("loss_weights", {
        "direction": 0.4,
        "magnitude": 0.3,
        "structure": 0.3,
    })
    
    total_metrics = {
        "loss": 0.0,
        "expected_reward": 0.0,
        "entropy": 0.0,
        "direction_accuracy": 0.0,
        "max_weight": 0.0,
        "weight_reward_corr": 0.0,
    }
    
    num_batches = 0
    
    for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
        context = batch["context"].to(device)  # [B, T, D]
        actual = batch["actual"].to(device)  # [B, horizon]
        paths = batch["paths"].to(device)  # [B, num_paths, horizon] (precomputed or generated)
        
        # Get evaluator scores
        scores = evaluator(context, paths)  # [B, num_paths]
        
        # Compute rewards against actual
        rewards = compute_path_rewards(
            paths.detach(),
            actual,
            weights=loss_weights,
        )
        
        # Compute loss
        metrics = loss_fn(scores, rewards)
        
        # Backprop
        optimizer.zero_grad()
        metrics["loss"].backward()
        torch.nn.utils.clip_grad_norm_(evaluator.parameters(), 1.0)
        optimizer.step()
        
        # Accumulate metrics
        for k, v in metrics.items():
            if k != "loss":
                total_metrics[k] += v.item()
        
        total_metrics["loss"] += metrics["loss"].item()
        num_batches += 1
    
    # Average metrics
    for k in total_metrics:
        total_metrics[k] /= num_batches
    
    logger.info(
        f"Epoch {epoch} - "
        f"loss: {total_metrics['loss']:.4f}, "
        f"reward: {total_metrics['expected_reward']:.4f}, "
        f"dir_acc: {total_metrics['direction_accuracy']:.2%}"
    )
    
    return total_metrics


def validate(
    evaluator: nn.Module,
    dataloader: DataLoader,
    path_generator: DiffusionPathGenerator | None,
    loss_fn: WeightedPathLoss,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    config: dict,
) -> dict[str, float]:
    """Validate on held-out data."""
    evaluator.eval()
    
    loss_weights = config.get("training", {}).get("loss_weights", {
        "direction": 0.4,
        "magnitude": 0.3,
        "structure": 0.3,
    })
    
    total_metrics = {
        "loss": 0.0,
        "expected_reward": 0.0,
        "entropy": 0.0,
        "direction_accuracy": 0.0,
        "max_weight": 0.0,
        "weight_reward_corr": 0.0,
        "calibration_error": 0.0,
    }
    
    num_batches = 0
    
    all_prob_up = []
    all_actual_up = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Val {epoch}"):
            context = batch["context"].to(device)
            actual = batch["actual"].to(device)
            actual_return = batch["actual_return"].to(device)
            paths = batch["paths"].to(device)  # [B, num_paths, horizon] (precomputed)
            
            # Evaluate
            scores = evaluator(context, paths)
            weights = torch.softmax(scores, dim=-1)
            
            # Rewards
            rewards = compute_path_rewards(
                paths,
                actual,
                weights=loss_weights,
            )
            
            # Metrics
            metrics = loss_fn(scores, rewards)
            
            # Compute prob_up for calibration
            returns = (paths[:, :, -1] - paths[:, :, 0]) / (paths[:, :, 0] + 1e-8)
            prob_up = (weights * (returns > 0).float()).sum(dim=-1)
            
            actual_up = (actual_return > 0).float()
            
            all_prob_up.extend(prob_up.cpu().numpy().tolist())
            all_actual_up.extend(actual_up.cpu().numpy().tolist())
            
            # Accumulate
            for k, v in metrics.items():
                if k != "loss":
                    total_metrics[k] += v.item()
            
            total_metrics["loss"] += metrics["loss"].item()
            num_batches += 1
    
    # Average
    for k in total_metrics:
        total_metrics[k] /= num_batches
    
    # Calibration error
    all_prob_up = np.array(all_prob_up)
    all_actual_up = np.array(all_actual_up)
    total_metrics["calibration_error"] = float(np.abs(all_prob_up - all_actual_up).mean())
    
    logger.info(
        f"Validation {epoch} - "
        f"loss: {total_metrics['loss']:.4f}, "
        f"reward: {total_metrics['expected_reward']:.4f}, "
        f"calib_err: {total_metrics['calibration_error']:.4f}"
    )
    
    return total_metrics


def main():
    parser = argparse.ArgumentParser(description="Train V30 Path Evaluator")
    parser.add_argument("--config", type=str, 
                        default=str(Path(__file__).parent.parent / "configs" / "v30_config.yaml"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, 
                        default=str(Path(__file__).parent.parent / "models" / "checkpoints"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use-precomputed", action="store_true", default=True,
                        help="Use precomputed paths from disk (FAST)")
    parser.add_argument("--paths-file", type=str, 
                        default=str(Path(__file__).parent.parent / "data" / "processed" / "v30_paths.npy"),
                        help="Path to precomputed paths file")
    parser.add_argument("--no-precomputed", action="store_true",
                        help="Force on-the-fly generation (SLOW)")
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = get_logger(
        "v30.train",
        log_file=str(output_dir / "training.log")
    )
    
    # Load config
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading data...")
    features, ohlcv = load_alignment_data(config.get("data", {}))
    logger.info(f"Loaded features: {features.shape}, OHLCV: {len(ohlcv)}")
    
    # Load precomputed paths if available and requested
    precomputed_paths = None
    use_precomputed = args.use_precomputed and not args.no_precomputed
    
    if use_precomputed:
        paths_file = Path(args.paths_file)
        if paths_file.exists():
            logger.info(f"Loading precomputed paths from {paths_file}...")
            precomputed_paths = np.load(paths_file)
            logger.info(f"Precomputed paths shape: {precomputed_paths.shape}")
        else:
            logger.warning(f"Precomputed paths not found at {paths_file}, falling back to generator")
            use_precomputed = False
    
    # Setup generator (only if not using precomputed)
    path_generator = None
    if not use_precomputed:
        logger.info("Setting up diffusion generator (on-the-fly mode)...")
        path_generator = setup_generator(config)
    
    # Create dataloaders (simplified - just use a slice for now)
    from v30.training.dataloader import DiffusionEvaluatorDataset
    
    model_cfg = config.get("model", {}).get("generator", {})
    eval_cfg = config.get("model", {}).get("evaluator", {})
    
    lookback = model_cfg.get("lookback", 128)
    horizon = model_cfg.get("horizon", 20)
    num_paths = model_cfg.get("num_paths", 64)
    
    # Simple split - use first 80% for train, last 20% for val
    split_idx = int(len(features) * 0.8)
    
    logger.info(f"Split: train samples 0-{split_idx}, val samples {split_idx}-{len(features)}")
    
    if use_precomputed:
        # Precomputed paths: paths array is 15-min aligned
        # split_idx is in 1-min units, convert to path indices
        path_start = 0  # Paths start at index 0 (precomputed already handles lookback)
        
        train_path_end = split_idx // 15
        val_path_end = len(features) // 15
        
        train_paths = precomputed_paths[path_start:train_path_end]
        val_paths = precomputed_paths[train_path_end:val_path_end]
        
        logger.info(f"Train paths: {train_paths.shape}, Val paths: {val_paths.shape}")
        logger.info(f"Train samples (paths): {len(train_paths)}, Val samples: {len(val_paths)}")
        
        # For precomputed mode, dataset length = number of paths
        train_dataset = DiffusionEvaluatorDataset(
            features=features,
            ohlcv=ohlcv,
            precomputed_paths=train_paths,
            lookback=lookback,
            horizon=horizon,
            num_paths=num_paths,
            start_idx=0,
            end_idx=len(train_paths),
        )
        
        val_dataset = DiffusionEvaluatorDataset(
            features=features,
            ohlcv=ohlcv,
            precomputed_paths=val_paths,
            lookback=lookback,
            horizon=horizon,
            num_paths=num_paths,
            start_idx=0,
            end_idx=len(val_paths),
        )
    else:
        # Use generator (slow mode)
        train_dataset = DiffusionEvaluatorDataset(
            features=features,
            ohlcv=ohlcv,
            generator=path_generator,
            lookback=lookback,
            horizon=horizon,
            num_paths=num_paths,
            start_idx=lookback,
            end_idx=split_idx,
        )
        
        val_dataset = DiffusionEvaluatorDataset(
            features=features,
            ohlcv=ohlcv,
            generator=path_generator,
            lookback=lookback,
            horizon=horizon,
            num_paths=num_paths,
            start_idx=split_idx,
            end_idx=len(features) - horizon - 1,
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Keep 0 for diffusion gen
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    model_cfg = config.get("model", {}).get("generator", {})
    eval_cfg = config.get("model", {}).get("evaluator", {})
    
    evaluator = PathEvaluator(
        feature_dim=144,  # Match features
        path_dim=horizon,
        num_paths=num_paths,
        hidden_dim=eval_cfg.get("hidden_dim", 256),
        num_layers=eval_cfg.get("num_layers", 3),
        dropout=eval_cfg.get("dropout", 0.1),
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(evaluator.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Loss
    loss_fn = WeightedPathLoss()
    
    # Training loop
    best_val_loss = float("inf")
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n=== Epoch {epoch}/{args.epochs} ===")
        
        # Train
        train_metrics = train_epoch(
            evaluator, train_loader, path_generator,
            optimizer, loss_fn, device, epoch, logger, config
        )
        
        # Validate
        val_metrics = validate(
            evaluator, val_loader, path_generator,
            loss_fn, device, epoch, logger, config
        )
        
        scheduler.step()
        
        # Save best
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            checkpoint_path = output_dir / "best_evaluator.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": evaluator.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["loss"],
                "config": config,
            }, checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")
        
        # Save periodic
        if epoch % 5 == 0:
            checkpoint_path = output_dir / f"evaluator_epoch{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": evaluator.state_dict(),
                "val_loss": val_metrics["loss"],
            }, checkpoint_path)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()