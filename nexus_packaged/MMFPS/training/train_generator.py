"""Training script for MMFPS diffusion generator.

Trains the diffusion generator with:
- Diffusion epsilon loss
- Diversity regularization
- Regime/quant conditioning loss
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add project root to path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from MMFPS.generator.diffusion_path_generator import DiffusionPathGenerator, DiffusionGeneratorConfig
from MMFPS.regime.regime_detector import RegimeDetector
from MMFPS.quant.quant_features import QuantFeatureExtractor


def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if log_file:
        logger.addHandler(logging.FileHandler(log_file))
    return logger


@dataclass
class TrainingConfig:
    """Generator training configuration on REAL data.
    """
    # Model - 368M params (proper for market complexity)
    in_channels: int = 144
    horizon: int = 20
    base_channels: int = 256
    channel_multipliers: tuple[int, ...] = (1, 2, 4, 8)
    time_dim: int = 512
    regime_dim: int = 64
    quant_dim: int = 64
    
    # Training - optimized for speed
    batch_size: int = 128  # Larger for full dataset
    num_paths: int = 64  # Fewer paths for faster training
    learning_rate: float = 1e-4
    epochs: int = 100
    gradient_clip: float = 1.0
    
    # Loss weights
    diffusion_weight: float = 1.0
    diversity_weight: float = 0.1
    regime_consistency_weight: float = 0.05
    
    # Sampling - reduced for faster training
    sampling_steps: int = 20  # Reduced from 50
    
    # Data
    features_path: str = "data/features/diffusion_fused_6m.npy"
    targets_path: str = "data/features/diffusion_fused_6m.npy"
    sequence_stride: int = 20
    
    # Checkpointing
    checkpoint_dir: str = "outputs/MMFPS/generator_checkpoints"
    checkpoint_interval: int = 1  # Save every epoch
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class PathDataset(Dataset):
    """Dataset for path generation training.
    
    Loads sequences from features and creates training pairs.
    """
    
    def __init__(
        self,
        features: np.ndarray,
        sequence_length: int = 20,
        stride: int = 20,
    ):
        self.features = features
        self.sequence_length = sequence_length
        self.stride = stride
        
        # Calculate valid indices
        self.max_idx = len(features) - sequence_length - 1
        self.indices = list(range(0, self.max_idx, stride))
        
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = self.indices[idx]
        
        # Data is (samples, features) = (N, 144)
        # Reshape to sequences: (horizon, features) = (20, 144)
        # We need stride of 20 to create overlapping sequences
        seq_len = self.sequence_length
        
        # Take consecutive chunks
        context_feat = self.features[i:i + seq_len]
        target_feat = self.features[i + seq_len:i + 2 * seq_len]
        
        # Flatten to 1D vector for DataLoader
        context = context_feat.flatten()  # (20*144,) = 2880
        target = target_feat.flatten()  # (20*144,) = 2880
        
        return (
            torch.from_numpy(context.astype(np.float32)),
            torch.from_numpy(target.astype(np.float32)),
        )


def train_epoch(
    model: DiffusionPathGenerator,
    regime_detector: RegimeDetector,
    quant_extractor: QuantFeatureExtractor,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    config: TrainingConfig,
    logger: logging.Logger,
    epoch: int,
) -> dict:
    """Train for one epoch."""
    model.train()
    device = torch.device(config.device)
    
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (context_seqs, target_seqs) in enumerate(pbar):
        context_seqs = context_seqs.to(device)
        target_seqs = target_seqs.to(device)
        
        B = context_seqs.shape[0]
        seq_len = config.horizon
        C = config.in_channels
        
        # Data is (B, seq_len*C) flat. Reshape to (B, seq_len, C)
        # seq_len*C = 20*144 = 2880
        total_features = seq_len * C
        context_seq = context_seqs.view(B, seq_len, C)
        context = context_seq[:, -1, :]  # Last timestep for regime: (B, C)
        
        # Get regime embedding  
        regime_state = regime_detector(context)
        regime_emb = regime_state.regime_embedding
        
        # Quant embedding
        quant_emb = torch.zeros(B, config.quant_dim, device=device)
        
        # ===== PROPER DIFFUSION TRAINING WITH FULL METRICS =====
        try:
            model.train()
            
            # Reshape targets: (B, seq_len*C) -> (B, 1, horizon, channels)
            target_for_model = target_seqs.view(B, 1, seq_len, C)
            
            # Forward pass with targets (enables diffusion loss with gradients)
            output = model(
                context=context,
                regime_emb=regime_emb,
                quant_emb=quant_emb,
                targets=target_for_model,
            )
            
            # Get losses from output
            loss = output.diversity_loss
            
            if loss is not None and loss.abs() > 0:
                optimizer.zero_grad()
                if loss.requires_grad == False:
                    loss = loss.detach().requires_grad_(True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.abs().item()
                
                # Get metrics from the generated paths
                if output.paths is not None:
                    gen_paths = output.paths
                    path_returns = (gen_paths[:, :, -1, :] - gen_paths[:, :, 0, :]) / (gen_paths[:, :, 0, :].abs() + 1e-8)
                    path_returns = path_returns.mean(dim=-1)
                    path_std = path_returns.std(dim=1).mean().item()
                    path_range = (path_returns.max(dim=1)[0] - path_returns.min(dim=1)[0]).mean().item()
                    path_mean = path_returns.mean().item()
                else:
                    path_std = 0.0
                    path_range = 0.0
                    path_mean = 0.0
                
                # Track metrics
                diff_loss_val = loss.item()
                num_batches += 1
            else:
                num_batches += 1
                path_std = 0.0
                path_range = 0.0
                path_mean = 0.0
                diff_loss_val = 0.0
             
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            logger.warning(f"Training error: {e}")
            continue
        
        # Update progress with metrics
        avg_loss = total_loss / max(num_batches, 1)
        pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "std": f"{path_std:.4f}",
            "range": f"{path_range:.4f}",
            "mean": f"{path_mean:.4f}",
        })
        
        # Run all batches
        # if batch_idx >= 100:
        #     break
    
    scheduler.step()
    
    return {
        "loss": total_loss / num_batches if num_batches > 0 else 0,
        "diffusion_loss": diff_loss_val,
        "diversity_loss": path_std,  # Reuse as proxy
    }


@torch.no_grad()
def validate(
    model: DiffusionPathGenerator,
    regime_detector: RegimeDetector,
    quant_extractor: QuantFeatureExtractor,
    dataloader: DataLoader,
    config: TrainingConfig,
) -> dict:
    """Validation pass."""
    model.eval()
    device = torch.device(config.device)
    
    total_diversity = 0.0
    total_path_std = 0.0
    total_path_range = 0.0
    num_samples = 0
    
    for context_seqs, target_seqs in dataloader:
        context_seqs = context_seqs.to(device)
        
        B = context_seqs.shape[0]
        context = context_seqs.mean(dim=1)
        
        # Generate paths
        regime_state = regime_detector(context)
        regime_emb = regime_state.regime_embedding
        quant_emb = quant_extractor(target_seqs.permute(0, 1, 2).to(device))
        
        paths = model.quick_generate(
            context=context,
            regime_emb=regime_emb,
            quant_emb=quant_emb,
            num_paths=config.num_paths // 2,  # Fewer for validation
        )
        
        # Compute diversity metrics
        returns = (paths[:, :, -1, :] - paths[:, :, 0, :]) / (paths[:, :, 0, :].abs() + 1e-8)
        returns = returns.mean(dim=-1)
        
        path_std = returns.std(dim=1).mean()
        path_range = (returns.max(dim=1)[0] - returns.min(dim=1)[0]).mean()
        
        # Pairwise separation
        returns_exp = returns.unsqueeze(2) - returns.unsqueeze(1)
        pairwise_dist = returns_exp.abs()
        eye = torch.eye(config.num_paths // 2, device=pairwise_dist.device).unsqueeze(0)
        pairwise_dist = pairwise_dist * (1 - eye)
        avg_sep = pairwise_dist.sum() / (B * (config.num_paths // 2) * ((config.num_paths // 2) - 1) + 1e-8)
        
        total_diversity += (path_std + 0.1 * path_range + 0.01 * avg_sep).item()
        total_path_std += path_std.item()
        total_path_range += path_range.item()
        num_samples += 1
    
    return {
        "diversity": total_diversity / num_samples,
        "path_std": total_path_std / num_samples,
        "path_range": total_path_range / num_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Train MMFPS diffusion generator")
    parser.add_argument("--config", type=str, default=None, help="Config JSON path")
    parser.add_argument("--features", type=str, default=None, help="Features path")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-paths", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    args = parser.parse_args()
    
    # Load config
    if args.config:
        with open(args.config, "r") as f:
            config_dict = json.load(f)
        config = TrainingConfig(**config_dict)
    else:
        config = TrainingConfig()
    
    # Override with args
    if args.features:
        config.features_path = args.features
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.num_paths:
        config.num_paths = args.num_paths
    if args.device:
        config.device = args.device
    
    # Setup
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / config.checkpoint_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger("generator_train", str(output_dir / "training.log"))
    logger.info(f"Config: {config}")
    logger.info(f"Device: {config.device}")
    
    # Load data - check multiple possible paths
    logger.info(f"Loading features from {config.features_path}...")
    features_path = repo_root / config.features_path
    if not features_path.exists():
        # Try alternative paths
        alternative_paths = [
            "data/features/diffusion_fused_6m.npy",
            "data/features/diffusion_fused_405k.npy",
            "C:/PersonalDrive/Programming/AiStudio/nexus-trader/data/features/diffusion_fused_6m.npy",
        ]
        for alt_path in alternative_paths:
            alt = Path(alt_path)
            if alt.exists():
                features_path = alt
                logger.info(f"Using alternative path: {features_path}")
                break
    
    if not features_path.exists():
        raise FileNotFoundError(f"Features not found at any path")
    
    logger.info(f"Loading from {features_path}")
    features = np.load(features_path, mmap_mode="r").astype(np.float32)
    logger.info(f"Loaded features shape: {features.shape}")
    
    # Dataset
    dataset = PathDataset(
        features=features,
        sequence_length=config.horizon,
        stride=config.sequence_stride,
    )
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size // 2,
        shuffle=False,
        num_workers=0,
    )
    
    logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Models
    generator_config = DiffusionGeneratorConfig(
        in_channels=config.in_channels,
        horizon=config.horizon,
        base_channels=config.base_channels,
        channel_multipliers=config.channel_multipliers,
        time_dim=config.time_dim,
        regime_dim=config.regime_dim,
        quant_dim=config.quant_dim,
        num_paths=config.num_paths,
        sampling_steps=config.sampling_steps,
    )
    
    model = DiffusionPathGenerator(generator_config).to(config.device)
    regime_detector = RegimeDetector(
        feature_dim=config.in_channels,
        embed_dim=config.regime_dim,
    ).to(config.device)
    quant_extractor = QuantFeatureExtractor(
        path_dim=config.horizon,
        embed_dim=config.quant_dim,
    ).to(config.device)
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {num_params:,}")
    
    # Use AdamW optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(regime_detector.parameters()) + list(quant_extractor.parameters()),
        lr=config.learning_rate,
        weight_decay=1e-4,
    )
    
    # Use cosine annealing with warmup
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        epochs=config.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )
    
    # Resume if needed
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=config.device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
    
    # Training loop
    best_diversity = 0.0
    metrics_history = []
    
    for epoch in range(start_epoch, config.epochs):
        logger.info(f"=== Epoch {epoch + 1}/{config.epochs} ===")
        
        train_metrics = train_epoch(
            model, regime_detector, quant_extractor,
            train_loader, optimizer, scheduler,
            config, logger, epoch,
        )
        
        logger.info(
            f"Train - loss: {train_metrics['loss']:.4f}, "
            f"diff: {train_metrics['diffusion_loss']:.4f}, "
            f"div: {train_metrics['diversity_loss']:.4f}"
        )
        
        # Validation every few epochs
        if epoch % 2 == 0:
            val_metrics = validate(
                model, regime_detector, quant_extractor,
                val_loader, config,
            )
            logger.info(
                f"Val - diversity: {val_metrics['diversity']:.4f}, "
                f"path_std: {val_metrics['path_std']:.4f}, "
                f"path_range: {val_metrics['path_range']:.4f}"
            )
            
            if val_metrics['diversity'] > best_diversity:
                best_diversity = val_metrics['diversity']
                # Save best
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "regime_detector": regime_detector.state_dict(),
                    "quant_extractor": quant_extractor.state_dict(),
                    "metrics": val_metrics,
                }, output_dir / "best_model.pt")
                logger.info(f"Saved best model (diversity: {best_diversity:.4f})")
        
        # Checkpoint
        if epoch % config.checkpoint_interval == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "regime_detector": regime_detector.state_dict(),
                "quant_extractor": quant_extractor.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
            }, output_dir / f"checkpoint_epoch_{epoch}.pt")
        
        # Save metrics
        metrics_history.append({
            "epoch": epoch,
            **train_metrics,
        })
        
        with open(output_dir / "metrics_history.json", "w") as f:
            json.dump(metrics_history, f, indent=2)
    
    logger.info("Training complete!")
    logger.info(f"Best diversity: {best_diversity:.4f}")


if __name__ == "__main__":
    main()