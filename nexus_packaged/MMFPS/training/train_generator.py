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
    batch_size: int = 256  # Increased for GPU utilization
    num_paths: int = 8  # Reduced for speed
    learning_rate: float = 1e-4
    max_steps: int = 25000  # Train by steps, not epochs
    gradient_clip: float = 0.5  # Reduced to prevent spikes
    
    # Loss weights
    diffusion_weight: float = 1.0
    diversity_weight: float = 0.05  # Reduced to prevent spikes
    regime_consistency_weight: float = 0.05
    
    # Sampling - reduced for speed
    sampling_steps: int = 8  # Reduced for 2x speed
    train_diffusion_steps: int = 8  # Training uses fewer steps
    
    # Mixed precision
    use_amp: bool = True
    
    # Logging
    log_every: int = 50
    
    # Data
    features_path: str = "data/features/diffusion_fused_6m.npy"
    targets_path: str = "data/features/diffusion_fused_6m.npy"
    sequence_stride: int = 20
    
    # Checkpointing
    checkpoint_dir: str = "outputs/MMFPS/generator_checkpoints"
    checkpoint_every: int = 50  # Save every 50 steps
    
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
        
        # Calculate valid indices - need 2*seq_len for context + target
        self.max_idx = len(features) - 2 * sequence_length - 1
        if self.max_idx < 0:
            self.max_idx = 0
        self.indices = list(range(0, self.max_idx, stride))
        
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = self.indices[idx]
        
        # Data is (samples, features) = (N, 144)
        # Reshape to sequences: (horizon, features) = (20, 144)
        # We need stride of 20 to create overlapping sequences
        seq_len = self.sequence_length
        
        # Take consecutive chunks - ensure we don't go out of bounds
        end_idx = min(i + 2 * seq_len, len(self.features))
        actual_len = end_idx - i
        
        # If not full length, pad with zeros
        if actual_len < 2 * seq_len:
            # Pad the context and target with zeros
            context_feat = np.zeros((2 * seq_len, self.features.shape[1]), dtype=np.float32)
            target_feat = np.zeros((2 * seq_len, self.features.shape[1]), dtype=np.float32)
            
            # Fill available data
            avail = min(seq_len, len(self.features) - i)
            if avail > 0:
                context_feat[:avail] = self.features[i:i + avail]
            if len(self.features) - i - seq_len > 0:
                avail_target = min(seq_len, len(self.features) - i - seq_len)
                if avail_target > 0:
                    target_feat[:avail_target] = self.features[i + seq_len:i + seq_len + avail_target]
        else:
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
            
            loss_val = float('nan')
            if loss is not None:
                with torch.no_grad():
                    loss_val = loss.abs().mean().item()
            if loss is not None and not (torch.isnan(torch.tensor(loss_val)) or loss_val == 0.0):
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                scheduler.step()
            
            total_loss += loss_val
            
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
            
            diff_loss_val = loss.abs().mean().item() if loss is not None else 0.0
             
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
        seq_len = config.horizon
        C = config.in_channels
        
        context_seq = context_seqs.view(B, seq_len, C)
        context = context_seq[:, -1, :]  # Last timestep for regime: (B, C)
        
        # Generate paths
        regime_state = regime_detector(context)
        regime_emb = regime_state.regime_embedding
        quant_emb = torch.zeros(B, config.quant_dim, device=device)
        
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
        total_steps=config.max_steps,
        pct_start=0.1,
    )
    
    # AMP scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
    
    # Resume if needed
    start_step = 0
    global_step = 0
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=config.device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = checkpoint.get("step", 0)
        start_step = global_step + 1
    
    # Training loop - step-based
    best_diversity = 0.0
    metrics_history = []
    total_loss = 0.0
    num_batches = 0
    
    # EMA tracking
    loss_ema = 0.0
    std_ema = 0.0
    range_ema = 0.0
    
    logger.info(f"=== Training for {config.max_steps} steps ===")
    
    model.train()
    device = torch.device(config.device)
    
    pbar = tqdm(total=config.max_steps, desc="Training")
    
    while global_step < config.max_steps:
        for batch_idx, (context_seqs, target_seqs) in enumerate(train_loader):
            if global_step >= config.max_steps:
                break
                
            context_seqs = context_seqs.to(device)
            target_seqs = target_seqs.to(device)
            
            B = context_seqs.shape[0]
            seq_len = config.horizon
            C = config.in_channels
            
            context_seq = context_seqs.view(B, seq_len, C)
            context = context_seq[:, -1, :]
            
            regime_state = regime_detector(context)
            regime_emb = regime_state.regime_embedding
            quant_emb = torch.zeros(B, config.quant_dim, device=device)
            
            try:
                target_for_model = target_seqs.view(B, 1, seq_len, C)
                
                # Forward with AMP
                if config.use_amp:
                    with torch.cuda.amp.autocast():
                        output = model(
                            context=context,
                            regime_emb=regime_emb,
                            quant_emb=quant_emb,
                            targets=target_for_model,
                        )
                        loss = output.diversity_loss
                        
                        loss_val = float('nan')
                        if loss is not None:
                            with torch.no_grad():
                                loss_val = loss.abs().mean().item()
                        if loss is not None and not (torch.isnan(torch.tensor(loss_val)) or loss_val == 0.0):
                            optimizer.zero_grad()
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                            scaler.step(optimizer)
                            scaler.update()
                            scheduler.step()
                else:
                    output = model(
                        context=context,
                        regime_emb=regime_emb,
                        quant_emb=quant_emb,
                        targets=target_for_model,
                    )
                    loss = output.diversity_loss
                    
                    loss_val = float('nan')
                    if loss is not None:
                        with torch.no_grad():
                            loss_val = loss.abs().mean().item()
                    if loss is not None and not (torch.isnan(torch.tensor(loss_val)) or loss_val == 0.0):
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                        optimizer.step()
                        scheduler.step()
                
                # Metrics - update EVERY step
                if output.paths is not None:
                    gen_paths = output.paths
                    path_returns = (gen_paths[:, :, -1, :] - gen_paths[:, :, 0, :]) / (gen_paths[:, :, 0, :].abs() + 1e-8)
                    path_returns = path_returns.mean(dim=-1)
                    path_std = path_returns.std(dim=1).mean().item()
                    path_range = (path_returns.max(dim=1)[0] - path_returns.min(dim=1)[0]).mean().item()
                    path_mean = path_returns.mean().item()
                else:
                    path_std = path_range = path_mean = 0.0
                
                loss_val = loss.abs().item() if loss is not None else 0.0
                
                # Update EMA metrics
                if global_step == 0:
                    loss_ema = loss_val
                    std_ema = path_std
                    range_ema = path_range
                else:
                    loss_ema = 0.98 * loss_ema + 0.02 * loss_val
                    std_ema = 0.98 * std_ema + 0.02 * path_std
                    range_ema = 0.98 * range_ema + 0.02 * path_range
                
                total_loss += loss_val
                num_batches += 1
                
                # Update progress EVERY step (not just every log_every)
                pbar.set_postfix({
                    "loss": f"{loss_val:.4f}",
                    "ema": f"{loss_ema:.4f}",
                    "std": f"{path_std:.2f}",
                })
            
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                logger.warning(f"Training error: {e}")
                continue
            
            global_step += 1
            pbar.update(1)
            
            # Checkpoint every 200 steps
            if global_step % config.checkpoint_every == 0:
                torch.save({
                    "step": global_step,
                    "model": model.state_dict(),
                    "regime_detector": regime_detector.state_dict(),
                    "quant_extractor": quant_extractor.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                }, output_dir / f"checkpoint_step_{global_step}.pt")
                logger.info(
                    f"[{global_step}] loss={loss_val:.4f}, ema={loss_ema:.4f}, "
                    f"std={path_std:.2f}, range={path_range:.2f}"
                )
            
            # Validation every 500 steps
            if global_step % 500 == 0 and global_step > 0:
                val_metrics = validate(
                    model, regime_detector, quant_extractor,
                    val_loader, config,
                )
                logger.info(
                    f"Step {global_step} - Val diversity: {val_metrics['diversity']:.4f}, "
                    f"path_std: {val_metrics['path_std']:.4f}"
                )
                
                if val_metrics['diversity'] > best_diversity:
                    best_diversity = val_metrics['diversity']
                    torch.save({
                        "step": global_step,
                        "model": model.state_dict(),
                        "regime_detector": regime_detector.state_dict(),
                        "quant_extractor": quant_extractor.state_dict(),
                        "metrics": val_metrics,
                    }, output_dir / "best_model.pt")
                    logger.info(f"Saved best model (diversity: {best_diversity:.4f})")
                
                model.train()
    
    pbar.close()
    logger.info(f"Training complete! Steps: {global_step}, Best diversity: {best_diversity:.4f}")


if __name__ == "__main__":
    main()