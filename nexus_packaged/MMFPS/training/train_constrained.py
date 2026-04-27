"""Training script for Context-Constrained Diffusion Generator.

This implements the vNext architecture with:
1. ContextEncoder → distribution envelope
2. DiffusionGenerator → z-space generation  
3. Envelope transformation → bounded returns
"""

import sys
import gc
from pathlib import Path
_p = Path(__file__).resolve().parents[2]
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))

import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from MMFPS.generator.constrained_diffusion_generator import (
    ConstrainedDiffusionGenerator,
    DiffusionGeneratorConfig,
)


class DiffusionDataset(Dataset):
    """Dataset for diffusion training."""
    
    def __init__(self, data_path: str, seq_len: int = 20, stride: int = 1):
        print(f"Loading data from {data_path}...")
        self.data = np.load(data_path)
        self.seq_len = seq_len
        self.stride = stride
        
        # Calculate valid indices
        n_samples = len(self.data)
        self.valid_indices = list(range(0, n_samples - seq_len, stride))
        print(f"Loaded {len(self.valid_indices)} sequences")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        seq = torch.from_numpy(self.data[real_idx:real_idx + self.seq_len]).float()
        return seq


def train_one_step(
    model: ConstrainedDiffusionGenerator,
    optimizer: optim.Optimizer,
    batch: torch.Tensor,
    device: torch.device,
    scaler: GradScaler,
) -> dict:
    """Train one step."""
    model.train()
    
    # Prepare batch
    batch = batch.to(device)
    B = batch.shape[0]
    
    # Context features (use first part of sequence as context)
    context = batch[:, :5, :].mean(dim=1)  # (B, C)
    
    # Embeddings (placeholder - would come from regime/quant modules)
    regime_emb = torch.zeros(B, 64, device=device)
    quant_emb = torch.zeros(B, 64, device=device)
    
    # Forward pass
    with autocast():
        output = model(
            context=context,
            regime_emb=regime_emb,
            quant_emb=quant_emb,
            targets=batch.unsqueeze(1),  # Add path dim
        )
    
    loss = output.diversity_loss
    
    # Backward
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    scaler.step(optimizer)
    scaler.update()
    
    return {
        'loss': loss.item(),
    }


@torch.no_grad()
def validate(
    model: ConstrainedDiffusionGenerator,
    val_loader: DataLoader,
    device: torch.device,
    num_batches: int = 10,
) -> dict:
    """Validate model."""
    model.eval()
    
    losses = []
    returns_list = []
    
    for i, batch in enumerate(val_loader):
        if i >= num_batches:
            break
            
        batch = batch.to(device)
        B = batch.shape[0]
        
        context = batch[:, :5, :].mean(dim=1)
        regime_emb = torch.zeros(B, 64, device=device)
        quant_emb = torch.zeros(B, 64, device=device)
        
        output = model(
            context=context,
            regime_emb=regime_emb,
            quant_emb=quant_emb,
            targets=batch.unsqueeze(1),
        )
        
        losses.append(output.diversity_loss.item())
        
        # Get returns from generated paths
        paths = output.paths
        rets = (paths[:, :, -1, 0] - paths[:, :, 0, 0]) / (paths[:, :, 0, 0].abs() + 1e-8)
        returns_list.append(rets.cpu().numpy())
    
    returns = np.concatenate(returns_list)
    
    return {
        'loss': np.mean(losses),
        'returns_mean': np.mean(returns),
        'returns_std': np.std(returns),
        'returns_min': np.min(returns),
        'returns_max': np.max(returns),
        'returns_p5': np.percentile(returns, 5),
        'returns_p95': np.percentile(returns, 95),
    }


def main():
    # Config
    data_path = "C:/PersonalDrive/Programming/AiStudio/nexus-trader/data/features/diffusion_fused_6m.npy"
    checkpoint_dir = Path("C:/PersonalDrive/Programming/AiStudio/nexus-trader/nexus_packaged/outputs/MMFPS/generator_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 64
    num_steps = 25000
    lr = 5e-5
    gradient_accumulation = 1
    
    # Model
    config = DiffusionGeneratorConfig(
        in_channels=144,
        horizon=20,
        base_channels=256,
        channel_multipliers=(1, 2, 4, 8),
        time_dim=512,
        ctx_dim=144,
        regime_dim=64,
        quant_dim=64,
        num_timesteps=1000,
        num_paths=8,
        sampling_steps=50,
        target_std=0.25,  # Real market std
        target_mean=0.0,
    )
    
    model = ConstrainedDiffusionGenerator(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Resume from checkpoint if exists
    resume_path = checkpoint_dir / "checkpoint_step_18000.pt"
    resume_step = 0
    if resume_path.exists():
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        resume_step = ckpt['step']
        print(f"Loaded model from step {resume_step}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    remaining_steps = num_steps - resume_step
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=remaining_steps,
        pct_start=0.1,
    )
    scaler = GradScaler()
    
    # Dataset
    dataset = DiffusionDataset(data_path, seq_len=20, stride=1)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Training loop
    step = resume_step
    best_val_loss = float('inf')
    log_file = checkpoint_dir.parent / "train.log"
    
    print(f"Starting training for {num_steps} steps (resuming from {resume_step})...")
    print(f"Logging to {log_file}")
    
    train_iter = iter(train_loader)
    
    while step < num_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        metrics = train_one_step(model, optimizer, batch, device, scaler)
        scheduler.step()
        step += 1
        
        # Periodic memory cleanup
        if step % 500 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        
        # Logging
        if step % 10 == 0:
            print(f"Training: {step}/{num_steps} [{step/num_steps*100:.1f}%] | loss={metrics['loss']:.4f}", flush=True)
        
        # Validation
        if step % 500 == 0:
            val_metrics = validate(model, val_loader, device)
            print(f"Validation [{step}]: loss={val_metrics['loss']:.4f}, returns_std={val_metrics['returns_std']:.4f}, range=[{val_metrics['returns_p5']:.2f}, {val_metrics['returns_p95']:.2f}]", flush=True)
            
            # Save if better
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                checkpoint_path = checkpoint_dir / f"best_model.pt"
                torch.save({
                    'step': step,
                    'model': model.state_dict(),
                    'config': config,
                    'metrics': val_metrics,
                }, checkpoint_path)
                print(f"Saved best model to {checkpoint_path}")
            
            # Save regular checkpoint
            if step % 1000 == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
                torch.save({
                    'step': step,
                    'model': model.state_dict(),
                    'config': config,
                }, checkpoint_path)
    
    print("Training complete!")


if __name__ == "__main__":
    main()