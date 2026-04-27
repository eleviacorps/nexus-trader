"""Simple direct path generator with proper diversity.

ARCHITECTURE:
- Direct MLP predicting paths from context
- Per-path independent noise injection
- Distribution matching loss (NOT maximization)
"""

from __future__ import annotations

import sys
from pathlib import Path
_p = Path(__file__).resolve().parents[2]
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))

from dataclasses import dataclass
from typing import Optional, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GeneratorOutput(NamedTuple):
    paths: Tensor
    diversity_loss: Tensor


@dataclass
class DirectGeneratorConfig:
    """Configuration for direct generator."""
    in_channels: int = 144
    horizon: int = 20
    hidden_dim: int = 512
    num_paths: int = 128
    
    # Target distribution (realistic values)
    target_std: float = 0.05   # ~5% std
    target_mean: float = 0.0


class DirectPathGenerator(nn.Module):
    """Direct multi-path generator with independent noise per path."""
    
    def __init__(
        self,
        config: Optional[DirectGeneratorConfig] = None,
    ) -> None:
        super().__init__()
        
        self.config = config or DirectGeneratorConfig()
        cfg = self.config
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(cfg.in_channels, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
        )
        
        # Path head - predicts all paths at once
        self.path_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.horizon * cfg.in_channels),
        )
        
        # Register targets
        self.register_buffer('target_std', torch.tensor(cfg.target_std))
        self.register_buffer('target_mean', torch.tensor(cfg.target_mean))
    
    def forward(
        self,
        context: Tensor,
        num_paths: Optional[int] = None,
    ) -> GeneratorOutput:
        """Generate multiple paths from context.
        
        Args:
            context: (B, in_channels) - context features
            
        Returns:
            GeneratorOutput with paths and diversity loss
        """
        B = context.shape[0]
        n_paths = num_paths or self.config.num_paths
        
        # Encode context
        h = self.context_encoder(context)  # (B, hidden)
        
        # Generate base prediction
        base = self.path_head(h)  # (B, horizon * in_channels)
        base = base.view(B, self.config.horizon, self.config.in_channels)
        
        # Inject per-path noise for diversity
        noise = torch.randn(B, n_paths, self.config.horizon, self.config.in_channels, 
                       device=context.device, dtype=context.dtype)
        
        # Scale noise by target_std
        paths = base.unsqueeze(1) + noise * self.target_std
        
        # Compute diversity loss (distribution matching NOT maximization)
        returns = paths[:, :, -1, 0]  # final returns per path
        ret_std = returns.std(dim=1).mean()
        ret_mean = returns.mean()
        
        std_loss = (ret_std - self.target_std) ** 2
        mean_loss = (ret_mean - self.target_mean) ** 2
        
        magnitude_penalty = (paths ** 2).mean() * 0.01
        
        diversity_loss = std_loss + mean_loss + magnitude_penalty
        
        return GeneratorOutput(paths=paths, diversity_loss=diversity_loss)
    
    def quick_generate(
        self,
        context: Tensor,
        num_paths: int = 32,
    ) -> Tensor:
        """Quick generation."""
        output = self.forward(context, num_paths)
        return output.paths


def train_direct_generator():
    """Train the direct generator."""
    import numpy as np
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    
    data_path = "C:/PersonalDrive/Programming/AiStudio/nexus-trader/data/features/diffusion_fused_6m.npy"
    checkpoint_dir = Path("C:/PersonalDrive/Programming/AiStudio/nexus-trader/nexus_packaged/outputs/MMFPS/generator_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Config
    config = DirectGeneratorConfig(
        in_channels=144,
        horizon=20,
        hidden_dim=256,
        num_paths=8,
        target_std=0.05,
        target_mean=0.0,
    )
    
    model = DirectPathGenerator(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    # Dataset - sample a subset for faster training
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    # Use subset for faster training
    data = data[:500000]
    print(f"Using {len(data)} samples")
    
    # Simple batching
    batch_size = 256
    num_steps = 10000
    
    model.train()
    step = 0
    best_loss = float('inf')
    
    while step < num_steps:
        # Sample random batches
        idx = np.random.randint(0, len(data), batch_size)
        batch = torch.from_numpy(data[idx]).float().to(device)
        
        # Forward
        output = model(batch)
        loss = output.diversity_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        step += 1
        
        if step % 100 == 0:
            print(f"Training: {step}/{num_steps} | loss={loss.item():.4f}")
        
        if step % 500 == 0 and loss.item() < best_loss:
            best_loss = loss.item()
            torch.save({
                'step': step,
                'model': model.state_dict(),
                'config': config,
            }, checkpoint_dir / "best_model.pt")
            print(f"Saved best model at step {step}")
    
    print("Training complete!")
    
    # Test generation
    model.eval()
    test_context = torch.randn(4, 144, device=device)
    with torch.no_grad():
        output = model(test_context, num_paths=32)
        paths = output.paths
        
        first_ch = paths[:, :, :, 0].cpu().numpy()
        print(f"=== GENERATED PATHS ===")
        print(f"Shape: {first_ch.shape}")
        for t in [0, 9, 19]:
            vals = first_ch[:, :, t].flatten()
            print(f"T={t}: mean={vals.mean():.4f}, std={vals.std():.4f}, range=[{vals.min():.2%}, {vals.max():.2%}]")


if __name__ == "__main__":
    train_direct_generator()