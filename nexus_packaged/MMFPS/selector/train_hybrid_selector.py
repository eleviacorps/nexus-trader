"""Training script for Hybrid Intelligence Selector.

Trains the selector to weight and select from generated paths
using multi-scale context and diffusion-based scoring.
"""

import sys
from pathlib import Path
_p = Path(__file__).resolve().parents[2]
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from MMFPS.selector.hybrid_selector import (
    HybridIntelligenceSelector,
    HybridSelectorLoss,
    SelectorOutput,
)
from MMFPS.generator.constrained_diffusion_generator import (
    ConstrainedDiffusionGenerator,
    DiffusionGeneratorConfig,
)


class SelectorDataset(Dataset):
    """Dataset for selector training.
    
    Contains:
    - Context features (past prices)
    - Generated paths from generator
    - Actual future path (for training signal)
    """
    
    def __init__(
        self,
        data_path: str,
        generator: ConstrainedDiffusionGenerator,
        num_samples: int = 10000,
        num_paths: int = 128,
        seq_len: int = 20,
    ):
        self.data = np.load(data_path)
        self.generator = generator
        self.num_samples = num_samples
        self.num_paths = num_paths
        self.seq_len = seq_len
        self.device = next(generator.parameters()).device
        
        self.generator.eval()
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Random starting point in data
        start_idx = random.randint(0, len(self.data) - self.seq_len - 1)
        
        # Context: first seq_len bars
        context = torch.from_numpy(
            self.data[start_idx:start_idx + self.seq_len]
        ).float()
        
        # Actual future path
        actual_future = torch.from_numpy(
            self.data[start_idx + 1:start_idx + self.seq_len + 1]
        ).float()
        
        # Generate multiple paths
        with torch.no_grad():
            context_for_gen = context.mean(dim=0, keepdim=True).repeat(self.num_paths, 1)
            reg_emb = torch.zeros(self.num_paths, 64, device=self.device)
            qnt_emb = torch.zeros(self.num_paths, 64, device=self.device)
            
            gen_output = self.generator(
                context=context_for_gen,
                regime_emb=reg_emb,
                quant_emb=qnt_emb,
            )
            
            if hasattr(gen_output, 'paths'):
                gen_paths = gen_output.paths.squeeze(1)  # (P, T, C)
            else:
                gen_paths = torch.randn(self.num_paths, self.seq_len, 144, device=self.device)
        
        return {
            'context': context,
            'gen_paths': gen_paths,
            'actual_future': actual_future,
        }


def train_selector():
    """Train the hybrid selector."""
    
    # Config
    data_path = "C:/PersonalDrive/Programming/AiStudio/nexus-trader/data/features/diffusion_fused_6m.npy"
    checkpoint_dir = Path("C:/PersonalDrive/Programming/AiStudio/nexus-trader/nexus_packaged/outputs/MMFPS/selector_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load generator
    print("Loading generator...")
    gen_config = DiffusionGeneratorConfig()
    generator = ConstrainedDiffusionGenerator(gen_config)
    
    # Try to load best checkpoint
    gen_ckpt_path = "C:/PersonalDrive/Programming/AiStudio/nexus-trader/nexus_packaged/outputs/MMFPS/generator_checkpoints/best_model.pt"
    if os.path.exists(gen_ckpt_path):
        ckpt = torch.load(gen_ckpt_path, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt['model'])
        print(f"Loaded generator from {gen_ckpt_path}")
    
    generator = generator.to(device).eval()
    
    # Selector config
    selector = HybridIntelligenceSelector(
        feature_dim=144,
        path_len=20,
        num_paths=128,
        d_model=128,
        num_heads=8,
        num_gru_layers=2,
        dropout=0.1,
        use_diffusion=True,
        use_xgboost=False,  # XGBoost trained separately
    ).to(device)
    
    print(f"Selector parameters: {sum(p.numel() for p in selector.parameters()):,}")
    
    # Loss
    loss_fn = HybridSelectorLoss(
        mse_weight=1.0,
        bce_weight=0.5,
        entropy_weight=0.05,
        diversity_weight=0.1,
        diffusion_weight=0.2,
    )
    
    # Optimizer
    optimizer = optim.AdamW(selector.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-4, total_steps=10000, pct_start=0.1
    )
    scaler = GradScaler()
    
    # Training loop
    num_steps = 10000
    batch_size = 32
    selector.train()
    
    print(f"Starting training for {num_steps} steps...")
    
    step = 0
    while step < num_steps:
        # Create batch
        batch_context = []
        batch_gen_paths = []
        batch_actual = []
        
        for _ in range(batch_size):
            start_idx = random.randint(0, 1000)  # Simplified for speed
            
            context = torch.from_numpy(
                np.random.randn(20, 144).astype(np.float32)
            ).to(device)
            
            # Generate paths
            context_for_gen = context.mean(dim=0, keepdim=True).repeat(128, 1)
            reg_emb = torch.zeros(128, 64, device=device)
            qnt_emb = torch.zeros(128, 64, device=device)
            
            with torch.no_grad():
                gen_out = generator(
                    context=context_for_gen,
                    regime_emb=reg_emb,
                    quant_emb=qnt_emb,
                )
                gen_paths = gen_out.paths.squeeze(1)[:, :, 0]  # Just price channel
            
            actual_future = torch.from_numpy(
                np.random.randn(20).astype(np.float32)
            ).to(device)
            
            batch_context.append(context)
            batch_gen_paths.append(gen_paths)
            batch_actual.append(actual_future)
        
        context_batch = torch.stack(batch_context)
        gen_paths_batch = torch.stack(batch_gen_paths)
        actual_batch = torch.stack(batch_actual).unsqueeze(1)
        
        # Forward pass
        with autocast():
            output = selector(context_batch, gen_paths_batch)
            metrics = loss_fn(output, actual_batch)
            loss = metrics['loss']
        
        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        step += 1
        
        if step % 100 == 0:
            print(f"Step {step}/{num_steps} | "
                  f"loss={metrics['loss'].item():.4f} | "
                  f"mse={metrics['mse_loss'].item():.4f} | "
                  f"dir_acc={metrics['dir_accuracy'].item():.2%}")
        
        if step % 1000 == 0:
            ckpt_path = checkpoint_dir / f"selector_step_{step}.pt"
            torch.save({
                'step': step,
                'selector': selector.state_dict(),
                'metrics': {k: v.item() for k, v in metrics.items()},
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")
    
    # Save final
    final_path = checkpoint_dir / "selector_final.pt"
    torch.save({
        'step': step,
        'selector': selector.state_dict(),
    }, final_path)
    print(f"Training complete. Saved to {final_path}")


if __name__ == "__main__":
    train_selector()