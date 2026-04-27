"""Selector dataset - OPTIMIZED for 1-2 hour generation.

Key optimizations:
- sampling_steps = 4 (fast DDPM)
- num_paths = 16 per batch
- batch_size = 8 (memory-safe)
- mixed precision (FP16)
- augmentation 16x (vol + noise + mixing + regime)
"""

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path
_p = Path(__file__).resolve().parents[2]
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))

import numpy as np
import torch
from tqdm import tqdm
import time

from MMFPS.generator.constrained_diffusion_generator import ConstrainedDiffusionGenerator, DiffusionGeneratorConfig


def augment_batch(paths: torch.Tensor, context_vols: np.ndarray, aug_factor: int) -> torch.Tensor:
    """Augment paths with scaling/noise/mixing.
    
    Input: (B, P, T) -> Output: (B, P * aug_factor, T)
    """
    B, P, T = paths.shape
    aug_paths = []
    
    for i in range(B):
        base = paths[i]  # (16, 20)
        vol = context_vols[i]
        
        # Regime-aware scaling range
        if vol > 0.03:
            scale_range = (1.0, 1.6)
        else:
            scale_range = (0.7, 1.2)
        
        for _ in range(aug_factor):
            aug = base.clone()
            
            # 1. Volatility scaling
            scale = np.random.uniform(*scale_range)
            aug = aug * scale
            
            # 2. Noise injection
            aug = aug + torch.randn_like(aug) * 0.01
            
            # 3. Path mixing (40% chance)
            if np.random.random() < 0.4:
                other = base[np.random.randint(P)]
                alpha = np.random.uniform(0.7, 0.9)
                aug = alpha * aug + (1 - alpha) * other
            
            aug_paths.append(aug)
    
    return torch.stack(aug_paths).view(B, -1, T)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate selector dataset from diffusion generator checkpoints.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="C:/PersonalDrive/Programming/AiStudio/nexus-trader/nexus_packaged/outputs/MMFPS/selector_data",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="C:/PersonalDrive/Programming/AiStudio/nexus-trader/data/features/diffusion_fused_6m.npy",
    )
    parser.add_argument(
        "--generator-ckpt",
        type=str,
        default="C:/PersonalDrive/Programming/AiStudio/nexus-trader/nexus_packaged/outputs/MMFPS/generator_checkpoints/best_model.pt",
    )
    parser.add_argument("--sampling-steps", type=int, default=4)
    parser.add_argument("--num-paths", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed-samples", type=int, default=40000)
    parser.add_argument("--aug-factor", type=int, default=16)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--context-len", type=int, default=120)
    parser.add_argument("--path-len", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=5000)
    return parser.parse_args()


def main():
    args = parse_args()

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    
    SAMPLING_STEPS = int(args.sampling_steps)
    NUM_PATHS = int(args.num_paths)
    BATCH_SIZE = int(args.batch_size)
    SEED_SAMPLES = int(args.seed_samples)
    AUG_FACTOR = int(args.aug_factor)
    STRIDE = int(args.stride)
    CONTEXT = int(args.context_len)
    PATH_LEN = int(args.path_len)
    chunk_size = int(args.chunk_size)
    
    print(f"=== OPTIMIZED Dataset Generation ===")
    print(f"Seeds: {SEED_SAMPLES} | Batch: {BATCH_SIZE} | Paths: {NUM_PATHS}")
    print(f"Augmentation: {AUG_FACTOR}x | Total: {SEED_SAMPLES * NUM_PATHS * AUG_FACTOR / 1e6:.1f}M paths")
    print(f"Diffusion steps: {SAMPLING_STEPS}")
    print(f"Device: {device}")
    
    # Generator
    gen_config = DiffusionGeneratorConfig()
    gen_config.sampling_steps = SAMPLING_STEPS
    gen_config.num_paths = NUM_PATHS
    generator = ConstrainedDiffusionGenerator(gen_config).to(device)
    
    # Load checkpoint
    ckpt_path = Path(args.generator_ckpt)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    generator.load_state_dict(ckpt['model'])
    generator.eval()
    print(f"Generator loaded from {ckpt_path}")
    
    # Data
    data = np.load(args.data_path).astype(np.float32)
    print(f"Data: {len(data):,} samples")
    
    start = time.time()
    
    contexts_all = []
    paths_all = []
    futures_all = []
    
    autocast_ctx = torch.autocast("cuda", torch.float16) if device.type == "cuda" else nullcontext()
    with autocast_ctx, torch.no_grad():
        for batch_start in tqdm(range(0, SEED_SAMPLES, BATCH_SIZE), desc="Generating"):
            cur_batch = min(BATCH_SIZE, SEED_SAMPLES - batch_start)
            
            # Get valid indices
            indices = []
            for i in range(cur_batch):
                idx = CONTEXT + (batch_start + i) * STRIDE
                if idx + PATH_LEN < len(data):
                    indices.append(idx)
            cur_batch = len(indices)
            if cur_batch == 0:
                break
            
            # Prepare batch data - FULL context (NO averaging)
            batch_ctx = []
            batch_future = []
            batch_vol = []
            
            for idx in indices:
                batch_ctx.append(data[idx - CONTEXT:idx])  # (120, 144) FULL
                batch_future.append(data[idx:idx + PATH_LEN, 0])  # (20,)
                batch_vol.append(np.std(data[idx - CONTEXT:idx, 0]))  # volatility
            
            batch_ctx = torch.from_numpy(np.array(batch_ctx)).to(device)  # (B, 120, 144)
            batch_future = np.array(batch_future)
            batch_vol = np.array(batch_vol)
            
            # Generate paths with mixed precision
            ctx_exp = batch_ctx.mean(dim=1)  # (B, 144) - for envelope only
            emb_dtype = torch.float16 if device.type == "cuda" else torch.float32
            reg_emb = torch.zeros(cur_batch, 64, device=device, dtype=emb_dtype)
            quant_emb = torch.zeros(cur_batch, 64, device=device, dtype=emb_dtype)
            
            # Generate - output is (B, P, H, C)
            gen_paths = generator.generate_paths(
                context=ctx_exp,
                regime_emb=reg_emb,
                quant_emb=quant_emb,
                num_paths=NUM_PATHS,
            )
            # Extract returns: (B, P, H) - take first channel
            gen_paths = gen_paths[:, :, :, 0].contiguous()
            
            # Augment
            aug_paths = augment_batch(gen_paths, batch_vol, AUG_FACTOR)
            
            # Store
            contexts_all.extend(batch_ctx.cpu().numpy())
            futures_all.append(batch_future)
            paths_all.append(aug_paths.cpu())
            
            # Progress
            done = batch_start + cur_batch
            if done % 10000 == 0:
                elapsed = time.time() - start
                rate = done / elapsed
                print(f"\n{done}/{SEED_SAMPLES} | {rate:.1f}/sec | ETA: {(SEED_SAMPLES-done)/rate/60:.1f}min")
            
            # Save chunks every chunk_size samples
            if done % chunk_size == 0 and done > 0 and paths_all:
                chunk_idx = done // chunk_size
                np.savez_compressed(
                    OUTPUT_DIR / f"chunk_{chunk_idx:04d}.npz",
                    contexts=np.array(contexts_all),
                    paths=np.concatenate(paths_all, axis=0),
                    futures=np.concatenate(futures_all, axis=0)
                )
                print(f"\nSaved chunk {chunk_idx}")
                contexts_all = []
                paths_all = []
                futures_all = []
                torch.cuda.empty_cache()
    
    # Save any remaining samples
    if contexts_all and paths_all and futures_all:
        np.savez_compressed(
            OUTPUT_DIR / "final.npz",
            contexts=np.array(contexts_all),
            paths=np.concatenate(paths_all, axis=0),
            futures=np.concatenate(futures_all, axis=0)
        )
    
    total_time = time.time() - start
    total_paths = SEED_SAMPLES * NUM_PATHS * AUG_FACTOR
    
    print(f"\n=== DONE ===")
    print(f"Time: {total_time/60:.1f} min")
    print(f"Total paths: {total_paths/1e6:.1f}M")
    print(f"Rate: {total_paths/total_time/1e6:.2f}M paths/sec")


if __name__ == "__main__":
    main()
