"""Dataset generation for selector training.

Generates (context, 128_paths, real_future) tuples from 2009-2026 data.
Uses batching for fast generation.
"""

import sys
from pathlib import Path
_p = Path(__file__).resolve().parents[2]
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time

from MMFPS.generator.constrained_diffusion_generator import (
    ConstrainedDiffusionGenerator,
    DiffusionGeneratorConfig,
)


def load_training_data():
    """Load the fused data from 2009-2026."""
    data_path = "C:/PersonalDrive/Programming/AiStudio/nexus-trader/data/features/diffusion_fused_6m.npy"
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    print(f"Data shape: {data.shape}")
    print(f"Date range: index 0 to {len(data)}")
    return data


def generate_selector_dataset(
    data: np.ndarray,
    generator: ConstrainedDiffusionGenerator,
    output_path: str,
    num_samples: int = 50000,
    context_len: int = 120,
    path_len: int = 20,
    num_paths: int = 128,
    batch_size: int = 64,
    start_idx: int = 120,
):
    """Generate dataset for selector training.
    
    Args:
        data: Full dataset (N, 144)
        generator: Trained generator
        output_path: Path to save dataset
        num_samples: Number of samples to generate
        context_len: Context length (lookback)
        path_len: Future path length
        num_paths: Number of paths to generate per sample
        batch_size: Batch size for generation
        start_idx: Starting index in data
    """
    device = next(generator.parameters()).device
    generator.eval()
    
    print(f"Generating {num_samples} samples with {num_paths} paths each...")
    print(f"Context length: {context_len}, Path length: {path_len}")
    
    # Storage for dataset
    contexts = []
    all_paths = []
    real_futures = []
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    samples_generated = 0
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Generating"):
            current_batch_size = min(batch_size, num_samples - samples_generated)
            
            # Sample random starting indices
            max_start = len(data) - context_len - path_len - 1
            start_indices = np.random.randint(start_idx, max_start, size=current_batch_size)
            
            # Prepare batch
            batch_context = []
            batch_real_future = []
            
            for i in range(current_batch_size):
                start = start_indices[i]
                
                # Context: context_len bars before the future
                context = data[start - context_len:start]  # (context_len, 144)
                
                # Real future: path_len bars after the start
                real_future = data[start:start + path_len, 0]  # Just price channel (path_len,)
                
                batch_context.append(context)
                batch_real_future.append(real_future)
            
            # Stack
            context_batch = torch.from_numpy(np.stack(batch_context)).float().to(device)
            real_future_batch = torch.from_numpy(np.stack(batch_real_future)).float().to(device)
            
            # Generate paths for each sample in batch
            for i in range(current_batch_size):
                ctx = context_batch[i:i+1].mean(dim=1).repeat(num_paths, 1)  # (P, 144)
                reg_emb = torch.zeros(num_paths, 64, device=device)
                qnt_emb = torch.zeros(num_paths, 64, device=device)
                
                gen_out = generator(
                    context=ctx,
                    regime_emb=reg_emb,
                    quant_emb=qnt_emb,
                )
                
                # Extract paths - get just the price channel
                gen_paths = gen_out.paths.squeeze(1)[:, :, 0].cpu().numpy()  # (P, T)
                
                contexts.append(context_batch[i].cpu().numpy())
                all_paths.append(gen_paths)
                real_futures.append(real_future_batch[i].cpu().numpy())
                
                samples_generated += 1
                
                if samples_generated % 1000 == 0:
                    print(f"Generated {samples_generated}/{num_samples}...")
    
    # Convert to arrays
    contexts = np.array(contexts, dtype=np.float32)  # (N, context_len, 144)
    all_paths = np.array(all_paths, dtype=np.float32)  # (N, num_paths, path_len)
    real_futures = np.array(real_futures, dtype=np.float32)  # (N, path_len)
    
    # Save
    print(f"Saving to {output_path}...")
    np.savez_compressed(
        output_path,
        contexts=contexts,
        paths=all_paths,
        real_futures=real_futures,
    )
    
    print(f"Dataset saved!")
    print(f"  Contexts: {contexts.shape}")
    print(f"  Paths: {all_paths.shape}")
    print(f"  Real futures: {real_futures.shape}")
    
    return contexts, all_paths, real_futures


def generate_dataset_fast(
    data: np.ndarray,
    generator: ConstrainedDiffusionGenerator,
    output_path: str,
    num_samples: int = 50000,
    context_len: int = 120,
    path_len: int = 20,
    num_paths: int = 128,
    batch_size: int = 32,
    start_idx: int = 120,
):
    """Faster dataset generation using larger batches.
    
    Generates all paths in parallel for efficiency.
    """
    device = next(generator.parameters()).device
    generator.eval()
    
    print(f"Generating {num_samples} samples with {num_paths} paths each...")
    print(f"Context length: {context_len}, Path length: {path_len}")
    print(f"Batch size: {batch_size}")
    
    # Pre-allocate arrays
    contexts = np.zeros((num_samples, context_len, 144), dtype=np.float32)
    all_paths = np.zeros((num_samples, num_paths, path_len), dtype=np.float32)
    real_futures = np.zeros((num_samples, path_len), dtype=np.float32)
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    samples_generated = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Generating"):
            current_batch_size = min(batch_size, num_samples - samples_generated)
            
            # Sample random starting indices
            max_start = len(data) - context_len - path_len - 1
            start_indices = np.random.randint(start_idx, max_start, size=current_batch_size)
            
            # Prepare batch
            batch_context = []
            batch_real_future = []
            
            for i in range(current_batch_size):
                start = start_indices[i]
                
                context = data[start - context_len:start]  # (context_len, 144)
                real_future = data[start:start + path_len, 0]  # Just price channel
                
                batch_context.append(context)
                batch_real_future.append(real_future)
            
            context_batch = torch.from_numpy(np.stack(batch_context)).float().to(device)
            real_future_batch = torch.from_numpy(np.stack(batch_real_future)).float().to(device)
            
            # Generate all paths for batch at once
            # Reshape context for parallel generation
            context_for_gen = context_batch.mean(dim=1, keepdim=True).repeat(1, num_paths, 1)
            context_for_gen = context_for_gen.view(current_batch_size * num_paths, 144)
            
            reg_emb = torch.zeros(current_batch_size * num_paths, 64, device=device)
            qnt_emb = torch.zeros(current_batch_size * num_paths, 64, device=device)
            
            gen_out = generator(
                context=context_for_gen,
                regime_emb=reg_emb,
                quant_emb=qnt_emb,
            )
            
            # Reshape paths back
            gen_paths = gen_out.paths  # (B*P, 1, T, C)
            gen_paths = gen_paths.squeeze(1)[:, :, 0].cpu().numpy()  # (B*P, T)
            gen_paths = gen_paths.reshape(current_batch_size, num_paths, path_len)
            
            # Store
            contexts[samples_generated:samples_generated + current_batch_size] = context_batch.cpu().numpy()
            all_paths[samples_generated:samples_generated + current_batch_size] = gen_paths
            real_futures[samples_generated:samples_generated + current_batch_size] = real_future_batch.cpu().numpy()
            
            samples_generated += current_batch_size
            
            if samples_generated % 5000 == 0:
                elapsed = time.time() - start_time
                rate = samples_generated / elapsed
                remaining = (num_samples - samples_generated) / rate
                print(f"Generated {samples_generated}/{num_samples} | {rate:.1f} samples/sec | ETA: {remaining/60:.1f} min")
    
    # Save
    print(f"\nSaving to {output_path}...")
    np.savez_compressed(
        output_path,
        contexts=contexts,
        paths=all_paths,
        real_futures=real_futures,
    )
    
    total_time = time.time() - start_time
    print(f"Dataset saved in {total_time/60:.1f} minutes!")
    print(f"  Contexts: {contexts.shape}")
    print(f"  Paths: {all_paths.shape}")
    print(f"  Real futures: {real_futures.shape}")
    
    return contexts, all_paths, real_futures


def main():
    # Paths
    output_dir = Path("C:/PersonalDrive/Programming/AiStudio/nexus-trader/nexus_packaged/outputs/MMFPS/selector_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "selector_dataset.npz"
    
    # Load generator
    print("Loading generator...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    gen_config = DiffusionGeneratorConfig()
    generator = ConstrainedDiffusionGenerator(gen_config).to(device)
    
    # Load trained checkpoint
    ckpt_path = "C:/PersonalDrive/Programming/AiStudio/nexus-trader/nexus_packaged/outputs/MMFPS/generator_checkpoints/best_model.pt"
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt['model'])
        print(f"Loaded checkpoint from step {ckpt.get('step', 'unknown')}")
    else:
        print(f"WARNING: No checkpoint found at {ckpt_path}, using untrained generator")
    
    # Load data
    data = load_training_data()
    
    # Generate dataset
    num_samples = 50000  # 50K samples
    batch_size = 8       # Small batch to avoid OOM
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    generate_dataset_fast(
        data=data,
        generator=generator,
        output_path=str(output_path),
        num_samples=num_samples,
        context_len=120,
        path_len=20,
        num_paths=128,
        batch_size=batch_size,
        start_idx=120,
    )
    
    print("\n=== Dataset generation complete ===")


if __name__ == "__main__":
    main()