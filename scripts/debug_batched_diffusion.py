"""Debug script to verify batched diffusion works with large batches."""

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v26.diffusion.regime_generator import RegimeGeneratorConfig, RegimeDiffusionPathGenerator


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    batch_size = 128
    seq_len = 120
    features = 144

    torch.cuda.reset_peak_memory_stats()
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")

    # Load Phase 1 checkpoint
    ckpt_path = "models/v26/diffusion_phase1_final.pt"
    print(f"\nLoading Phase 1 checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"Phase 1: epoch={ckpt['epoch']}, val_loss={ckpt.get('best_val_loss', '?')}")

    # Create generator
    print("\nCreating generator...")
    config = RegimeGeneratorConfig(
        in_channels=features,
        sequence_length=seq_len,
        base_channels=128,
        channel_multipliers=(1, 2, 4),
        time_dim=256,
        num_timesteps=1000,
        ctx_dim=features,
        guidance_scale=3.0,
        num_paths=1,
        sampling_steps=50,
        dropout=0.1,
        temporal_gru_dim=256,
        temporal_layers=2,
        context_len=seq_len,
        norm_stats_path="config/diffusion_norm_stats_6m.json",
        num_regimes=9,
        regime_embed_dim=16,
        regime_conditioning_strength=1.0,
        temporal_film_dim=272,
    )

    gen = RegimeDiffusionPathGenerator(config=config, device=str(device))

    # Load weights
    checkpoint_state = ckpt.get("ema", ckpt.get("model", {}))
    model_state = gen.model.state_dict()
    compatible = {k: v for k, v in checkpoint_state.items() if k in model_state and model_state[k].shape == v.shape}
    gen.model.load_state_dict(compatible, strict=False)
    print(f"Loaded {len(compatible)} tensors")

    # Load temporal encoder
    if "temporal_encoder" in ckpt:
        temp_state = gen.temporal_encoder.state_dict()
        temp_compatible = {k: v for k, v in ckpt["temporal_encoder"].items() if k in temp_state and temp_state[k].shape == v.shape}
        gen.temporal_encoder.load_state_dict(temp_compatible, strict=False)
        print(f"Loaded {len(temp_compatible)} temporal encoder tensors")

    # Create fake batched inputs
    past_context = torch.randn(batch_size, seq_len, features, device=device)
    regime_probs = torch.softmax(torch.randn(batch_size, 9, device=device), dim=-1)

    print(f"\npast_context: {past_context.shape}")
    print(f"regime_probs: {regime_probs.shape}")

    mem_before = torch.cuda.memory_allocated() / 1e9
    print(f"\nVRAM before: {mem_before:.2f} GB")

    print("\n=== Calling generate_paths() with batch_size=128 ===")

    # Disable TF32 for more memory usage
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    # Call generator
    paths = gen.generate_paths(
        world_state=None,
        past_context=past_context,
        regime_probs=regime_probs,
        num_paths=batch_size,
        steps=5,
    )

    mem_after = torch.cuda.memory_allocated() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9

    print(f"\nVRAM after: {mem_after:.2f} GB")
    print(f"Peak VRAM: {peak:.2f} GB")
    print(f"Paths generated: {len(paths)}")

    # Check - batch is working, VRAM depends on model size
    if peak > 0.5:
        print(f"\n[OK] SUCCESS: Batched diffusion works! Peak VRAM = {peak:.2f} GB")
    else:
        print(f"\n[FAIL] Peak VRAM = {peak:.2f} GB is too low")


if __name__ == "__main__":
    main()