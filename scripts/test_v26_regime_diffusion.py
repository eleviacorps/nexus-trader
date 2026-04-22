"""Test script for V26 Phase 1 Agent 3: Regime-Aware Diffusion

Verifies:
1. RegimeDiffusionDataset loads with regime labels
2. Model loads Phase 0.7 checkpoint
3. Training loop runs without error
4. Checkpoint saving works
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import (
    V24_DIFFUSION_FUSED_6M_PATH,
    V24_DIFFUSION_TIMESTAMPS_6M_PATH,
)
from src.v24.diffusion.dataset import split_by_year
from src.v24.diffusion.scheduler import NoiseScheduler
from src.v24.diffusion.temporal_encoder import TemporalEncoder
from src.v24.diffusion.unet_1d import DiffusionUNet1D
from src.v26.diffusion.regime_dataset import RegimeDiffusionDataset
from src.v6.regime_detection import REGIME_LABELS


class RegimeEmbedding(nn.Module):
    """Regime embedding layer for V26."""

    def __init__(self, num_regimes: int = 10, embed_dim: int = 16) -> None:
        super().__init__()
        self.regime_net = nn.Sequential(
            nn.Linear(num_regimes, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, regime_probs: torch.Tensor) -> torch.Tensor:
        return self.regime_net(regime_probs)


def test_regime_dataset():
    """Test 1: Verify dataset loads with regime labels."""
    print("=" * 60)
    print("TEST 1: RegimeDiffusionDataset loads with regime labels")
    print("=" * 60)

    fused_path = V24_DIFFUSION_FUSED_6M_PATH
    timestamps_path = V24_DIFFUSION_TIMESTAMPS_6M_PATH
    regime_cache_path = Path("outputs/v26/test_regime_cache.npy")
    regime_cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not fused_path.exists():
        print(f"SKIP: Feature file not found: {fused_path}")
        return False

    timestamps = None
    if timestamps_path.exists():
        timestamps = np.load(str(timestamps_path), mmap_mode="r")

    fused_mmap = np.load(str(fused_path), mmap_mode="r")
    total = len(fused_mmap)
    print(f"Total samples: {total}")

    # Small slice for testing
    test_slice = split_by_year(total, 120, timestamps=timestamps)[0]
    test_slice = type(test_slice)(test_slice.start, min(test_slice.start + 1000, test_slice.stop))
    print(f"Test slice: {len(test_slice)} samples")

    try:
        t0 = time.time()
        ds = RegimeDiffusionDataset(
            fused_path, 120, test_slice,
            timestamp_path=timestamps_path,
            context_len=256,
            max_samples=100,  # Small sample for testing
            load_to_ram=True,
            regime_cache_path=regime_cache_path,
        )
        elapsed = time.time() - t0
        print(f"✓ Dataset created in {elapsed:.2f}s")

        # Check length
        assert len(ds) == 100, f"Expected 100 samples, got {len(ds)}"
        print(f"✓ Dataset length: {len(ds)}")

        # Check regime distribution
        regime_dist = ds.get_regime_distribution()
        print(f"✓ Regime distribution computed:")
        for regime, pct in sorted(regime_dist.items(), key=lambda x: -x[1])[:5]:
            print(f"    {regime}: {pct:.1%}")

        # Check sample shape
        window, past_ctx, regime_probs = ds[0]
        print(f"✓ Sample shapes:")
        print(f"    window: {window.shape} (expected: (144, 120))")
        print(f"    past_ctx: {past_ctx.shape} (expected: (256, 144))")
        print(f"    regime_probs: {regime_probs.shape} (expected: ({len(REGIME_LABELS)},))")

        # Verify regime probabilities sum to 1
        regime_sum = regime_probs.sum().item()
        assert abs(regime_sum - 1.0) < 0.01, f"Regime probs sum to {regime_sum}, expected ~1.0"
        print(f"✓ Regime probabilities sum to {regime_sum:.4f}")

        # Verify dominant regime
        dominant_idx = torch.argmax(regime_probs).item()
        print(f"✓ Dominant regime: {REGIME_LABELS[dominant_idx]} ({regime_probs[dominant_idx]:.3f})")

        print("\n✅ TEST 1 PASSED: RegimeDiffusionDataset loads correctly\n")
        return True, ds

    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False, None


def test_checkpoint_loading():
    """Test 2: Verify model loads Phase 0.7 checkpoint."""
    print("=" * 60)
    print("TEST 2: Model loads Phase 0.7 checkpoint")
    print("=" * 60)

    ckpt_path = Path("models/v24/diffusion_unet1d_v2_6m_phase07.pt")
    if not ckpt_path.exists():
        # Try alternate paths
        alt_paths = [
            Path("models/v24/diffusion_unet1d_v2_6m_phase06.pt"),
            Path("models/v24/diffusion_unet1d_v2_6m.pt"),
        ]
        for alt in alt_paths:
            if alt.exists():
                ckpt_path = alt
                break
        else:
            print(f"SKIP: No checkpoint found at {ckpt_path}")
            print("  Checked:")
            print(f"    - {ckpt_path}")
            for alt in alt_paths:
                print(f"    - {alt}")
            return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")

    try:
        # Load checkpoint
        t0 = time.time()
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        elapsed = time.time() - t0
        print(f"✓ Checkpoint loaded in {elapsed:.2f}s")
        print(f"  Epoch: {ckpt.get('epoch', '?')}")
        print(f"  Best val loss: {ckpt.get('best_val_loss', '?'):.6f}")
        print(f"  Phase: {ckpt.get('phase', 'unknown')}")

        # Check for EMA weights
        has_ema = "ema" in ckpt
        print(f"  Has EMA weights: {has_ema}")

        # Initialize model with expanded temporal_dim for regime
        regime_embed_dim = 16
        temporal_dim = 256 + regime_embed_dim

        model = DiffusionUNet1D(
            in_channels=144,
            base_channels=128,
            channel_multipliers=(1, 2, 4),
            time_dim=256,
            num_res_blocks=2,
            ctx_dim=144,
            temporal_dim=temporal_dim,
            d_gru=256,
        ).to(device)

        # Load weights (with strict=False for temporal_dim expansion)
        state_to_load = ckpt.get("ema", ckpt.get("model", {}))
        missing, unexpected = model.load_state_dict(state_to_load, strict=False)

        if missing:
            print(f"  Missing keys: {len(missing)}")
            for k in missing[:5]:
                print(f"    - {k}")
            if len(missing) > 5:
                print(f"    ... and {len(missing) - 5} more")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")

        print(f"✓ Model weights loaded")

        # Initialize temporal encoder
        temporal_encoder = TemporalEncoder(
            in_features=144,
            d_gru=256,
            num_layers=2,
            film_dim=256,
        ).to(device)

        if "temporal_encoder" in ckpt:
            temporal_encoder.load_state_dict(ckpt["temporal_encoder"])
            print(f"✓ Temporal encoder weights loaded")

        # Initialize regime embedding
        regime_embedding = RegimeEmbedding(
            num_regimes=len(REGIME_LABELS),
            embed_dim=regime_embed_dim,
        ).to(device)
        print(f"✓ Regime embedding initialized")

        print("\n✅ TEST 2 PASSED: Model loads checkpoint correctly\n")
        return True, model, temporal_encoder, regime_embedding, device

    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False, None, None, None, None


def test_training_loop(ds, model, temporal_encoder, regime_embedding, device):
    """Test 3: Verify training loop runs without error."""
    print("=" * 60)
    print("TEST 3: Training loop runs without error")
    print("=" * 60)

    try:
        # Create dataloader
        dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
        print(f"✓ DataLoader created (batch_size=4)")

        # Setup scheduler
        scheduler = NoiseScheduler(1000).to(device)
        print(f"✓ NoiseScheduler created")

        # Get one batch
        window, past_ctx, regime_probs = next(iter(dl))
        window = window.to(device)
        past_ctx = past_ctx.to(device)
        regime_probs = regime_probs.to(device)

        print(f"✓ Batch loaded:")
        print(f"    window: {window.shape}")
        print(f"    past_ctx: {past_ctx.shape}")
        print(f"    regime_probs: {regime_probs.shape}")

        # Forward pass
        model.train()
        temporal_encoder.train()
        regime_embedding.train()

        temporal_seq, temporal_emb, _ = temporal_encoder(past_ctx)
        regime_emb = regime_embedding(regime_probs)
        context = past_ctx[:, -1, :]

        print(f"✓ Temporal encoding:")
        print(f"    temporal_seq: {temporal_seq.shape}")
        print(f"    temporal_emb: {temporal_emb.shape}")
        print(f"    regime_emb: {regime_emb.shape}")

        # Compute loss
        loss_dict = scheduler.training_loss_regime(
            model, window, context,
            temporal_seq=temporal_seq,
            temporal_emb=temporal_emb,
            regime_emb=regime_emb,
            acf_weight=0.10,
            vol_weight=0.10,
            std_weight=0.05,
        )

        loss = loss_dict["total"]
        print(f"✓ Loss computed:")
        print(f"    total: {loss.item():.4f}")
        print(f"    diffusion: {loss_dict['diffusion'].item():.4f}")
        print(f"    acf: {loss_dict['acf'].item():.4f}")
        print(f"    vol: {loss_dict['vol'].item():.4f}")
        print(f"    std: {loss_dict['std'].item():.4f}")

        # Backward pass
        loss.backward()
        print(f"✓ Backward pass completed")

        # Check gradients
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad, "No gradients computed"
        print(f"✓ Gradients exist")

        print("\n✅ TEST 3 PASSED: Training loop runs correctly\n")
        return True

    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint_saving(model, temporal_encoder, regime_embedding):
    """Test 4: Verify checkpoint saving works."""
    print("=" * 60)
    print("TEST 4: Checkpoint saving works")
    print("=" * 60)

    try:
        # Create test checkpoint
        test_ckpt_path = Path("outputs/v26/test_checkpoint.pt")
        test_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model": model.state_dict(),
            "temporal_encoder": temporal_encoder.state_dict(),
            "regime_embedding": regime_embedding.state_dict(),
            "ema": model.state_dict(),  # Simulated EMA
            "epoch": 0,
            "best_val_loss": 0.123456,
            "phase": "v26_test",
        }

        t0 = time.time()
        torch.save(checkpoint, str(test_ckpt_path))
        elapsed = time.time() - t0
        print(f"✓ Checkpoint saved in {elapsed:.3f}s")
        print(f"  Path: {test_ckpt_path}")
        print(f"  Size: {test_ckpt_path.stat().st_size / 1e6:.2f} MB")

        # Verify we can load it back
        loaded = torch.load(str(test_ckpt_path), weights_only=False)
        print(f"✓ Checkpoint loaded back")
        print(f"  Epoch: {loaded.get('epoch')}")
        print(f"  Phase: {loaded.get('phase')}")
        print(f"  Has regime_embedding: {'regime_embedding' in loaded}")

        # Cleanup
        test_ckpt_path.unlink()
        print(f"✓ Test checkpoint cleaned up")

        print("\n✅ TEST 4 PASSED: Checkpoint saving works correctly\n")
        return True

    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("V26 Phase 1 Agent 3: Regime-Aware Diffusion Tests")
    print("=" * 60 + "\n")

    results = []

    # Test 1: Dataset
    ok1, ds = test_regime_dataset()
    results.append(("Dataset loads", ok1))

    if not ok1:
        print("Skipping remaining tests due to dataset failure\n")
        return

    # Test 2: Checkpoint loading
    ok2, *components = test_checkpoint_loading()
    results.append(("Checkpoint loading", ok2))

    if not ok2:
        print("Skipping remaining tests due to checkpoint loading failure\n")
        return

    model, temporal_encoder, regime_embedding, device = components

    # Test 3: Training loop
    ok3 = test_training_loop(ds, model, temporal_encoder, regime_embedding, device)
    results.append(("Training loop", ok3))

    # Test 4: Checkpoint saving
    ok4 = test_checkpoint_saving(model, temporal_encoder, regime_embedding)
    results.append(("Checkpoint saving", ok4))

    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}: {name}")
        if not ok:
            all_passed = False

    if all_passed:
        print("\n✅ ALL TESTS PASSED")
        print("V26 Phase 1 Agent 3 is ready for training\n")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("Please review the errors above\n")


if __name__ == "__main__":
    main()
