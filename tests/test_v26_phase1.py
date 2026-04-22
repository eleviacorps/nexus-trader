"""Test script for V26 Phase 1: Regime Embedding + Generator Extension."""

from __future__ import annotations

import torch
from torch import Tensor

def test_regime_embedding():
    """Test RegimeEmbedding module standalone."""
    print("=" * 60)
    print("TEST 1: RegimeEmbedding Standalone")
    print("=" * 60)

    from src.v26.diffusion.regime_embedding import RegimeEmbedding

    # Test 1.1: Basic embedding
    print("\n[1.1] Testing basic embedding...")
    emb = RegimeEmbedding(num_regimes=9, embed_dim=16, use_learned_embedding=False)
    regime_probs = torch.softmax(torch.randn(4, 9), dim=-1)  # Batch of 4, normalized
    output = emb(regime_probs)
    assert output.shape == (4, 16), f"Expected (4, 16), got {output.shape}"
    print(f"[PASS] Output shape: {output.shape} (expected (4, 16))")
    print(f"[INFO] Sample values: {output[0, :5].tolist()}")

    # Test 1.2: Single vector input
    print("\n[1.2] Testing single vector input...")
    single_regime = torch.softmax(torch.randn(9), dim=-1)
    output_single = emb(single_regime)
    assert output_single.shape == (16,), f"Expected (16,), got {output_single.shape}"
    print(f"[PASS] Single vector output shape: {output_single.shape} (expected (16,))")

    # Test 1.3: With learned embedding
    print("\n[1.3] Testing with learned embedding...")
    emb_learned = RegimeEmbedding(num_regimes=9, embed_dim=16, use_learned_embedding=True)
    output_learned = emb_learned(regime_probs)
    assert output_learned.shape == (4, 16), f"Expected (4, 16), got {output_learned.shape}"
    print(f"[PASS] Learned embedding output shape: {output_learned.shape}")

    # Test 1.4: From class indices
    print("\n[1.4] Testing forward_from_class...")
    regime_classes = torch.tensor([0, 3, 5, 8])
    output_class = emb.forward_from_class(regime_classes)
    assert output_class.shape == (4, 16), f"Expected (4, 16), got {output_class.shape}"
    print(f"[PASS] Class-based embedding output shape: {output_class.shape}")

    print("\n[OK] RegimeEmbedding tests passed!")
    return True


def test_regime_generator():
    """Test RegimeDiffusionPathGenerator."""
    print("\n" + "=" * 60)
    print("TEST 2: RegimeDiffusionPathGenerator")
    print("=" * 60)

    from src.v26.diffusion.regime_generator import (
        RegimeDiffusionPathGenerator,
        RegimeGeneratorConfig,
    )

    # Test 2.1: Basic initialization
    print("\n[2.1] Testing generator initialization...")
    config = RegimeGeneratorConfig(
        in_channels=100,
        sequence_length=64,
        base_channels=64,
        channel_multipliers=(1, 2),
        num_paths=4,
        sampling_steps=10,
        temporal_gru_dim=0,  # Disable temporal for simplicity
    )
    generator = RegimeDiffusionPathGenerator(config, device="cpu")
    print(f"[PASS] Generator initialized with {config.in_channels} channels")
    print(f"[PASS] Regime embedder: {config.num_regimes} -> {config.regime_embed_dim}")

    # Test 2.2: Generate paths with regime
    print("\n[2.2] Testing path generation with regime conditioning...")
    world_state = {"feature_0": 1.0, "feature_1": 0.5, "feature_2": -0.3}
    regime_probs = torch.softmax(torch.randn(9), dim=-1)  # Single regime probs

    paths = generator.generate_paths(
        world_state=world_state,
        regime_probs=regime_probs,
        num_paths=4,
    )
    assert len(paths) == 4, f"Expected 4 paths, got {len(paths)}"
    print(f"[PASS] Generated {len(paths)} paths")
    print(f"[PASS] Path length: {paths[0]['metadata']['path_length']}")
    print(f"[PASS] Regime conditioned: {paths[0]['regime']['conditioned']}")
    print(f"[PASS] Generator type: {paths[0]['metadata']['generator']}")

    # Test 2.3: Batch regime probabilities
    print("\n[2.3] Testing batch regime probabilities...")
    world_state = {"feature_0": 0.0}
    batch_regime_probs = torch.softmax(torch.randn(4, 9), dim=-1)  # Batch of 4
    paths_batch = generator.generate_paths(
        world_state=world_state,
        regime_probs=batch_regime_probs,
        num_paths=4,
    )
    assert len(paths_batch) == 4
    print(f"[PASS] Batch regime generation successful")

    # Test 2.4: Without regime (backward compat)
    print("\n[2.4] Testing without regime (backward compatibility)...")
    paths_no_regime = generator.generate_paths(
        world_state=world_state,
        num_paths=4,
    )
    assert len(paths_no_regime) == 4
    print(f"[PASS] Generation without regime conditioning works")
    print(f"[PASS] Regime conditioned: {paths_no_regime[0]['regime']['conditioned']}")

    print("\n[OK] RegimeDiffusionPathGenerator tests passed!")
    return True


def test_unet_regime_conditioning():
    """Test U-Net with regime conditioning."""
    print("\n" + "=" * 60)
    print("TEST 3: U-Net Regime Conditioning")
    print("=" * 60)

    from src.v24.diffusion.unet_1d import DiffusionUNet1D

    # Test 3.1: U-Net with regime_dim
    print("\n[3.1] Testing U-Net with regime_dim...")
    model = DiffusionUNet1D(
        in_channels=100,
        base_channels=64,
        channel_multipliers=(1, 2),
        time_dim=256,
        temporal_dim=64,
        regime_dim=16,  # Enable regime conditioning
        ctx_dim=100,
    )
    print(f"[PASS] Model initialized with regime_dim=16")

    # Test 3.2: Forward pass with regime
    print("\n[3.2] Testing forward pass with regime embedding...")
    batch_size = 2
    x = torch.randn(batch_size, 100, 64)  # (B, C, L)
    t = torch.randint(0, 1000, (batch_size,))
    context = torch.randn(batch_size, 100)
    temporal_seq = torch.randn(batch_size, 32, 256)  # (B, T_past, d_gru)
    temporal_emb = torch.randn(batch_size, 64)
    regime_emb = torch.randn(batch_size, 16)

    output = model(
        x, t, context,
        temporal_seq=temporal_seq,
        temporal_emb=temporal_emb,
        regime_emb=regime_emb,
    )
    assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
    print(f"[PASS] Forward pass with all conditioning successful")
    print(f"[PASS] Output shape: {output.shape}")

    # Test 3.3: Forward without regime (backward compat)
    print("\n[3.3] Testing forward pass without regime (backward compatibility)...")
    output_no_regime = model(
        x, t, context,
        temporal_seq=temporal_seq,
        temporal_emb=temporal_emb,
    )
    assert output_no_regime.shape == x.shape
    print(f"[PASS] Forward pass without regime works")

    # Test 3.4: U-Net without regime_dim (pure backward compat)
    print("\n[3.4] Testing U-Net without regime_dim...")
    model_no_regime = DiffusionUNet1D(
        in_channels=100,
        base_channels=64,
        channel_multipliers=(1, 2),
        time_dim=256,
        temporal_dim=64,
        regime_dim=0,  # Disable
        ctx_dim=100,
    )
    output_v24 = model_no_regime(
        x, t, context,
        temporal_seq=temporal_seq,
        temporal_emb=temporal_emb,
    )
    assert output_v24.shape == x.shape
    print(f"[PASS] V24-style U-Net works")

    print("\n[OK] U-Net regime conditioning tests passed!")
    return True


def test_backward_compat_v24():
    """Test backward compatibility with V24 generator."""
    print("\n" + "=" * 60)
    print("TEST 4: V24 Generator Backward Compatibility")
    print("=" * 60)

    from src.v24.diffusion.generator import DiffusionPathGeneratorV2, GeneratorConfig

    # Test 4.1: V24 generator still works
    print("\n[4.1] Testing V24 generator with new regime_probs parameter...")
    config = GeneratorConfig(
        in_channels=100,
        sequence_length=64,
        base_channels=64,
        channel_multipliers=(1, 2),
        num_paths=4,
        sampling_steps=10,
        temporal_gru_dim=0,
    )
    generator = DiffusionPathGeneratorV2(config, device="cpu")

    world_state = {"feature_0": 1.0}
    # New parameter is accepted but ignored in V24
    paths = generator.generate_paths(
        world_state=world_state,
        num_paths=4,
        regime_probs=torch.randn(9),  # Should be accepted but ignored
    )
    assert len(paths) == 4
    print(f"[PASS] V24 generator accepts regime_probs parameter (ignored)")
    print(f"[PASS] Backward compatibility maintained")

    print("\n[OK] V24 backward compatibility tests passed!")
    return True


def test_checkpoint_compatibility():
    """Test checkpoint loading with/without regime."""
    print("\n" + "=" * 60)
    print("TEST 5: Checkpoint Compatibility")
    print("=" * 60)

    from src.v26.diffusion.regime_generator import (
        RegimeDiffusionPathGenerator,
        RegimeGeneratorConfig,
    )

    # Test 5.1: Simulate loading V24 checkpoint
    print("\n[5.1] Testing V24 checkpoint compatibility...")
    config = RegimeGeneratorConfig(
        in_channels=100,
        sequence_length=64,
        base_channels=64,
        channel_multipliers=(1, 2),
        num_paths=4,
    )
    generator = RegimeDiffusionPathGenerator(config, device="cpu")

    # Create a mock V24 checkpoint (without regime)
    import tempfile
    import os

    checkpoint = {
        "model": {k: v for k, v in generator.model.state_dict().items() if "regime" not in k},
        "temporal_encoder": generator.temporal_encoder.state_dict() if generator.temporal_encoder else {},
    }

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(checkpoint, f.name)
        temp_path = f.name

    try:
        # Load with regime generator
        info = generator.load_checkpoint(temp_path, strict=False)
        print(f"[PASS] Loaded {info['loaded_keys']} keys")
        print(f"[PASS] Missing keys: {info['missing_keys']}")
        print(f"[PASS] Regime initialized fresh: {info['regime_initialized_fresh']} params")
        print(f"[PASS] V24 checkpoint compatible with V26 generator")
    finally:
        os.unlink(temp_path)

    print("\n[OK] Checkpoint compatibility tests passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("V26 PHASE 1: REGIME EMBEDDING + GENERATOR EXTENSION")
    print("=" * 60)

    tests = [
        ("Regime Embedding", test_regime_embedding),
        ("Regime Generator", test_regime_generator),
        ("U-Net Regime Conditioning", test_unet_regime_conditioning),
        ("V24 Backward Compatibility", test_backward_compat_v24),
        ("Checkpoint Compatibility", test_checkpoint_compatibility),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n[FAIL] {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status}: {name}")

    total = len(results)
    passed = sum(1 for _, s in results if s)
    print(f"\n{passed}/{total} tests passed")

    return all(s for _, s in results)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
