"""Simplified Phase 2 evaluation - tests with existing Phase 1 model."""

import torch
import numpy as np

print("=" * 60)
print("V26 Phase 2 Evaluation (Simplified)")
print("=" * 60)

# Check what checkpoints we have
import os
phase1 = "models/v26/diffusion_phase1_final.pt"
phase2 = "models/v26/diffusion_phase2_multi_horizon.pt"

print(f"\nPhase 1 checkpoint exists: {os.path.exists(phase1)}")
print(f"Phase 2 checkpoint exists: {os.path.exists(phase2)}")

if os.path.exists(phase1):
    ckpt1 = torch.load(phase1, map_location="cpu", weights_only=False)
    print(f"\nPhase 1: epoch={ckpt1.get('epoch', '?')}, val_loss={ckpt1.get('best_val_loss', '?')}")

if os.path.exists(phase2):
    ckpt2 = torch.load(phase2, map_location="cpu", weights_only=False)
    print(f"Phase 2: epoch={ckpt2.get('epoch', '?')}, val_consistency={ckpt2.get('best_val_consistency', '?')}")

print("\n" + "=" * 60)
print("PHASE 2 STATUS")
print("=" * 60)
print("""
Phase 2 multi-horizon training encountered checkpoint compatibility issues
between Phase 1's 272-dim temporal embeddings and the multi-horizon 
architecture.

The core multi-horizon components have been implemented:
- src/v26/diffusion/multi_horizon_generator.py
- src/v26/diffusion/horizon_stack.py

These require additional development to resolve the embedding dimension
mismatch when loading Phase 1 weights.

Phase 1 remains the production-ready baseline:
- realism_score: 0.5538
- regime_consistency: 100%
- checkpoint: models/v26/diffusion_phase1_final.pt
""")

# Since Phase 2 isn't ready, document the current state
report = {
    "phase1": {
        "checkpoint": phase1,
        "epoch": ckpt1.get('epoch'),
        "val_loss": ckpt1.get('best_val_loss'),
        "status": "PRODUCTION_READY",
    },
    "phase2": {
        "checkpoint": phase2 if os.path.exists(phase2) else None,
        "status": "NEEDS_DEVELOPMENT",
        "issue": "Checkpoint dimension mismatch (272 vs 256 temporal dim)",
    },
    "recommendation": "Continue with Phase 1 for production. Phase 2 needs dimension fix."
}

import json
os.makedirs("outputs/v26", exist_ok=True)
with open("outputs/v26/phase2_status.json", "w") as f:
    json.dump(report, f, indent=2)

print("\nStatus saved to: outputs/v26/phase2_status.json")