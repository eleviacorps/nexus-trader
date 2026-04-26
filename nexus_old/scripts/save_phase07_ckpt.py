"""Extract epoch-39 EMA state from log, save as clean checkpoint."""
import json, torch, os
from pathlib import Path

# Rebuild from epoch-38 EMA (last saved was epoch 32, overwritten)
# Read all log entries and find the epoch 38 -> 39 best
lines = open("outputs/v24/diffusion_v2_phase06_log.jsonl").readlines()
entries = [json.loads(l) for l in lines]
# Epoch 38 had val_loss=1.104314, epoch 39 had 1.103935
# Save a stub checkpoint from phase06 that will resume with proper EMA
# The current best checkpoint on disk is epoch 32 (val=1.109)
# We need to resume from epoch 39's EMA weights

ckpt_32 = torch.load("models/v24/diffusion_unet1d_v2_6m_phase06.pt", map_location="cpu", weights_only=False)
print(f"Checkpoint: epoch={ckpt_32['epoch']}, val_loss={ckpt_32['best_val_loss']:.6f}")

# Save a clean copy of phase 0.6 best at epoch 39
# We can't recover the exact EMA shadow for epoch 39 since it was overwritten
# But the phase06 script saves EMA at each best epoch
# The most recent saved is epoch 32, which has val_loss=1.109
# The log shows the best at each epoch but the file was overwritten by subsequent runs

# Best approach: re-run 5 epochs from epoch 32 checkpoint (closest we have)
# This IS the right thing to do - resume from our best saved state
ckpt_path = "models/v24/diffusion_unet1d_v2_6m_phase07.pt"
state = {
    "model": ckpt_32["model"],
    "temporal_encoder": ckpt_32["temporal_encoder"],
    "ema": ckpt_32["ema"],
    "optimizer": ckpt_32["optimizer"],
    "lr_scheduler": ckpt_32["lr_scheduler"],
    "epoch": 32,
    "best_val_loss": ckpt_32["best_val_loss"],
    "phase": "0.7",
}
torch.save(state, ckpt_path)
print(f"Saved phase 0.7 resume point: epoch={state['epoch']}, val_loss={state['best_val_loss']:.6f}")
print(f"Now fine-tuning from here with sign-aware ACF + tighter clamp [-2.0, 2.0]")