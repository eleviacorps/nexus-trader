"""Phase 0.5 evaluation on existing checkpoint — NO TRAINING."""

import torch
import numpy as np
import json
from pathlib import Path

def acf(x, max_lag=20):
    n = len(x)
    x = x - x.mean()
    var = np.var(x)
    if var < 1e-12:
        return np.zeros(max_lag + 1)
    acf_vals = np.correlate(x, x, mode="full")[n - 1:]
    acf_vals = acf_vals[:max_lag + 1] / acf_vals[0]
    return acf_vals

# Load checkpoint
ckpt_path = "models/v24/diffusion_unet1d_v2_6m.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
epoch = ckpt["epoch"]
val_loss = ckpt.get("best_val_loss", 0)

print(f"Checkpoint: epoch {epoch}, val_loss {val_loss:.6f}")

# Load norm stats
with open("config/diffusion_norm_stats_6m.json", "r") as f:
    stats = json.load(f)
means = np.array(stats["means"], dtype=np.float32)
stds = np.array(stats["stds"], dtype=np.float32)
stds = np.where(stds < 1e-8, 1.0, stds)

# Load real data
fused = np.load("data/features/diffusion_fused_6m.npy")
total = len(fused)
test_start = int(total * 0.85)
SEQ_LEN = 120
N_REAL = 200
N_SYNTH = 50
CONTEXT_LEN = 256

# Extract real windows
real_windows = []
for i in range(test_start, min(test_start + N_REAL, total - SEQ_LEN)):
    real_windows.append(fused[i:i + SEQ_LEN].copy())
real_windows = np.array(real_windows, dtype=np.float32)
print(f"Real windows: {real_windows.shape}")

# Set up model (same architecture as training)
from src.v24.diffusion.unet_1d import DiffusionUNet1D
from src.v24.diffusion.temporal_encoder import TemporalEncoder
from src.v24.diffusion.scheduler import NoiseScheduler
from src.v24.diffusion.generator import DiffusionPathGeneratorV2, GeneratorConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d_gru = 256
temporal_dim = 256
model = DiffusionUNet1D(
    in_channels=144, base_channels=128, channel_multipliers=(1, 2, 4),
    time_dim=256, num_res_blocks=2, ctx_dim=144,
    temporal_dim=temporal_dim, d_gru=d_gru,
).to(device)

ema_state = ckpt.get("ema", ckpt["model"])
model.load_state_dict(ema_state)
model.eval()
print(f"Model loaded from EMA")

temporal_encoder = TemporalEncoder(in_features=144, d_gru=d_gru, num_layers=2, film_dim=256).to(device)
if "temporal_encoder" in ckpt:
    temporal_encoder.load_state_dict(ckpt["temporal_encoder"])
temporal_encoder.eval()
print(f"Temporal encoder loaded")

scheduler = NoiseScheduler(1000).to(device)
config = GeneratorConfig(
    in_channels=144, sequence_length=SEQ_LEN, base_channels=128,
    channel_multipliers=(1, 2, 4), num_timesteps=1000, ctx_dim=144,
    guidance_scale=3.0, num_paths=1, sampling_steps=50,
    temporal_gru_dim=d_gru, temporal_layers=2, context_len=CONTEXT_LEN,
    norm_stats_path="config/diffusion_norm_stats_6m.json",
)
gen = DiffusionPathGeneratorV2(config=config, model=model, scheduler=scheduler, temporal_encoder=temporal_encoder, device=str(device))

# Generate synthetic paths
print(f"Generating {N_SYNTH} synthetic paths...")
synth_list = []
for i in range(N_SYNTH):
    row_idx = test_start + i
    context_vec = torch.tensor(fused[row_idx + SEQ_LEN - 1, :144], dtype=torch.float32).to(device)
    
    past_start = max(0, row_idx - CONTEXT_LEN)
    past_data = fused[past_start:row_idx].copy()
    if len(past_data) < CONTEXT_LEN:
        pad = np.zeros((CONTEXT_LEN - len(past_data), 144), dtype=np.float32)
        past_data = np.concatenate([pad, past_data], axis=0)
    past_context = torch.tensor(past_data, dtype=torch.float32).unsqueeze(0).to(device)
    
    paths = gen.generate_paths({"ctx": context_vec.cpu().numpy()}, num_paths=1, steps=50, past_context=past_context)
    path_data = np.array(paths[0]["data"], dtype=np.float32)
    if path_data.ndim == 2:
        synth_list.append(path_data[:SEQ_LEN, :144])
    if (i + 1) % 10 == 0:
        print(f"  Generated {i + 1}/{N_SYNTH}")

synth_windows = np.array(synth_list, dtype=np.float32) if synth_list else np.zeros((1, SEQ_LEN, 144))
print(f"Synthetic windows: {synth_windows.shape}")

# Denormalize
def denorm(x):
    return x * stds[None, None, :] + means[None, None, :]

real_denorm = denorm(real_windows[:len(synth_windows)])
synth_denorm = gen.denormalize(synth_windows)

# === METRICS ===
RETURN_IDX = 0
real_ret = real_denorm[:, :, RETURN_IDX].flatten()
synth_ret = synth_denorm[:, :, RETURN_IDX].flatten()

# ACF
real_acf_vals = []
synth_acf_vals = []
for i in range(min(100, len(real_denorm))):
    real_acf_vals.append(acf(real_denorm[i, :, RETURN_IDX], max_lag=10))
for i in range(min(100, len(synth_denorm))):
    synth_acf_vals.append(acf(synth_denorm[i, :, RETURN_IDX], max_lag=10))
real_acf_mean = np.mean(real_acf_vals, axis=0)
synth_acf_mean = np.mean(synth_acf_vals, axis=0)
acf_lag1_real = real_acf_mean[1]
acf_lag1_synth = synth_acf_mean[1]

# Vol clustering
real_vol = []
synth_vol = []
for i in range(min(100, len(real_denorm))):
    r = real_denorm[i, :, RETURN_IDX]
    real_vol.append(acf(np.abs(r), max_lag=5))
for i in range(min(100, len(synth_denorm))):
    r = synth_denorm[i, :, RETURN_IDX]
    synth_vol.append(acf(np.abs(r), max_lag=5))
vol_cluster_real = np.mean(real_vol, axis=0)[1]
vol_cluster_synth = np.mean(synth_vol, axis=0)[1]

# Return std
return_std_real = np.std(real_ret)
return_std_synth = np.std(synth_ret)
std_ratio = return_std_synth / (return_std_real + 1e-12)

# Cone containment
real_per_t = real_denorm[:, :, RETURN_IDX]
synth_per_t = synth_denorm[:, :, RETURN_IDX]
for pct in [50, 80, 90]:
    lo = np.percentile(synth_per_t, 100 - pct, axis=0)
    hi = np.percentile(synth_per_t, pct, axis=0)
    within = np.mean((real_per_t >= lo[None, :]) & (real_per_t <= hi[None, :]))
    print(f"cone_{pct} = {within:.4f}")

# Path diversity (std across paths at each timestep)
path_diversity = np.mean(np.std(synth_denorm, axis=0))

# Cross-corr
n_corr = min(10, real_denorm.shape[-1])
real_corr = np.corrcoef(real_denorm.reshape(-1, real_denorm.shape[-1])[:, :n_corr].T)
synth_corr = np.corrcoef(synth_denorm.reshape(-1, synth_denorm.shape[-1])[:, :n_corr].T)
mask = np.triu(np.ones_like(real_corr, dtype=bool), k=1)
cross_corr_diff = np.mean(np.abs(real_corr[mask] - synth_corr[mask]))

# Realism score
acf_score = max(0, 1.0 - abs(acf_lag1_real - acf_lag1_synth))
vol_score = max(0, 1.0 - abs(vol_cluster_real - vol_cluster_synth))
corr_score = max(0, 1.0 - min(cross_corr_diff, 1.0))
cone_score = min(within, 1.0)
realism_score = 0.3 * acf_score + 0.2 * vol_score + 0.2 * corr_score + 0.3 * cone_score

print(f"\n=== DIFFUSION V26 EVALUATION ===")
print(f"Checkpoint: epoch={epoch}, val_loss={val_loss:.6f}")
print(f"\nRealism metrics:")
print(f"acf_lag1_real = {acf_lag1_real:.4f}")
print(f"acf_lag1_synth = {acf_lag1_synth:.4f}")
print(f"vol_cluster_real = {vol_cluster_real:.4f}")
print(f"vol_cluster_synth = {vol_cluster_synth:.4f}")
print(f"return_std_real = {return_std_real:.6f}")
print(f"return_std_synth = {return_std_synth:.6f}")
print(f"std_ratio = {std_ratio:.4f}")
print(f"path_diversity = {path_diversity:.6f}")
print(f"cross_corr_diff = {cross_corr_diff:.4f}")
print(f"realism_score = {realism_score:.4f}")

# Assessment
print(f"\nAssessment:")
print(f"- Temporal structure realistic: {'YES' if acf_lag1_synth > 0.5 else 'NO'} (ACF={acf_lag1_synth:.4f}, target>0.5)")
print(f"- Model overdispersed: {'YES' if std_ratio > 2.0 else 'NO'} (std_ratio={std_ratio:.4f}, target 1.0-2.0)")
print(f"- Vol clustering realistic: {'YES' if vol_cluster_synth > 0.4 else 'NO'} (vol={vol_cluster_synth:.4f}, target>0.4)")

# Decision
targets_met = 0
if acf_lag1_synth > 0.5:
    targets_met += 1
if vol_cluster_synth > 0.4:
    targets_met += 1
if 1.0 <= std_ratio <= 2.0:
    targets_met += 1
if 0.75 <= within <= 0.90:
    targets_met += 1

print(f"\nTargets met: {targets_met}/4")
if targets_met >= 3 and realism_score > 0.6:
    print("Recommendation: STOP TRAINING")
else:
    print("Recommendation: TRAIN 5-10 MORE EPOCHS")