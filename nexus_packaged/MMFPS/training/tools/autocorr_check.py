import sys
sys.path.insert(0, 'C:/PersonalDrive/Programming/AiStudio/nexus-trader/nexus_packaged')
import torch
import numpy as np

device = 'cuda'
from MMFPS.generator.constrained_diffusion_generator import ConstrainedDiffusionGenerator, DiffusionGeneratorConfig

config = DiffusionGeneratorConfig()
model = ConstrainedDiffusionGenerator(config)
ckpt = torch.load('C:/PersonalDrive/Programming/AiStudio/nexus-trader/nexus_packaged/outputs/MMFPS/generator_checkpoints/checkpoint_step_25000.pt', map_location=device, weights_only=False)
model.load_state_dict(ckpt['model'])
model = model.to(device).eval()

print(f'Checkpoint step: {ckpt.get("step", "unknown")}')
print()

context = torch.randn(128, 144, device=device)
regime_emb = torch.zeros(128, 64, device=device)
quant_emb = torch.zeros(128, 64, device=device)

with torch.no_grad():
    out = model(context, regime_emb, quant_emb)
    paths = out.paths
    returns = paths[:, 0, :, 0].cpu().numpy()

print('=== DISTRIBUTION ===')
print(f'Mean: {returns.mean():.6f}')
print(f'Std:  {returns.std():.6f}')
print(f'Min:  {returns.min():.6f}')
print(f'Max:  {returns.max():.6f}')
print(f'|ret| > 10%: {(np.abs(returns) > 0.10).sum() / returns.size * 100:.2f}%')
print()

print('=== TEMPORAL ===')
vel = np.diff(returns, axis=1)
print(f'Velocity mean abs: {np.abs(vel).mean():.6f}')
accel = np.diff(vel, axis=1)
print(f'Accel mean abs: {np.abs(accel).mean():.6f}')
print()

print('=== AUTOCORRELATION ===')
for lag in [1, 2, 3]:
    ac = []
    for i in range(128):
        r = returns[i]
        c = np.corrcoef(r[:-lag], r[lag:])[0, 1]
        if not np.isnan(c):
            ac.append(c)
    print(f'Lag-{lag}: {np.mean(ac):.4f}  (std: {np.std(ac):.4f})')

print()
sign_changes = np.sign(vel[:, 1:]) != np.sign(vel[:, :-1])
print(f'Sign change rate: {sign_changes.mean():.4f}  (<0.35 = coherent)')