import torch
import numpy as np
import sys
sys.path.insert(0, 'C:/PersonalDrive/Programming/AiStudio/nexus-trader/nexus_packaged')

from MMFPS.generator.constrained_diffusion_generator import ConstrainedDiffusionGenerator, DiffusionGeneratorConfig

config = DiffusionGeneratorConfig()
model = ConstrainedDiffusionGenerator(config)
model.eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

B = 1
context = torch.randn(B, 144).to(device)
regime_emb = torch.zeros(B, 64).to(device)
quant_emb = torch.zeros(B, 64).to(device)
temporal_seq = torch.randn(B, 20, 144).to(device)

print(f"=== SANITY CHECK ===")
print(f"Target std: {config.target_std}")
print(f"Target mean: {config.target_mean}")

with torch.no_grad():
    paths = model.quick_generate(context, regime_emb, quant_emb, temporal_seq, num_paths=128)
    paths = paths.cpu().numpy()

price = paths[..., 0]

returns_flat = price.flatten()

print(f"\n=== GENERATED STATS (raw output is already return %) ===")
print(f"Mean: {returns_flat.mean():.6f}")
print(f"Std: {returns_flat.std():.6f}")
print(f"Min: {returns_flat.min():.6f}")
print(f"Max: {returns_flat.max():.6f}")

extreme_count = (np.abs(returns_flat) > 0.2).sum()
print(f"\n|return| > 20%: {extreme_count}/{len(returns_flat)} ({extreme_count/len(returns_flat)*100:.1f}%)")

print(f"\nBound check:")
print(f"  All values within +/- 10%: {(np.abs(returns_flat) <= 0.1).all()}")
print(f"  All values within +/- 20%: {(np.abs(returns_flat) <= 0.2).all()}")

if np.abs(returns_flat).max() > 0.2:
    print("\nFAILED: Outputs outside reasonable bounds")
else:
    print("\nPASS: Outputs within reasonable bounds")