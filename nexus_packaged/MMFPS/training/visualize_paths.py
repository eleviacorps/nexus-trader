import numpy as np
import matplotlib.pyplot as plt

# Load paths
paths = np.load('outputs/sample_paths.npy')  # (50, 20)

# Filter extreme outliers for visualization
filtered = paths[(np.abs(paths[:, -1]) < 5).all(axis=1)]
print(f"Filtered paths: {len(filtered)}/{len(paths)}")

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: All 50 paths
ax1 = axes[0, 0]
for i in range(len(paths)):
    ax1.plot(paths[i], alpha=0.5, linewidth=0.8)
ax1.set_title(f'All 50 Generated Paths (UNFILTERED)\nReturns: min={paths[:, -1].min():.1%}, max={paths[:, -1].max():.1%}')
ax1.set_xlabel('Time Step')
ax1.set_ylabel('Return (%)')
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax1.grid(True, alpha=0.3)

# Plot 2: Filtered paths (realistic ones only)
ax2 = axes[0, 1]
for i in range(min(len(filtered), 30)):
    ax2.plot(filtered[i], alpha=0.7, linewidth=1)
ax2.set_title(f'Reasonable Paths Only ({len(filtered)} paths)\n|return| < 500%')
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Return (%)')
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax2.grid(True, alpha=0.3)

# Plot 3: Return distribution
ax3 = axes[1, 0]
returns = paths[:, -1]
ax3.hist(returns, bins=50, edgecolor='black', alpha=0.7)
ax3.axvline(x=0, color='red', linestyle='--', label='Zero')
ax3.axvline(x=returns.mean(), color='green', linestyle='--', label=f'Mean: {returns.mean():.2f}')
ax3.set_title(f'Final Return Distribution\nMean: {returns.mean():.2f}, Std: {returns.std():.2f}')
ax3.set_xlabel('Final Return')
ax3.set_ylabel('Count')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Real data comparison
ax4 = axes[1, 1]
data = np.load('C:/PersonalDrive/Programming/AiStudio/nexus-trader/data/features/diffusion_fused_6m.npy')
# Get returns from real data
real_prices = data[:1000, -20:]  # first 1000 samples, last 20 timesteps
real_returns = (real_prices[:, 1:] - real_prices[:, :-1]) / (np.abs(real_prices[:, :-1]) + 1e-8)
real_returns_flat = real_returns.flatten()
real_returns_flat = real_returns_flat[np.abs(real_returns_flat) < 1]  # filter outliers

ax4.hist(real_returns_flat, bins=50, alpha=0.7, edgecolor='black', color='green')
ax4.set_title(f'Reall Data Returns Distribution\nStd: {np.std(real_returns_flat):.4f}')
ax4.set_xlabel('Return')
ax4.set_ylabel('Count')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/path_visualization.png', dpi=150, bbox_inches='tight')
print('Saved visualization to outputs/path_visualization.png')

# Print key statistics
print(f"\n=== ANALYSIS ===")
print(f"Generated paths: {len(paths)}")
print(f"Final returns - min: {paths[:, -1].min():.2%}, max: {paths[:, -1].max():.2%}")
print(f"Final returns - mean: {paths[:, -1].mean():.2%}, std: {paths[:, -1].std():.2%}")
print(f"Extreme paths (|return| > 100%): {np.sum(np.abs(paths[:, -1]) > 1)}/{len(paths)}")
print(f"Real data returns - std: {np.std(real_returns_flat):.4f}")
