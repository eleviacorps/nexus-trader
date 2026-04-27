import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, 'C:/PersonalDrive/Programming/AiStudio/nexus-trader/nexus_packaged')

from MMFPS.generator.constrained_diffusion_generator import ConstrainedDiffusionGenerator, DiffusionGeneratorConfig

def plot_paths(model_path=None, num_paths=50, save_path=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = DiffusionGeneratorConfig()
    model = ConstrainedDiffusionGenerator(config)
    
    if model_path:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
    
    model = model.to(device)
    model.eval()
    
    # Random context
    context = torch.randn(1, 144).to(device)
    regime_emb = torch.zeros(1, 64).to(device)
    quant_emb = torch.zeros(1, 64).to(device)
    temporal_seq = torch.randn(1, 20, 144).to(device)
    
    with torch.no_grad():
        paths = model.quick_generate(context, regime_emb, quant_emb, temporal_seq, num_paths=num_paths)
        paths = paths.cpu().numpy()[0]  # (num_paths, 20, 144)
    
    price = paths[..., 0]  # (num_paths, 20) - returns in %
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. All paths over time
    ax1 = axes[0, 0]
    for i in range(num_paths):
        ax1.plot(price[i] * 100, alpha=0.5, linewidth=0.8)
    ax1.set_title(f'Generated Paths ({num_paths} samples)')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Return (%)')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    
    # 2. Final return distribution
    ax2 = axes[0, 1]
    final_returns = price[:, -1] * 100
    ax2.hist(final_returns, bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', label='Zero')
    ax2.axvline(x=np.mean(final_returns), color='green', linestyle='--', label=f'Mean: {np.mean(final_returns):.2f}%')
    ax2.set_title(f'Final Return Distribution\nStd: {np.std(final_returns):.2f}%')
    ax2.set_xlabel('Return (%)')
    ax2.set_ylabel('Count')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Mean/std over time
    ax3 = axes[1, 0]
    mean_over_time = price.mean(axis=0) * 100
    std_over_time = price.std(axis=0) * 100
    ax3.plot(mean_over_time, label='Mean', linewidth=2)
    ax3.fill_between(range(len(mean_over_time)), 
                    mean_over_time - std_over_time, 
                    mean_over_time + std_over_time, 
                    alpha=0.3, label='+/- 1 std')
    ax3.set_title('Mean & Std Over Time')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Return (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Stats summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    stats_text = f"""
=== GENERATION STATS ===

Final Returns (T=19):
  Mean: {final_returns.mean():.2f}%
  Std: {final_returns.std():.2f}%
  Min: {final_returns.min():.2f}%
  Max: {final_returns.max():.2f}%

Bound Check:
  Within +/- 10%: {(np.abs(final_returns) <= 10).sum()}/{num_paths} ({(np.abs(final_returns) <= 10).mean()*100:.1f}%)
  Within +/- 20%: {(np.abs(final_returns) <= 20).sum()}/{num_paths} ({(np.abs(final_returns) <= 20).mean()*100:.1f}%)

Target:
  Mean: 0.0%
  Std: 5.0%
    """
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    save_path = save_path or 'outputs/generated_paths.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved to {save_path}')
    
    return price

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--num_paths', type=int, default=50)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    
    plot_paths(args.model, args.num_paths, args.output)