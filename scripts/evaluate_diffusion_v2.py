from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import (
    V24_DIFFUSION_CHECKPOINT_PATH,
    V24_DIFFUSION_FUSED_PATH,
    V24_DIFFUSION_NORM_STATS_PATH,
    OUTPUTS_V24_DIR,
)
from src.v24.diffusion.unet_1d import DiffusionUNet1D
from src.v24.diffusion.scheduler import NoiseScheduler
from src.v24.diffusion.generator import DiffusionPathGeneratorV2, GeneratorConfig


def _load_norm_stats():
    with open(str(V24_DIFFUSION_NORM_STATS_PATH), "r") as f:
        stats = json.load(f)
    means = np.array(stats["means"], dtype=np.float32)
    stds = np.array(stats["stds"], dtype=np.float32)
    stds = np.where(stds < 1e-8, 1.0, stds)
    return means, stds


def _denormalize(synthetic_normed: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    return synthetic_normed * stds[None, None, :] + means[None, None, :]


def _acf(x: np.ndarray, max_lag: int = 20) -> np.ndarray:
    n = len(x)
    x = x - x.mean()
    var = np.var(x)
    if var < 1e-12:
        return np.zeros(max_lag + 1)
    acf_vals = np.correlate(x, x, mode="full")[n - 1:]
    acf_vals = acf_vals[: max_lag + 1] / acf_vals[0]
    return acf_vals


def _evaluate_realism(real_windows: np.ndarray, synth_windows: np.ndarray) -> dict:
    results = {}

    # 1. Return distribution statistics (feature index 0 = return_1)
    real_ret = real_windows[:, :, 0].flatten()
    synth_ret = synth_windows[:, :, 0].flatten()

    results["real_return_mean"] = float(np.mean(real_ret))
    results["synth_return_mean"] = float(np.mean(synth_ret))
    results["real_return_std"] = float(np.std(real_ret))
    results["synth_return_std"] = float(np.std(synth_ret))
    results["real_return_skew"] = float(_skewness(real_ret))
    results["synth_return_skew"] = float(_skewness(synth_ret))
    results["real_return_kurt"] = float(_kurtosis(real_ret))
    results["synth_return_kurt"] = float(_kurtosis(synth_ret))

    # 2. Autocorrelation comparison
    real_acfs = []
    synth_acfs = []
    for i in range(min(100, len(real_windows))):
        real_acfs.append(_acf(real_windows[i, :, 0], max_lag=10))
    for i in range(min(100, len(synth_windows))):
        synth_acfs.append(_acf(synth_windows[i, :, 0], max_lag=10))
    real_acf_mean = np.mean(real_acfs, axis=0)
    synth_acf_mean = np.mean(synth_acfs, axis=0)
    acf_diff = np.mean(np.abs(real_acf_mean - synth_acf_mean))
    results["acf_diff_mean"] = float(acf_diff)
    results["real_acf_lag1"] = float(real_acf_mean[1])
    results["synth_acf_lag1"] = float(synth_acf_mean[1])

    # 3. Volatility clustering (GARCH effect: |r| autocorrelation)
    real_vol_acf = []
    synth_vol_acf = []
    for i in range(min(100, len(real_windows))):
        r = real_windows[i, :, 0]
        real_vol_acf.append(_acf(np.abs(r), max_lag=5))
    for i in range(min(100, len(synth_windows))):
        r = synth_windows[i, :, 0]
        synth_vol_acf.append(_acf(np.abs(r), max_lag=5))
    real_vol_acf_mean = np.mean(real_vol_acf, axis=0)
    synth_vol_acf_mean = np.mean(synth_vol_acf, axis=0)
    results["real_vol_clustering_lag1"] = float(real_vol_acf_mean[1])
    results["synth_vol_clustering_lag1"] = float(synth_vol_acf_mean[1])
    results["vol_clustering_diff"] = float(np.abs(real_vol_acf_mean[1] - synth_vol_acf_mean[1]))

    # 4. Cross-feature correlation preservation
    real_corr = np.corrcoef(real_windows.reshape(-1, real_windows.shape[-1])[:, :10].T)
    synth_corr = np.corrcoef(synth_windows.reshape(-1, synth_windows.shape[-1])[:, :10].T)
    if real_corr.ndim == 2 and synth_corr.ndim == 2:
        mask = np.triu(np.ones_like(real_corr, dtype=bool), k=1)
        corr_diff = np.mean(np.abs(real_corr[mask] - synth_corr[mask]))
        results["cross_corr_diff"] = float(corr_diff)
    else:
        results["cross_corr_diff"] = -1.0

    # 5. Cone containment: fraction of real paths that fall within synthetic cone
    if len(real_windows) > 0 and len(synth_windows) > 0:
        real_per_t = real_windows[:, :, 0]
        synth_per_t = synth_windows[:, :, 0]
        synth_lo = np.percentile(synth_per_t, 5, axis=0)
        synth_hi = np.percentile(synth_per_t, 95, axis=0)
        within = np.mean((real_per_t >= synth_lo[None, :]) & (real_per_t <= synth_hi[None, :]))
        results["cone_containment_90"] = float(within)

    # 6. Summary score
    acf_score = max(0, 1.0 - acf_diff)
    vol_score = max(0, 1.0 - results["vol_clustering_diff"])
    corr_score = max(0, 1.0 - min(results["cross_corr_diff"], 1.0))
    cone_score = min(results.get("cone_containment_90", 0.5), 1.0)
    results["realism_score"] = float(0.3 * acf_score + 0.2 * vol_score + 0.2 * corr_score + 0.3 * cone_score)

    return results


def _skewness(x: np.ndarray) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s < 1e-12:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    n = len(x)
    if n < 4:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s < 1e-12:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4) - 3.0)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    means, stds = _load_norm_stats()
    print(f"Norm stats loaded: {means.shape}")

    model = DiffusionUNet1D(
        in_channels=100, base_channels=128, channel_multipliers=(1, 2, 4),
        time_dim=256, num_res_blocks=2, ctx_dim=100,
    ).to(device)

    ckpt = torch.load(str(V24_DIFFUSION_CHECKPOINT_PATH), map_location=device, weights_only=False)
    ema_state = ckpt.get("ema", ckpt["model"])
    model.load_state_dict(ema_state)
    model.eval()
    print(f"Loaded EMA checkpoint from epoch {ckpt['epoch']}, val_loss={ckpt['best_val_loss']:.6f}")

    scheduler = NoiseScheduler(num_timesteps=1000).to(device)
    config = GeneratorConfig(
        in_channels=100, sequence_length=120, base_channels=128,
        channel_multipliers=(1, 2, 4), num_timesteps=1000, ctx_dim=100,
        guidance_scale=3.0, num_paths=64, sampling_steps=50,
    )
    gen = DiffusionPathGeneratorV2(config=config, model=model, scheduler=scheduler, device=str(device))

    # Load real data windows for comparison
    real_normed = np.load(str(V24_DIFFUSION_FUSED_PATH), mmap_mode="r")
    total = len(real_normed)
    test_start = int(total * 0.85)
    n_real = 200
    seq_len = 120

    real_windows_list = []
    for i in range(test_start, min(test_start + n_real, total - seq_len)):
        real_windows_list.append(np.asarray(real_normed[i : i + seq_len], dtype=np.float32).copy())
    real_windows = np.stack(real_windows_list)
    print(f"Real windows: {real_windows.shape}")

    # Generate synthetic paths
    print("Generating synthetic paths...")
    n_contexts = min(50, len(real_windows_list))
    synth_windows_list = []

    for i in range(n_contexts):
        context_vec = torch.tensor(real_normed[test_start + i + seq_len - 1], dtype=torch.float32).to(device)
        paths = gen.generate_paths(
            {"ctx": context_vec.cpu().numpy()},
            num_paths=1,
            steps=50,
        )
        path_data = np.array(paths[0]["data"], dtype=np.float32)
        if path_data.ndim == 2 and path_data.shape == (seq_len, 100):
            synth_windows_list.append(path_data)
        elif path_data.ndim == 2:
            synth_windows_list.append(path_data[:seq_len, :100])
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{n_contexts} paths")

    synth_normed = np.stack(synth_windows_list) if synth_windows_list else np.zeros((1, seq_len, 100))
    print(f"Synthetic windows: {synth_normed.shape}")

    # Denormalize for return statistics
    real_denorm = _denormalize(real_windows[:len(synth_normed)], means, stds)
    synth_denorm = _denormalize(synth_normed, means, stds)

    results = _evaluate_realism(real_denorm, synth_denorm)

    print("\n" + "=" * 60)
    print("DIFFUSION V2 PATH REALISM EVALUATION")
    print("=" * 60)
    for k, v in sorted(results.items()):
        if isinstance(v, float):
            print(f"  {k:40s}: {v:.6f}")
        else:
            print(f"  {k:40s}: {v}")

    out_path = OUTPUTS_V24_DIR / "diffusion_v2_realism_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved: {out_path}")


if __name__ == "__main__":
    main()
