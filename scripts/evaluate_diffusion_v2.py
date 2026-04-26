"""Evaluate Phase 0.5 diffusion path realism on 6M data with temporal encoder.

Compares Phase 0 (405K, no temporal) vs Phase 0.5 (6M, temporal) side-by-side.
Uses proper denormalization via generator.denormalize().

Usage:
    python scripts/evaluate_diffusion_v2.py
    python scripts/evaluate_diffusion_v2.py --phase0-only
    python scripts/evaluate_diffusion_v2.py --phase05-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import (
    V24_DIFFUSION_CHECKPOINT_6M_PATH,
    V24_DIFFUSION_CHECKPOINT_PATH,
    V24_DIFFUSION_FUSED_6M_PATH,
    V24_DIFFUSION_FUSED_PATH,
    V24_DIFFUSION_NORM_STATS_6M_PATH,
    V24_DIFFUSION_NORM_STATS_PATH,
    OUTPUTS_V24_DIR,
)
from src.v24.diffusion.generator import DiffusionPathGeneratorV2, GeneratorConfig
from src.v24.diffusion.scheduler import NoiseScheduler
from src.v24.diffusion.temporal_encoder import TemporalEncoder
from src.v24.diffusion.unet_1d import DiffusionUNet1D


RETURN_FEATURE_IDX = 0
N_REAL_WINDOWS = 200
N_SYNTH_CONTEXTS = 50
SEQ_LEN = 120
CONTEXT_LEN = 256


def _acf(x: np.ndarray, max_lag: int = 20) -> np.ndarray:
    n = len(x)
    x = x - x.mean()
    var = np.var(x)
    if var < 1e-12:
        return np.zeros(max_lag + 1)
    acf_vals = np.correlate(x, x, mode="full")[n - 1:]
    acf_vals = acf_vals[: max_lag + 1] / acf_vals[0]
    return acf_vals


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


def _load_norm_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(str(path), "r") as f:
        stats = json.load(f)
    means = np.array(stats["means"], dtype=np.float32)
    stds = np.array(stats["stds"], dtype=np.float32)
    stds = np.where(stds < 1e-8, 1.0, stds)
    return means, stds


def _denormalize(synthetic_normed: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    return synthetic_normed * stds[None, None, :] + means[None, None, :]


def _extract_real_windows(fused_path: Path, n_features: int) -> np.ndarray:
    real_normed = np.load(str(fused_path), mmap_mode="r")
    total = len(real_normed)
    test_start = int(total * 0.85)
    windows = []
    for i in range(test_start, min(test_start + N_REAL_WINDOWS, total - SEQ_LEN)):
        w = np.asarray(real_normed[i : i + SEQ_LEN, :n_features], dtype=np.float32).copy()
        windows.append(w)
    return np.stack(windows)


def _evaluate_realism(real_windows: np.ndarray, synth_windows: np.ndarray) -> dict:
    results = {}

    real_ret = real_windows[:, :, RETURN_FEATURE_IDX].flatten()
    synth_ret = synth_windows[:, :, RETURN_FEATURE_IDX].flatten()

    results["real_return_mean"] = float(np.mean(real_ret))
    results["synth_return_mean"] = float(np.mean(synth_ret))
    results["real_return_std"] = float(np.std(real_ret))
    results["synth_return_std"] = float(np.std(synth_ret))
    results["return_std_ratio"] = float(np.std(synth_ret) / (np.std(real_ret) + 1e-12))
    results["real_return_skew"] = float(_skewness(real_ret))
    results["synth_return_skew"] = float(_skewness(synth_ret))
    results["real_return_kurt"] = float(_kurtosis(real_ret))
    results["synth_return_kurt"] = float(_kurtosis(synth_ret))

    real_acfs = []
    synth_acfs = []
    for i in range(min(100, len(real_windows))):
        real_acfs.append(_acf(real_windows[i, :, RETURN_FEATURE_IDX], max_lag=10))
    for i in range(min(100, len(synth_windows))):
        synth_acfs.append(_acf(synth_windows[i, :, RETURN_FEATURE_IDX], max_lag=10))
    real_acf_mean = np.mean(real_acfs, axis=0)
    synth_acf_mean = np.mean(synth_acfs, axis=0)
    acf_diff = np.mean(np.abs(real_acf_mean - synth_acf_mean))
    results["acf_diff_mean"] = float(acf_diff)
    results["real_acf_lag1"] = float(real_acf_mean[1])
    results["synth_acf_lag1"] = float(synth_acf_mean[1])

    real_vol_acf = []
    synth_vol_acf = []
    for i in range(min(100, len(real_windows))):
        r = real_windows[i, :, RETURN_FEATURE_IDX]
        real_vol_acf.append(_acf(np.abs(r), max_lag=5))
    for i in range(min(100, len(synth_windows))):
        r = synth_windows[i, :, RETURN_FEATURE_IDX]
        synth_vol_acf.append(_acf(np.abs(r), max_lag=5))
    real_vol_acf_mean = np.mean(real_vol_acf, axis=0)
    synth_vol_acf_mean = np.mean(synth_vol_acf, axis=0)
    results["real_vol_clustering_lag1"] = float(real_vol_acf_mean[1])
    results["synth_vol_clustering_lag1"] = float(synth_vol_acf_mean[1])
    results["vol_clustering_diff"] = float(np.abs(real_vol_acf_mean[1] - synth_vol_acf_mean[1]))

    n_corr_feats = min(10, real_windows.shape[-1])
    real_corr = np.corrcoef(real_windows.reshape(-1, real_windows.shape[-1])[:, :n_corr_feats].T)
    synth_corr = np.corrcoef(synth_windows.reshape(-1, synth_windows.shape[-1])[:, :n_corr_feats].T)
    if real_corr.ndim == 2 and synth_corr.ndim == 2:
        mask = np.triu(np.ones_like(real_corr, dtype=bool), k=1)
        corr_diff = np.mean(np.abs(real_corr[mask] - synth_corr[mask]))
        results["cross_corr_diff"] = float(corr_diff)
    else:
        results["cross_corr_diff"] = -1.0

    if len(real_windows) > 0 and len(synth_windows) > 0:
        real_per_t = real_windows[:, :, RETURN_FEATURE_IDX]
        synth_per_t = synth_windows[:, :, RETURN_FEATURE_IDX]
        synth_lo = np.percentile(synth_per_t, 5, axis=0)
        synth_hi = np.percentile(synth_per_t, 95, axis=0)
        within = np.mean((real_per_t >= synth_lo[None, :]) & (real_per_t <= synth_hi[None, :]))
        results["cone_containment_90"] = float(within)

    acf_score = max(0, 1.0 - acf_diff)
    vol_score = max(0, 1.0 - results["vol_clustering_diff"])
    corr_score = max(0, 1.0 - min(results["cross_corr_diff"], 1.0))
    cone_score = min(results.get("cone_containment_90", 0.5), 1.0)
    results["realism_score"] = float(0.3 * acf_score + 0.2 * vol_score + 0.2 * corr_score + 0.3 * cone_score)

    return results


def _run_phase0_eval(device: torch.device) -> dict:
    print("\n" + "=" * 60)
    print("PHASE 0 EVALUATION (405K, 100 features, no temporal)")
    print("=" * 60)

    means, stds = _load_norm_stats(V24_DIFFUSION_NORM_STATS_PATH)

    model = DiffusionUNet1D(
        in_channels=100, base_channels=128, channel_multipliers=(1, 2, 4),
        time_dim=256, num_res_blocks=2, ctx_dim=100,
    ).to(device)

    ckpt = torch.load(str(V24_DIFFUSION_CHECKPOINT_PATH), map_location=device, weights_only=False)
    ema_state = ckpt.get("ema", ckpt["model"])
    model.load_state_dict(ema_state)
    model.eval()
    print(f"Loaded Phase 0 checkpoint: epoch {ckpt['epoch']}, val_loss={ckpt['best_val_loss']:.6f}")

    scheduler = NoiseScheduler(num_timesteps=1000).to(device)
    config = GeneratorConfig(
        in_channels=100, sequence_length=SEQ_LEN, base_channels=128,
        channel_multipliers=(1, 2, 4), num_timesteps=1000, ctx_dim=100,
        guidance_scale=3.0, num_paths=1, sampling_steps=50,
        temporal_gru_dim=0, context_len=0,
    )
    gen = DiffusionPathGeneratorV2(config=config, model=model, scheduler=scheduler, device=str(device))

    real_windows = _extract_real_windows(V24_DIFFUSION_FUSED_PATH, n_features=100)
    print(f"Real windows: {real_windows.shape}")

    real_normed = np.load(str(V24_DIFFUSION_FUSED_PATH), mmap_mode="r")
    total = len(real_normed)
    test_start = int(total * 0.85)

    synth_list = []
    for i in range(N_SYNTH_CONTEXTS):
        context_vec = torch.tensor(real_normed[test_start + i + SEQ_LEN - 1, :100], dtype=torch.float32).to(device)
        paths = gen.generate_paths({"ctx": context_vec.cpu().numpy()}, num_paths=1, steps=50)
        path_data = np.array(paths[0]["data"], dtype=np.float32)
        if path_data.ndim == 2:
            synth_list.append(path_data[:SEQ_LEN, :100])
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{N_SYNTH_CONTEXTS} paths")

    synth_normed = np.stack(synth_list) if synth_list else np.zeros((1, SEQ_LEN, 100))

    real_denorm = _denormalize(real_windows[:len(synth_normed)], means, stds)
    synth_denorm = _denormalize(synth_normed, means, stds)

    results = _evaluate_realism(real_denorm, synth_denorm)
    results["phase"] = "phase0"
    results["dataset"] = "405K_15m"
    results["n_features"] = 100
    results["temporal"] = False
    return results


def _run_phase05_eval(device: torch.device) -> dict:
    print("\n" + "=" * 60)
    print("PHASE 0.5 EVALUATION (6M, 144 features, GRU temporal)")
    print("=" * 60)

    means, stds = _load_norm_stats(V24_DIFFUSION_NORM_STATS_6M_PATH)

    temporal_dim = 256
    d_gru = 256

    model = DiffusionUNet1D(
        in_channels=144, base_channels=128, channel_multipliers=(1, 2, 4),
        time_dim=256, num_res_blocks=2, ctx_dim=144,
        temporal_dim=temporal_dim, d_gru=d_gru,
    ).to(device)

    ckpt_path = str(V24_DIFFUSION_CHECKPOINT_6M_PATH)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ema_state = ckpt.get("ema", ckpt["model"])
    model.load_state_dict(ema_state)
    model.eval()
    print(f"Loaded Phase 0.5 checkpoint: epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('best_val_loss', '?')}")

    temporal_encoder = TemporalEncoder(
        in_features=144, d_gru=d_gru, num_layers=2, film_dim=256,
    ).to(device)
    if "temporal_encoder" in ckpt:
        temporal_encoder.load_state_dict(ckpt["temporal_encoder"])
        temporal_encoder.eval()
        print("Loaded temporal encoder from checkpoint")
    else:
        print("WARNING: No temporal_encoder in checkpoint — using random init")

    scheduler = NoiseScheduler(num_timesteps=1000).to(device)
    config = GeneratorConfig(
        in_channels=144, sequence_length=SEQ_LEN, base_channels=128,
        channel_multipliers=(1, 2, 4), num_timesteps=1000, ctx_dim=144,
        guidance_scale=3.0, num_paths=1, sampling_steps=50,
        temporal_gru_dim=d_gru, temporal_layers=2, context_len=CONTEXT_LEN,
        norm_stats_path=str(V24_DIFFUSION_NORM_STATS_6M_PATH),
    )
    gen = DiffusionPathGeneratorV2(
        config=config, model=model, scheduler=scheduler,
        temporal_encoder=temporal_encoder, device=str(device),
    )

    real_windows = _extract_real_windows(V24_DIFFUSION_FUSED_6M_PATH, n_features=144)
    print(f"Real windows: {real_windows.shape}")

    real_normed = np.load(str(V24_DIFFUSION_FUSED_6M_PATH), mmap_mode="r")
    total = len(real_normed)
    test_start = int(total * 0.85)

    synth_list = []
    for i in range(N_SYNTH_CONTEXTS):
        row_idx = test_start + i
        context_vec = torch.tensor(real_normed[row_idx + SEQ_LEN - 1, :144], dtype=torch.float32).to(device)

        past_start = max(0, row_idx - CONTEXT_LEN)
        past_data = np.asarray(real_normed[past_start:row_idx, :144], dtype=np.float32).copy()
        if len(past_data) < CONTEXT_LEN:
            pad = np.zeros((CONTEXT_LEN - len(past_data), 144), dtype=np.float32)
            past_data = np.concatenate([pad, past_data], axis=0)
        past_context = torch.tensor(past_data, dtype=torch.float32).unsqueeze(0).to(device)

        paths = gen.generate_paths(
            {"ctx": context_vec.cpu().numpy()},
            num_paths=1, steps=50,
            past_context=past_context,
        )
        path_data = np.array(paths[0]["data"], dtype=np.float32)
        if path_data.ndim == 2:
            synth_list.append(path_data[:SEQ_LEN, :144])
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{N_SYNTH_CONTEXTS} paths")

    synth_normed = np.stack(synth_list) if synth_list else np.zeros((1, SEQ_LEN, 144))

    real_denorm = _denormalize(real_windows[:len(synth_normed)], means, stds)
    synth_denorm = gen.denormalize(synth_normed)

    results = _evaluate_realism(real_denorm, synth_denorm)
    results["phase"] = "phase0.5"
    results["dataset"] = "6M_1m"
    results["n_features"] = 144
    results["temporal"] = True
    results["context_len"] = CONTEXT_LEN
    results["d_gru"] = d_gru
    return results


def _print_results(label: str, results: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"{label}")
    print("=" * 60)
    for k, v in sorted(results.items()):
        if isinstance(v, float):
            print(f"  {k:40s}: {v:.6f}")
        else:
            print(f"  {k:40s}: {v}")


def _print_comparison(phase0: dict, phase05: dict) -> None:
    print(f"\n{'=' * 60}")
    print("PHASE 0 vs PHASE 0.5 COMPARISON")
    print("=" * 60)

    comparison_keys = [
        ("realism_score", "Realism score", "higher_better"),
        ("synth_acf_lag1", "Synthetic ACF lag-1", "target>0.5"),
        ("synth_vol_clustering_lag1", "Synthetic vol clustering lag-1", "target>0.4"),
        ("return_std_ratio", "Return std ratio", "target 1.5-2x"),
        ("cone_containment_90", "Cone containment 90%", "target 75-90%"),
        ("real_acf_lag1", "Real ACF lag-1", "reference"),
        ("real_vol_clustering_lag1", "Real vol clustering lag-1", "reference"),
        ("acf_diff_mean", "ACF diff (mean)", "lower_better"),
        ("vol_clustering_diff", "Vol clustering diff", "lower_better"),
        ("cross_corr_diff", "Cross-corr diff", "lower_better"),
    ]

    print(f"  {'Metric':40s} {'Phase 0':>12s} {'Phase 0.5':>12s} {'Target':>15s}")
    print("  " + "-" * 85)
    for key, label, target in comparison_keys:
        v0 = phase0.get(key, float("nan"))
        v05 = phase05.get(key, float("nan"))
        print(f"  {label:40s} {v0:12.6f} {v05:12.6f} {target:>15s}")

    targets_met = 0
    targets_total = 4
    if phase05.get("synth_acf_lag1", 0) > 0.5:
        targets_met += 1
    if phase05.get("synth_vol_clustering_lag1", 0) > 0.4:
        targets_met += 1
    ratio = phase05.get("return_std_ratio", 0)
    if 1.5 <= ratio <= 2.0:
        targets_met += 1
    cone = phase05.get("cone_containment_90", 0)
    if 0.75 <= cone <= 0.90:
        targets_met += 1
    print(f"\n  Phase 0.5 targets met: {targets_met}/{targets_total}")
    if targets_met == targets_total:
        print("  >>> ALL TARGETS MET — ready for V26 Phase 1 (regime threading)")
    else:
        print("  >>> NOT ALL TARGETS MET — further tuning required")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate diffusion path realism")
    parser.add_argument("--phase0-only", action="store_true", help="Only evaluate Phase 0")
    parser.add_argument("--phase05-only", action="store_true", help="Only evaluate Phase 0.5")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    phase0_results = None
    phase05_results = None

    if not args.phase05_only:
        phase0_results = _run_phase0_eval(device)
        _print_results("PHASE 0 RESULTS", phase0_results)

    if not args.phase0_only:
        phase05_results = _run_phase05_eval(device)
        _print_results("PHASE 0.5 RESULTS", phase05_results)

    if phase0_results and phase05_results:
        _print_comparison(phase0_results, phase05_results)

    out_dir = OUTPUTS_V24_DIR / "diffusion_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    if phase0_results:
        with open(str(out_dir / "phase0_realism_report.json"), "w") as f:
            json.dump(phase0_results, f, indent=2)
        print(f"\nPhase 0 report saved: {out_dir / 'phase0_realism_report.json'}")

    if phase05_results:
        with open(str(out_dir / "phase05_realism_report.json"), "w") as f:
            json.dump(phase05_results, f, indent=2)
        print(f"Phase 0.5 report saved: {out_dir / 'phase05_realism_report.json'}")

    if phase0_results and phase05_results:
        comparison = {"phase0": phase0_results, "phase05": phase05_results}
        with open(str(out_dir / "phase_comparison.json"), "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"Comparison saved: {out_dir / 'phase_comparison.json'}")


if __name__ == "__main__":
    main()
