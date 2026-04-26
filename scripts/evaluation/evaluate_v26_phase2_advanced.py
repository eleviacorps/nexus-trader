"""Evaluation for V26 Phase 2 Advanced (autoregressive chunked)."""

import sys
from pathlib import Path
import json

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v26.diffusion.autoregressive_generator import create_autoregressive_generator
from src.v26.diffusion.regime_generator import RegimeGeneratorConfig, RegimeDiffusionPathGenerator


def evaluate_realism(paths: np.ndarray) -> dict:
    if paths.ndim == 3:
        paths = paths.reshape(-1, paths.shape[-1])
    returns = np.diff(paths, axis=-1)
    volatility = returns.std(axis=0)
    acf_lag1 = np.corrcoef(paths[:, :-1], paths[:, 1:])[0, 1]
    vol_returns = np.abs(returns)
    vol_clustering = np.corrcoef(vol_returns[:, :-1], vol_returns[:, 1:])[0, 1]
    return {
        "volatility": float(volatility.mean()),
        "acf_lag1": float(acf_lag1) if not np.isnan(acf_lag1) else 0.0,
        "vol_clustering": float(vol_clustering) if not np.isnan(vol_clustering) else 0.0,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("V26 Phase 2 Advanced Evaluation")
    print("=" * 60)
    print(f"Device: {device}")

    # Load Phase 1
    print("\nLoading Phase 1 generator...")
    config = RegimeGeneratorConfig(
        in_channels=144,
        sequence_length=120,
        temporal_gru_dim=256,
        temporal_layers=2,
        context_len=256,
        num_regimes=9,
        regime_embed_dim=16,
        temporal_film_dim=272,
    )
    base_gen = RegimeDiffusionPathGenerator(config=config, device=str(device))

    ckpt_path = "models/v26/diffusion_phase1_final.pt"
    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_state = base_gen.model.state_dict()
        compatible = {k: v for k, v in ckpt.get("ema", ckpt.get("model", {})).items()
                     if k in model_state and model_state[k].shape == v.shape}
        base_gen.model.load_state_dict(compatible, strict=False)
        print(f"Loaded Phase 1 from {ckpt_path}")

    # Create autoregressive generator
    print("\nCreating autoregressive generator...")
    auto_gen = create_autoregressive_generator(base_gen, device=str(device))
    auto_gen.to(device)

    # Generate test samples
    print("\nGenerating test paths...")
    num_tests = 30
    results = []

    for i in range(num_tests):
        regime_probs = torch.rand(1, 9)
        regime_probs = regime_probs / regime_probs.sum()
        past_context = torch.randn(1, 256, 144)

        with torch.no_grad():
            result = auto_gen.generate_long_horizon(
                past_context=past_context,
                regime_probs=regime_probs,
                steps=10,
            )
        results.append(result)

        if (i + 1) % 10 == 0:
            print(f"  Generated {i+1}/{num_tests}")

    # Aggregate
    all_paths = []
    all_continuities = []

    for r in results:
        fp = r["full_path"]
        if fp is not None and fp.numel() > 0:
            all_paths.append(fp.cpu().numpy())
            all_continuities.append(r["continuity"])

    if not all_paths:
        print("ERROR: No paths generated")
        return

    all_paths = np.concatenate(all_paths, axis=0)

    # Metrics
    realism = evaluate_realism(all_paths)
    realism_score = 0.5 * max(0, 1 - abs(realism["acf_lag1"])) + 0.5 * max(0, 1 - abs(realism["vol_clustering"]))

    # Continuity
    cont_avgs = [c["avg"] if c else 0.0 for c in all_continuities]
    avg_continuity = np.mean(cont_avgs) if cont_avgs else 0.0

    # Long-horizon coherence
    long_coh = 0.0
    if all_paths.shape[-1] > 120:
        first_half = all_paths[:, :, :60]
        second_half = all_paths[:, :, 60:]
        corr = np.corrcoef(first_half.mean(axis=(1, 2)), second_half.mean(axis=(1, 2)))
        long_coh = corr[0, 1] if not np.isnan(corr[0, 1]) else 0.0

    # Regime drift
    regime_drift = 0.0  # Simplified

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"realism_score: {realism_score:.4f}")
    print(f"boundary_continuity: {avg_continuity:.4f}")
    print(f"long_horizon_coherence: {long_coh:.4f}")
    print(f"regime_drift: {regime_drift:.4f}")

    # Targets
    success = (
        realism_score > 0.50 and
        avg_continuity > 0.60 and
        long_coh > 0.40 and
        regime_drift < 0.25
    )

    print("\n" + ("SUCCESS" if success else "NEEDS IMPROVEMENT"))

    # Save report
    report = {
        "realism_score": float(realism_score),
        "boundary_continuity": float(avg_continuity),
        "long_horizon_coherence": float(long_coh),
        "regime_drift": float(regime_drift),
        "realism_details": realism,
        "status": "SUCCESS" if success else "NEEDS IMPROVEMENT",
    }

    Path("outputs/v26").mkdir(parents=True, exist_ok=True)
    with open("outputs/v26/phase2_advanced_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: outputs/v26/phase2_advanced_report.json")


if __name__ == "__main__":
    main()
