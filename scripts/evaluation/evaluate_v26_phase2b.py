"""Evaluation script for Phase 2B chained horizon generator."""

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v26.diffusion.chained_horizon_generator import create_chained_generator
from src.v26.diffusion.regime_generator import RegimeGeneratorConfig, RegimeDiffusionPathGenerator


def evaluate_realism(paths: np.ndarray) -> dict:
    """Evaluate path realism."""
    if paths.ndim == 3:
        paths = paths.reshape(-1, paths.shape[-1])

    returns = np.diff(paths, axis=-1)
    volatility = returns.std(axis=0)

    # ACF at lag 1
    acf_lag1 = np.corrcoef(paths[:, :-1], paths[:, 1:])[0, 1]

    # Vol clustering
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
    print("V26 Phase 2B Evaluation")
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

    # Create chained generator
    print("\nCreating chained generator...")
    chained_gen = create_chained_generator(base_gen, device=str(device))
    chained_gen.to(device)

    # Generate test samples
    print("\nGenerating test paths...")
    num_tests = 50
    results = []

    for i in range(num_tests):
        regime_probs = torch.rand(1, 9)
        regime_probs = regime_probs / regime_probs.sum()
        past_context = torch.randn(1, 256, 144)  # (B, T, C)

        with torch.no_grad():
            result = chained_gen.generate_multi_horizon(
                world_state=None,
                past_context=past_context,
                regime_probs=regime_probs,
                steps=10,  # Fast for eval
            )
        results.append(result)

        if (i + 1) % 10 == 0:
            print(f"  Generated {i+1}/{num_tests}")

    # Aggregate metrics
    all_short = torch.cat([r["short"] for r in results], dim=0)
    all_medium = torch.cat([r["medium"] for r in results], dim=0)
    all_long = torch.cat([r["long"] for r in results], dim=0)

    short_np = all_short.cpu().numpy()
    medium_np = all_medium.cpu().numpy()
    long_np = all_long.cpu().numpy()

    # Compute metrics
    realism_short = evaluate_realism(short_np)
    realism_medium = evaluate_realism(medium_np)
    realism_long = evaluate_realism(long_np)
    realism_avg = {
        "volatility": np.mean([realism_short["volatility"], realism_medium["volatility"], realism_long["volatility"]]),
        "acf_lag1": np.mean([realism_short["acf_lag1"], realism_medium["acf_lag1"], realism_long["acf_lag1"]]),
        "vol_clustering": np.mean([realism_short["vol_clustering"], realism_medium["vol_clustering"], realism_long["vol_clustering"]]),
    }

    realism_score = 0.5 * max(0, 1 - abs(realism_avg["acf_lag1"])) + 0.5 * max(0, 1 - abs(realism_avg["vol_clustering"]))

    # Continuity scores
    sm_continuity = np.mean([r["short_med_continuity"] for r in results])
    ml_continuity = np.mean([r["med_long_continuity"] for r in results])
    avg_continuity = np.mean([r["avg_continuity"] for r in results])

    # Long-horizon coherence
    long_coherence = np.corrcoef(
        all_short[:, :, -1].mean(dim=1).cpu().numpy(),
        all_long[:, :, 0].mean(dim=1).cpu().numpy()
    )[0, 1]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"realism_score: {realism_score:.4f}")
    print(f"  - short: acf={realism_short['acf_lag1']:.3f}, vol={realism_short['vol_clustering']:.3f}")
    print(f"  - medium: acf={realism_medium['acf_lag1']:.3f}, vol={realism_medium['vol_clustering']:.3f}")
    print(f"  - long: acf={realism_long['acf_lag1']:.3f}, vol={realism_long['vol_clustering']:.3f}")
    print(f"\nboundary_continuity:")
    print(f"  - short->medium: {sm_continuity:.4f}")
    print(f"  - medium->long: {ml_continuity:.4f}")
    print(f"  - average: {avg_continuity:.4f}")
    print(f"\nlong_horizon_coherence: {long_coherence:.4f}")

    # Check success
    success = (
        realism_score > 0.50 and
        avg_continuity > 0.80 and
        long_coherence > 0.30
    )

    print("\n" + ("SUCCESS" if success else "NEEDS IMPROVEMENT"))

    # Save results
    import json
    report = {
        "realism_score": float(realism_score),
        "boundary_continuity": float(avg_continuity),
        "short_med_continuity": float(sm_continuity),
        "med_long_continuity": float(ml_continuity),
        "long_horizon_coherence": float(long_coherence),
        "status": "SUCCESS" if success else "NEEDS IMPROVEMENT",
    }

    Path("outputs/v26").mkdir(parents=True, exist_ok=True)
    with open("outputs/v26/phase2b_evaluation.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: outputs/v26/phase2b_evaluation.json")


if __name__ == "__main__":
    main()
