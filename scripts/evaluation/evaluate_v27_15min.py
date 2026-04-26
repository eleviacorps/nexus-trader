"""Evaluation for V27 15-Minute Trade Predictor."""

import sys
import json
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v27.short_horizon_predictor import create_short_horizon_predictor
from src.v26.diffusion.regime_generator import RegimeGeneratorConfig, RegimeDiffusionPathGenerator


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("V27 15-Minute Trade Predictor Evaluation")
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

    # Create V27 predictor
    print("\nCreating V27 predictor...")
    predictor = create_short_horizon_predictor(base_gen, device=str(device))

    # Generate predictions
    print("\nGenerating predictions...")
    num_tests = 50
    predictions = []
    expires_correctly = 0
    holds = 0

    for i in range(num_tests):
        regime_probs = torch.rand(9)
        regime_probs = regime_probs / regime_probs.sum()
        past_context = torch.randn(256, 144)

        result = predictor.predict_15min_trade(
            past_context=past_context,
            regime_probs=regime_probs,
            current_price=100.0,
            steps=10,
        )
        predictions.append(result)

        if result.decision == "HOLD":
            holds += 1

        if result.is_expired():
            expires_correctly += 1

        if (i + 1) % 10 == 0:
            print(f"  Generated {i+1}/{num_tests}")

    # Compute metrics
    decisions = [p.decision for p in predictions]
    confidences = [p.confidence for p in predictions]
    durations = [p.expected_duration_min for p in predictions if p.decision != "HOLD"]

    buy_count = decisions.count("BUY")
    sell_count = decisions.count("SELL")
    hold_count = decisions.count("HOLD")

    avg_confidence = np.mean(confidences)
    avg_duration = np.mean(durations) if durations else 0.0
    trade_rate = (buy_count + sell_count) / num_tests
    hold_rate = hold_count / num_tests

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Predictions generated: {num_tests}")
    print(f"  BUY: {buy_count} ({buy_count/num_tests*100:.1f}%)")
    print(f"  SELL: {sell_count} ({sell_count/num_tests*100:.1f}%)")
    print(f"  HOLD: {hold_count} ({hold_count/num_tests*100:.1f}%)")
    print(f"\nAverage confidence: {avg_confidence:.4f}")
    print(f"Average trade duration: {avg_duration:.1f} min")
    print(f"Trade rate: {trade_rate:.4f}")
    print(f"Hold rate: {hold_rate:.4f}")

    # Check validity targets
    targets_met = (
        avg_confidence > 0.50 and
        1 <= avg_duration <= 15 and
        trade_rate >= 0.20 and
        hold_rate >= 0.10
    )

    print("\n" + ("SUCCESS" if targets_met else "NEEDS IMPROVEMENT"))

    # Save report
    report = {
        "num_predictions": num_tests,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "hold_count": hold_count,
        "avg_confidence": float(avg_confidence),
        "avg_duration_min": float(avg_duration),
        "trade_rate": float(trade_rate),
        "hold_rate": float(hold_rate),
        "status": "SUCCESS" if targets_met else "NEEDS IMPROVEMENT",
    }

    Path("outputs/v27").mkdir(parents=True, exist_ok=True)
    with open("outputs/v27/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Save latest prediction
    if predictions:
        latest = predictions[-1]
        with open("outputs/v27/latest_prediction.json", "w") as f:
            json.dump(latest.to_dict(), f, indent=2)

    print(f"\nReport saved to: outputs/v27/evaluation_report.json")


if __name__ == "__main__":
    main()
