from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from config.project_config import (  # noqa: E402
    CALIBRATION_REPORT_PATH,
    FEATURE_IMPORTANCE_REPORT_PATH,
    FINAL_TFT_METRICS_PATH,
    FUSED_FEATURE_MATRIX_PATH,
    LEGACY_TFT_CHECKPOINT_PATH,
    LOOKAHEAD,
    MODEL_MANIFEST_PATH,
    SEQUENCE_LEN,
    TARGETS_PATH,
    TEST_SPLIT,
    TRAINING_SUMMARY_PATH,
    TFT_CHECKPOINT_PATH,
    TRAIN_SPLIT,
    VAL_SPLIT,
)
from src.data.fused_dataset import FusedSequenceDataset, split_row_slices  # noqa: E402
from src.models.nexus_tft import (  # noqa: E402
    NexusTFT,
    NexusTFTConfig,
    load_checkpoint_with_expansion,
    summarize_feature_importance,
)
from src.training.train_tft import (  # noqa: E402
    TrainingConfig,
    build_calibration_report,
    build_optimizer,
    collect_binary_metrics,
    evaluate_binary_model,
    find_optimal_threshold,
    save_feature_importance_report,
    save_json_report,
    train_binary_model,
)
from src.utils.device import get_torch_device  # noqa: E402

try:
    import torch  # type: ignore  # noqa: E402
    from torch.utils.data import DataLoader  # type: ignore  # noqa: E402
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"PyTorch is required for training: {exc}")


def build_loaders(feature_path: Path, target_path: Path, batch_size: int):
    total_rows = len(np.load(target_path, mmap_mode="r"))
    train_slice, val_slice, test_slice = split_row_slices(total_rows, SEQUENCE_LEN, TRAIN_SPLIT, VAL_SPLIT)
    train_ds = FusedSequenceDataset(feature_path, target_path, sequence_len=SEQUENCE_LEN, row_slice=train_slice)
    val_ds = FusedSequenceDataset(feature_path, target_path, sequence_len=SEQUENCE_LEN, row_slice=val_slice)
    test_ds = FusedSequenceDataset(feature_path, target_path, sequence_len=SEQUENCE_LEN, row_slice=test_slice)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the fused Nexus TFT model.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sample-limit", type=int, default=0, help="Optional cap for quick smoke runs.")
    parser.add_argument("--skip-checkpoint", action="store_true")
    parser.add_argument("--metric", default="accuracy", choices=["accuracy", "f1"], help="Validation metric for threshold tuning.")
    args = parser.parse_args()

    feature_path = FUSED_FEATURE_MATRIX_PATH
    target_path = TARGETS_PATH
    if not feature_path.exists() or not target_path.exists():
        raise FileNotFoundError("Missing fused artifacts. Run scripts/build_fused_artifacts.py first.")

    if args.sample_limit > 0:
        features = np.load(feature_path, mmap_mode="r")[: args.sample_limit]
        targets = np.load(target_path, mmap_mode="r")[: args.sample_limit]
        feature_path = feature_path.with_name("fused_features.sample.npy")
        target_path = target_path.with_name("targets.sample.npy")
        np.save(feature_path, np.asarray(features, dtype=np.float32))
        np.save(target_path, np.asarray(targets, dtype=np.float32))

    train_loader, val_loader, test_loader = build_loaders(feature_path, target_path, args.batch_size)
    device = get_torch_device()

    model = NexusTFT(NexusTFTConfig()).to(device)
    if LEGACY_TFT_CHECKPOINT_PATH.exists() and not args.skip_checkpoint:
        load_checkpoint_with_expansion(model, LEGACY_TFT_CHECKPOINT_PATH)

    optimizer = build_optimizer(model)
    training_config = TrainingConfig(epochs=args.epochs, batch_size=args.batch_size)
    model, history, best_val_metrics = train_binary_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        optimizer=optimizer,
        epochs=training_config.epochs,
        patience=training_config.patience,
    )
    val_metrics, val_targets, val_probabilities = evaluate_binary_model(model, val_loader, device)
    threshold_selection = find_optimal_threshold(val_targets, val_probabilities, metric=args.metric)
    threshold = float(threshold_selection["threshold"])
    calibrated_val_metrics = collect_binary_metrics(val_targets, val_probabilities, threshold=threshold)
    calibration_report = {
        "selection": threshold_selection,
        "validation_curve": build_calibration_report(val_targets, val_probabilities),
    }

    test_metrics, test_targets, test_probabilities = evaluate_binary_model(model, test_loader, device, threshold=threshold)
    calibration_report["test_curve"] = build_calibration_report(test_targets, test_probabilities)

    TFT_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "history": history,
        "best_val_metrics": best_val_metrics,
        "val_metrics": calibrated_val_metrics,
        "test_metrics": test_metrics,
        "classification_threshold": threshold,
        "sequence_len": SEQUENCE_LEN,
        "feature_dim": model.config.input_dim,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(checkpoint_payload, TFT_CHECKPOINT_PATH)

    feature_names = [f"f{i}" for i in range(model.config.input_dim)]
    with torch.no_grad():
        sample_batch = next(iter(val_loader))[0].to(device)
        _, importances = model(sample_batch, return_feature_importance=True)
    importance_report = summarize_feature_importance(feature_names, importances.detach().cpu().numpy())
    save_feature_importance_report(FEATURE_IMPORTANCE_REPORT_PATH, importance_report)

    summary = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "best_val_metrics": best_val_metrics,
        "val_metrics": calibrated_val_metrics,
        "test_metrics": test_metrics,
        "history_tail": history[-5:],
        "sample_limit": args.sample_limit,
        "classification_threshold": threshold,
        "threshold_metric": args.metric,
    }
    metrics_report = {
        "validation": calibrated_val_metrics,
        "test": test_metrics,
        "threshold_selection": threshold_selection,
    }
    manifest = {
        "model_name": "nexus-trader-tft",
        "checkpoint_path": str(TFT_CHECKPOINT_PATH),
        "sequence_len": SEQUENCE_LEN,
        "feature_dim": model.config.input_dim,
        "lookahead": LOOKAHEAD,
        "classification_threshold": threshold,
        "metrics_path": str(FINAL_TFT_METRICS_PATH),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    save_json_report(TRAINING_SUMMARY_PATH, summary)
    save_json_report(FINAL_TFT_METRICS_PATH, metrics_report)
    save_json_report(CALIBRATION_REPORT_PATH, calibration_report)
    save_json_report(MODEL_MANIFEST_PATH, manifest)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
