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
    FUSED_TIMESTAMPS_PATH,
    LEGACY_TFT_CHECKPOINT_PATH,
    LOOKAHEAD,
    MODEL_MANIFEST_PATH,
    SAMPLE_WEIGHTS_PATH,
    SEQUENCE_LEN,
    SIM_CONFIDENCE_PATH,
    SIM_TARGETS_PATH,
    TARGETS_PATH,
    TEST_SPLIT,
    TEST_YEARS,
    TRAINING_SUMMARY_PATH,
    TFT_CHECKPOINT_PATH,
    TRAIN_SPLIT,
    TRAIN_YEARS,
    VAL_SPLIT,
    VAL_YEARS,
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
    save_training_config,
    train_binary_model,
)
from src.utils.device import get_torch_device  # noqa: E402
from src.utils.training_splits import split_by_years  # noqa: E402

try:
    import torch  # type: ignore  # noqa: E402
    from torch.utils.data import DataLoader  # type: ignore  # noqa: E402
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"PyTorch is required for training: {exc}")


def parse_year_list(text: str | None, default: tuple[int, ...]) -> list[int]:
    if text is None or not text.strip():
        return [int(year) for year in default]
    return [int(part.strip()) for part in text.split(',') if part.strip()]


def save_sample_artifact(path: Path, values: np.ndarray) -> Path:
    np.save(path, np.asarray(values, dtype=np.float32))
    return path


def build_loaders(
    feature_path: Path,
    target_path: Path,
    batch_size: int,
    sequence_len: int,
    train_slice,
    val_slice,
    test_slice,
    sim_target_path: Path | None = None,
    sim_confidence_path: Path | None = None,
    sample_weight_path: Path | None = None,
):
    train_ds = FusedSequenceDataset(
        feature_path,
        target_path,
        sequence_len=sequence_len,
        row_slice=train_slice,
        sim_target_path=sim_target_path,
        sim_confidence_path=sim_confidence_path,
        sample_weight_path=sample_weight_path,
    )
    val_ds = FusedSequenceDataset(
        feature_path,
        target_path,
        sequence_len=sequence_len,
        row_slice=val_slice,
        sim_target_path=sim_target_path,
        sim_confidence_path=sim_confidence_path,
        sample_weight_path=sample_weight_path,
    )
    test_ds = FusedSequenceDataset(
        feature_path,
        target_path,
        sequence_len=sequence_len,
        row_slice=test_slice,
        sim_target_path=sim_target_path,
        sim_confidence_path=sim_confidence_path,
        sample_weight_path=sample_weight_path,
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True),
    )


def resolve_slices(total_rows: int, sequence_len: int, split_mode: str, train_years: list[int], val_years: list[int], test_years: list[int]):
    if split_mode == 'ratio':
        train_slice, val_slice, test_slice = split_row_slices(total_rows, sequence_len, TRAIN_SPLIT, VAL_SPLIT)
        return train_slice, val_slice, test_slice, {'mode': 'ratio', 'train_split': TRAIN_SPLIT, 'val_split': VAL_SPLIT, 'test_split': TEST_SPLIT}

    if not FUSED_TIMESTAMPS_PATH.exists():
        raise FileNotFoundError('timestamps.npy is required for year-based splits. Run scripts/build_fused_artifacts.py first.')
    timestamps = np.load(FUSED_TIMESTAMPS_PATH, mmap_mode='r')[:total_rows]
    split_config = split_by_years(timestamps, sequence_len, train_years, val_years, test_years)
    return split_config.train, split_config.val, split_config.test, {
        'mode': split_config.mode,
        'train_years': train_years,
        'val_years': val_years,
        'test_years': test_years,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Train the fused Nexus TFT model.')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--sample-limit', type=int, default=0, help='Optional cap for quick smoke runs.')
    parser.add_argument('--skip-checkpoint', action='store_true')
    parser.add_argument('--metric', default='accuracy', choices=['accuracy', 'f1'], help='Validation metric for threshold tuning.')
    parser.add_argument('--selection-metric', default='accuracy', choices=['accuracy', 'f1', 'roc_auc'], help='Metric used for early stopping/model selection.')
    parser.add_argument('--sequence-len', type=int, default=SEQUENCE_LEN)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--lstm-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--old-lr', type=float, default=1e-4)
    parser.add_argument('--new-lr', type=float, default=5e-4)
    parser.add_argument('--split-mode', default='ratio', choices=['ratio', 'year'])
    parser.add_argument('--train-years', default=None, help='Comma-separated years for training split.')
    parser.add_argument('--val-years', default=None, help='Comma-separated years for validation split.')
    parser.add_argument('--test-years', default=None, help='Comma-separated years for test split.')
    args = parser.parse_args()

    feature_path = FUSED_FEATURE_MATRIX_PATH
    target_path = TARGETS_PATH
    if not feature_path.exists() or not target_path.exists():
        raise FileNotFoundError('Missing fused artifacts. Run scripts/build_fused_artifacts.py first.')

    sim_target_path = SIM_TARGETS_PATH if SIM_TARGETS_PATH.exists() else None
    sim_confidence_path = SIM_CONFIDENCE_PATH if SIM_CONFIDENCE_PATH.exists() else None
    sample_weight_path = SAMPLE_WEIGHTS_PATH if SAMPLE_WEIGHTS_PATH.exists() else None
    if sim_target_path is None or sim_confidence_path is None:
        sim_target_path = None
        sim_confidence_path = None

    train_years = parse_year_list(args.train_years, TRAIN_YEARS)
    val_years = parse_year_list(args.val_years, VAL_YEARS)
    test_years = parse_year_list(args.test_years, TEST_YEARS)

    if args.sample_limit > 0 and args.split_mode == 'year':
        raise ValueError('sample-limit with split-mode=year is not supported because it can invalidate year coverage. Use ratio splits for smoke runs.')

    if args.sample_limit > 0:
        features = np.load(feature_path, mmap_mode='r')[: args.sample_limit]
        targets = np.load(target_path, mmap_mode='r')[: args.sample_limit]
        feature_path = feature_path.with_name('fused_features.sample.npy')
        target_path = target_path.with_name('targets.sample.npy')
        np.save(feature_path, np.asarray(features, dtype=np.float32))
        np.save(target_path, np.asarray(targets, dtype=np.float32))
        if sim_target_path is not None and sim_confidence_path is not None:
            sim_targets = np.load(sim_target_path, mmap_mode='r')[: args.sample_limit]
            sim_confidence = np.load(sim_confidence_path, mmap_mode='r')[: args.sample_limit]
            sim_target_path = save_sample_artifact(target_path.with_name('sim_targets.sample.npy'), sim_targets)
            sim_confidence_path = save_sample_artifact(target_path.with_name('sim_confidence.sample.npy'), sim_confidence)
        if sample_weight_path is not None:
            sample_weights = np.load(sample_weight_path, mmap_mode='r')[: args.sample_limit]
            sample_weight_path = save_sample_artifact(target_path.with_name('sample_weights.sample.npy'), sample_weights)

    total_rows = len(np.load(target_path, mmap_mode='r'))
    train_slice, val_slice, test_slice, split_report = resolve_slices(total_rows, args.sequence_len, args.split_mode, train_years, val_years, test_years)

    train_loader, val_loader, test_loader = build_loaders(
        feature_path,
        target_path,
        args.batch_size,
        args.sequence_len,
        train_slice,
        val_slice,
        test_slice,
        sim_target_path=sim_target_path,
        sim_confidence_path=sim_confidence_path,
        sample_weight_path=sample_weight_path,
    )
    device = get_torch_device()

    model_config = NexusTFTConfig(
        input_dim=int(np.load(feature_path, mmap_mode='r').shape[1]),
        hidden_dim=args.hidden_dim,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
    )
    model = NexusTFT(model_config).to(device)
    if LEGACY_TFT_CHECKPOINT_PATH.exists() and not args.skip_checkpoint:
        load_checkpoint_with_expansion(model, LEGACY_TFT_CHECKPOINT_PATH, new_input_dim=model_config.input_dim)

    optimizer = build_optimizer(model, old_layers_lr=args.old_lr, new_layers_lr=args.new_lr)
    training_config = TrainingConfig(
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        inherited_lr=args.old_lr,
        new_layers_lr=args.new_lr,
    )
    model, history, best_val_metrics = train_binary_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        optimizer=optimizer,
        epochs=training_config.epochs,
        patience=training_config.patience,
        selection_metric=args.selection_metric,
    )
    val_metrics, val_targets, val_probabilities = evaluate_binary_model(model, val_loader, device)
    threshold_selection = find_optimal_threshold(val_targets, val_probabilities, metric=args.metric)
    threshold = float(threshold_selection['threshold'])
    calibrated_val_metrics = collect_binary_metrics(val_targets, val_probabilities, threshold=threshold)
    calibration_report = {
        'selection': threshold_selection,
        'validation_curve': build_calibration_report(val_targets, val_probabilities),
    }

    test_metrics, test_targets, test_probabilities = evaluate_binary_model(model, test_loader, device, threshold=threshold)
    calibration_report['test_curve'] = build_calibration_report(test_targets, test_probabilities)

    TFT_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        'model_state_dict': model.state_dict(),
        'history': history,
        'best_val_metrics': best_val_metrics,
        'val_metrics': calibrated_val_metrics,
        'test_metrics': test_metrics,
        'classification_threshold': threshold,
        'sequence_len': args.sequence_len,
        'feature_dim': model.config.input_dim,
        'model_config': vars(model.config),
        'split_report': split_report,
        'generated_at': datetime.now(timezone.utc).isoformat(),
    }
    torch.save(checkpoint_payload, TFT_CHECKPOINT_PATH)

    feature_names = [f'f{i}' for i in range(model.config.input_dim)]
    with torch.no_grad():
        first_batch = next(iter(val_loader))
        sample_batch = first_batch[0].to(device)
        _, importances = model(sample_batch, return_feature_importance=True)
    importance_report = summarize_feature_importance(feature_names, importances.detach().cpu().numpy())
    save_feature_importance_report(FEATURE_IMPORTANCE_REPORT_PATH, importance_report)
    save_training_config(TRAINING_SUMMARY_PATH.with_name('training_config.json'), training_config)

    summary = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'best_val_metrics': best_val_metrics,
        'val_metrics': calibrated_val_metrics,
        'test_metrics': test_metrics,
        'history_tail': history[-5:],
        'sample_limit': args.sample_limit,
        'classification_threshold': threshold,
        'threshold_metric': args.metric,
        'selection_metric': args.selection_metric,
        'simulation_supervision': bool(sim_target_path and sim_confidence_path),
        'sample_weighting': bool(sample_weight_path),
        'sequence_len': args.sequence_len,
        'model_config': vars(model.config),
        'split_report': split_report,
    }
    metrics_report = {
        'validation': calibrated_val_metrics,
        'test': test_metrics,
        'threshold_selection': threshold_selection,
        'split_report': split_report,
    }
    manifest = {
        'model_name': 'nexus-trader-tft',
        'checkpoint_path': str(TFT_CHECKPOINT_PATH),
        'sequence_len': args.sequence_len,
        'feature_dim': model.config.input_dim,
        'lookahead': LOOKAHEAD,
        'classification_threshold': threshold,
        'simulation_supervision': bool(sim_target_path and sim_confidence_path),
        'sample_weighting': bool(sample_weight_path),
        'model_config': vars(model.config),
        'split_report': split_report,
        'metrics_path': str(FINAL_TFT_METRICS_PATH),
        'generated_at': datetime.now(timezone.utc).isoformat(),
    }
    save_json_report(TRAINING_SUMMARY_PATH, summary)
    save_json_report(FINAL_TFT_METRICS_PATH, metrics_report)
    save_json_report(CALIBRATION_REPORT_PATH, calibration_report)
    save_json_report(MODEL_MANIFEST_PATH, manifest)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
