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
    GATE_CONTEXT_PATH,
    LEGACY_TFT_CHECKPOINT_PATH,
    LOOKAHEAD,
    MODEL_MANIFEST_PATH,
    PRECISION_GATE_PATH,
    SAMPLE_WEIGHTS_PATH,
    SEQUENCE_LEN,
    SIM_CONFIDENCE_PATH,
    SIM_TARGETS_PATH,
    TARGETS_PATH,
    TARGETS_MULTIHORIZON_PATH,
    TEST_SPLIT,
    TEST_YEARS,
    TFT_MODEL_DIR,
    TRAINING_SUMMARY_PATH,
    TFT_CHECKPOINT_PATH,
    TRAIN_SPLIT,
    TRAIN_YEARS,
    VAL_SPLIT,
    VAL_YEARS,
)
from src.data.fused_dataset import FusedMultiHorizonSequenceDataset, FusedSequenceDataset, split_row_slices  # noqa: E402
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
    collect_multihorizon_metrics,
    evaluate_binary_model,
    evaluate_multihorizon_model,
    find_optimal_threshold,
    train_multihorizon_model,
    train_precision_gate,
    apply_precision_gate,
    save_feature_importance_report,
    save_json_report,
    save_training_config,
    train_binary_model,
)
from src.training.meta_gate import (  # noqa: E402
    apply_meta_gate,
    combine_gate_scores,
    save_meta_gate,
    train_meta_gate,
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


def tagged_path(path: Path, run_tag: str) -> Path:
    if not run_tag:
        return path
    return path.with_name(f"{path.stem}_{run_tag}{path.suffix}")


def slice_gate_context(path: Path, row_slice, sequence_len: int) -> np.ndarray | None:
    if not path.exists():
        return None
    context = np.load(path, mmap_mode='r')
    start = row_slice.start + sequence_len - 1
    stop = row_slice.stop + sequence_len - 1
    return np.asarray(context[start:stop], dtype=np.float32)


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


def build_multihorizon_loaders(
    feature_path: Path,
    target_bundle_path: Path,
    target_keys: list[str],
    batch_size: int,
    sequence_len: int,
    train_slice,
    val_slice,
    test_slice,
    sim_target_path: Path | None = None,
    sim_confidence_path: Path | None = None,
    sample_weight_path: Path | None = None,
):
    train_ds = FusedMultiHorizonSequenceDataset(
        feature_path,
        target_bundle_path,
        target_keys=target_keys,
        sequence_len=sequence_len,
        row_slice=train_slice,
        sim_target_path=sim_target_path,
        sim_confidence_path=sim_confidence_path,
        sample_weight_path=sample_weight_path,
    )
    val_ds = FusedMultiHorizonSequenceDataset(
        feature_path,
        target_bundle_path,
        target_keys=target_keys,
        sequence_len=sequence_len,
        row_slice=val_slice,
        sim_target_path=sim_target_path,
        sim_confidence_path=sim_confidence_path,
        sample_weight_path=sample_weight_path,
    )
    test_ds = FusedMultiHorizonSequenceDataset(
        feature_path,
        target_bundle_path,
        target_keys=target_keys,
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
    parser.add_argument('--run-tag', default='', help='Optional tag to save an alternate checkpoint/manifest set.')
    parser.add_argument('--disable-multihorizon', action='store_true', help='Fallback to the old single-target training path.')
    args = parser.parse_args()

    feature_path = FUSED_FEATURE_MATRIX_PATH
    target_path = TARGETS_PATH
    if not feature_path.exists() or not target_path.exists():
        raise FileNotFoundError('Missing fused artifacts. Run scripts/build_fused_artifacts.py first.')
    multi_target_path = TARGETS_MULTIHORIZON_PATH if TARGETS_MULTIHORIZON_PATH.exists() and not args.disable_multihorizon else None
    horizon_labels = ['5m', '10m', '15m', '30m']
    direction_keys = [f'target_{label}' for label in horizon_labels]
    hold_keys = [f'hold_{label}' for label in horizon_labels]
    confidence_keys = [f'confidence_{label}' for label in horizon_labels]
    horizon_keys = direction_keys + hold_keys + confidence_keys
    use_multihorizon = multi_target_path is not None

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
        gate_context_path = GATE_CONTEXT_PATH
        if gate_context_path.exists():
            gate_context = np.load(gate_context_path, mmap_mode='r')[: args.sample_limit]
            gate_context_path = save_sample_artifact(target_path.with_name('gate_context.sample.npy'), gate_context)
        else:
            gate_context_path = GATE_CONTEXT_PATH
        if use_multihorizon and multi_target_path is not None:
            bundle = np.load(multi_target_path, mmap_mode='r')
            sample_bundle_path = target_path.with_name('targets_multihorizon.sample.npz')
            np.savez(
                sample_bundle_path,
                **{key: np.asarray(bundle[key][: args.sample_limit], dtype=np.float32) for key in bundle.files},
            )
            multi_target_path = sample_bundle_path
    else:
        gate_context_path = GATE_CONTEXT_PATH

    total_rows = len(np.load(target_path, mmap_mode='r'))
    train_slice, val_slice, test_slice, split_report = resolve_slices(total_rows, args.sequence_len, args.split_mode, train_years, val_years, test_years)

    if use_multihorizon and multi_target_path is not None:
        bundle = np.load(multi_target_path, mmap_mode='r')
        missing_horizons = [key for key in horizon_keys if key not in bundle.files]
        if missing_horizons:
            raise KeyError(f'Missing multi-horizon targets: {missing_horizons}')
        train_loader, val_loader, test_loader = build_multihorizon_loaders(
            feature_path,
            multi_target_path,
            target_keys=horizon_keys,
            batch_size=args.batch_size,
            sequence_len=args.sequence_len,
            train_slice=train_slice,
            val_slice=val_slice,
            test_slice=test_slice,
            sim_target_path=sim_target_path,
            sim_confidence_path=sim_confidence_path,
            sample_weight_path=sample_weight_path,
        )
    else:
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
        output_dim=len(horizon_keys) if use_multihorizon else 1,
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
    if use_multihorizon:
        gate_context_val = slice_gate_context(gate_context_path, val_slice, args.sequence_len)
        gate_context_test = slice_gate_context(gate_context_path, test_slice, args.sequence_len)
        model, history, best_val_metrics = train_multihorizon_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            optimizer=optimizer,
            epochs=training_config.epochs,
            patience=training_config.patience,
            selection_metric=args.selection_metric,
            horizon_labels=horizon_labels,
        )
        val_metrics, val_targets, val_probabilities = evaluate_multihorizon_model(model, val_loader, device, horizon_labels=horizon_labels)
        threshold_selection = find_optimal_threshold(val_targets[:, 2], val_probabilities[:, 2], metric=args.metric)
        threshold = float(threshold_selection['threshold'])
        horizon_thresholds = [0.5, 0.5, threshold, threshold]
        calibrated_val_metrics = collect_multihorizon_metrics(val_targets, val_probabilities, thresholds=horizon_thresholds, horizon_labels=horizon_labels)
        gate = train_precision_gate(
            val_probabilities,
            val_targets,
            context_features=gate_context_val,
            threshold=threshold,
        )
        gate_probabilities_val = apply_precision_gate(val_probabilities, gate, context_features=gate_context_val)
        test_metrics, test_targets, test_probabilities = evaluate_multihorizon_model(model, test_loader, device, thresholds=horizon_thresholds, horizon_labels=horizon_labels)
        gate_probabilities_test = apply_precision_gate(test_probabilities, gate, context_features=gate_context_test)
        meta_gate = train_meta_gate(
            val_probabilities,
            val_targets,
            context_features=gate_context_val,
            threshold=threshold,
        )
        meta_gate_probabilities_val = apply_meta_gate(val_probabilities, meta_gate, context_features=gate_context_val)
        meta_gate_probabilities_test = apply_meta_gate(test_probabilities, meta_gate, context_features=gate_context_test)
        combined_gate_probabilities_val = combine_gate_scores(gate_probabilities_val, meta_gate_probabilities_val)
        combined_gate_probabilities_test = combine_gate_scores(gate_probabilities_test, meta_gate_probabilities_test)
        calibration_report = {
            'selection': threshold_selection,
            'validation_curve': build_calibration_report(val_targets[:, 2], val_probabilities[:, 2]),
            'test_curve': build_calibration_report(test_targets[:, 2], test_probabilities[:, 2]),
            'precision_gate': {
                'validation_curve': build_calibration_report(
                    ((val_probabilities[:, 2] >= threshold).astype(np.float32) == val_targets[:, 2].astype(np.float32)).astype(np.float32),
                    gate_probabilities_val,
                ),
                'test_curve': build_calibration_report(
                    ((test_probabilities[:, 2] >= threshold).astype(np.float32) == test_targets[:, 2].astype(np.float32)).astype(np.float32),
                    gate_probabilities_test,
                ),
            },
            'meta_gate': {
                'provider': meta_gate.get('provider', 'none'),
                'validation_curve': build_calibration_report(
                    ((val_probabilities[:, 2] >= threshold).astype(np.float32) == val_targets[:, 2].astype(np.float32)).astype(np.float32),
                    meta_gate_probabilities_val if meta_gate_probabilities_val is not None else gate_probabilities_val,
                ),
                'test_curve': build_calibration_report(
                    ((test_probabilities[:, 2] >= threshold).astype(np.float32) == test_targets[:, 2].astype(np.float32)).astype(np.float32),
                    meta_gate_probabilities_test if meta_gate_probabilities_test is not None else gate_probabilities_test,
                ),
            },
            'combined_gate': {
                'validation_curve': build_calibration_report(
                    ((val_probabilities[:, 2] >= threshold).astype(np.float32) == val_targets[:, 2].astype(np.float32)).astype(np.float32),
                    combined_gate_probabilities_val if combined_gate_probabilities_val is not None else gate_probabilities_val,
                ),
                'test_curve': build_calibration_report(
                    ((test_probabilities[:, 2] >= threshold).astype(np.float32) == test_targets[:, 2].astype(np.float32)).astype(np.float32),
                    combined_gate_probabilities_test if combined_gate_probabilities_test is not None else gate_probabilities_test,
                ),
            },
        }
    else:
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
        gate = None
        horizon_thresholds = [threshold]

    checkpoint_path = tagged_path(TFT_CHECKPOINT_PATH, args.run_tag)
    metrics_path = tagged_path(FINAL_TFT_METRICS_PATH, args.run_tag)
    summary_path = tagged_path(TRAINING_SUMMARY_PATH, args.run_tag)
    calibration_path = tagged_path(CALIBRATION_REPORT_PATH, args.run_tag)
    importance_path = tagged_path(FEATURE_IMPORTANCE_REPORT_PATH, args.run_tag)
    manifest_path = tagged_path(MODEL_MANIFEST_PATH, args.run_tag)
    precision_gate_path = tagged_path(PRECISION_GATE_PATH, args.run_tag)
    meta_gate_path = tagged_path(META_GATE_PATH, args.run_tag)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        'model_state_dict': model.state_dict(),
        'history': history,
        'best_val_metrics': best_val_metrics,
        'val_metrics': calibrated_val_metrics,
        'test_metrics': test_metrics,
        'classification_threshold': threshold,
        'classification_thresholds': horizon_thresholds,
        'sequence_len': args.sequence_len,
        'feature_dim': model.config.input_dim,
        'model_config': vars(model.config),
        'split_report': split_report,
        'horizon_labels': horizon_labels if use_multihorizon else ['5m'],
        'output_labels': horizon_keys if use_multihorizon else ['target_5m'],
        'multi_horizon': bool(use_multihorizon),
        'generated_at': datetime.now(timezone.utc).isoformat(),
    }
    torch.save(checkpoint_payload, checkpoint_path)

    feature_names = [f'f{i}' for i in range(model.config.input_dim)]
    with torch.no_grad():
        first_batch = next(iter(val_loader))
        sample_batch = first_batch[0].to(device)
        _, importances = model(sample_batch, return_feature_importance=True)
    importance_report = summarize_feature_importance(feature_names, importances.detach().cpu().numpy())
    save_feature_importance_report(importance_path, importance_report)
    save_training_config(tagged_path(TRAINING_SUMMARY_PATH.with_name('training_config.json'), args.run_tag), training_config)

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
        'run_tag': args.run_tag,
        'multi_horizon': bool(use_multihorizon),
        'horizon_labels': horizon_labels if use_multihorizon else ['5m'],
        'output_labels': horizon_keys if use_multihorizon else ['target_5m'],
        'precision_gate_path': str(precision_gate_path) if gate is not None else '',
        'meta_gate_path': str(meta_gate_path) if meta_gate is not None and meta_gate.get('available', False) else '',
        'meta_gate_provider': meta_gate.get('provider', '') if meta_gate is not None else '',
        'gate_context_path': str(gate_context_path) if gate is not None and gate_context_path.exists() else '',
    }
    metrics_report = {
        'validation': calibrated_val_metrics,
        'test': test_metrics,
        'threshold_selection': threshold_selection,
        'split_report': split_report,
        'multi_horizon': bool(use_multihorizon),
    }
    manifest = {
        'model_name': 'nexus-trader-tft',
        'checkpoint_path': str(checkpoint_path),
        'sequence_len': args.sequence_len,
        'feature_dim': model.config.input_dim,
        'lookahead': LOOKAHEAD,
        'classification_threshold': threshold,
        'classification_thresholds': horizon_thresholds,
        'simulation_supervision': bool(sim_target_path and sim_confidence_path),
        'sample_weighting': bool(sample_weight_path),
        'model_config': vars(model.config),
        'split_report': split_report,
        'metrics_path': str(metrics_path),
        'horizon_labels': horizon_labels if use_multihorizon else ['5m'],
        'output_labels': horizon_keys if use_multihorizon else ['target_5m'],
        'primary_horizon': '15m' if use_multihorizon else '5m',
        'multi_horizon': bool(use_multihorizon),
        'precision_gate_path': str(precision_gate_path) if gate is not None else '',
        'meta_gate_path': str(meta_gate_path) if meta_gate is not None and meta_gate.get('available', False) else '',
        'meta_gate_provider': meta_gate.get('provider', '') if meta_gate is not None else '',
        'gate_context_path': str(gate_context_path) if gate is not None and gate_context_path.exists() else '',
        'run_tag': args.run_tag,
        'generated_at': datetime.now(timezone.utc).isoformat(),
    }
    save_json_report(summary_path, summary)
    save_json_report(metrics_path, metrics_report)
    save_json_report(calibration_path, calibration_report)
    save_json_report(manifest_path, manifest)
    if gate is not None:
        save_json_report(precision_gate_path, gate)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())










