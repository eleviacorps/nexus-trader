from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.project_config import (  # noqa: E402
    CROWD_EMBEDDINGS_NPY_PATH,
    FUSION_REPORT_PATH,
    FUSED_FEATURE_MATRIX_PATH,
    FUSED_TENSOR_PATH,
    FUSED_TIMESTAMPS_PATH,
    LEGACY_CROWD_EMBEDDINGS_NPY_PATH,
    LEGACY_NEWS_EMBEDDINGS_NPY_PATH,
    LEGACY_NEWS_EMBEDDINGS_RAW_PATH,
    LEGACY_PRICE_FEATURES_CSV,
    LEGACY_PRICE_FEATURES_PARQUET,
    LOOKAHEAD,
    NEWS_EMBEDDINGS_NPY_PATH,
    NEWS_EMBEDDINGS_RAW_PATH,
    PERSONA_OUTPUTS_PATH,
    PRICE_FEATURES_CSV_FALLBACK,
    PRICE_FEATURES_PATH,
    SAMPLE_WEIGHTS_PATH,
    SEQUENCE_LEN,
    TARGET_HOLD_MASK_PATH,
    TARGETS_PATH,
    TARGETS_MULTIHORIZON_PATH,
)
from src.pipeline.fusion import (  # noqa: E402
    FusionReport,
    build_trade_target_artifacts,
    build_sequence_tensor,
    build_fused_feature_matrix,
    extract_price_block,
    load_price_frame,
    save_fusion_report,
    save_numpy_artifact,
)


def resolve_first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    joined = ', '.join(str(path) for path in paths)
    raise FileNotFoundError(f'No artifact found in: {joined}')


def main() -> int:
    parser = argparse.ArgumentParser(description='Build fused feature artifacts from price, news, and crowd blocks.')
    parser.add_argument('--limit-rows', type=int, default=0, help='Optional cap for smoke runs.')
    parser.add_argument('--materialize-sequences', action='store_true', help='Write fused_tensor.npy for notebook/debug use.')
    parser.add_argument('--sequence-limit', type=int, default=0, help='Optional cap on sequence windows when materializing.')
    parser.add_argument('--lookahead', type=int, default=LOOKAHEAD, help='Forward horizon used for sample weighting.')
    parser.add_argument('--horizons', default='5,10,15,30', help='Comma-separated forecast horizons used for target generation.')
    parser.add_argument('--atr-multiplier', type=float, default=0.35, help='ATR percentage multiplier for hold-aware move thresholds.')
    parser.add_argument('--min-abs-return', type=float, default=4e-4, help='Absolute return floor used in target thresholding.')
    parser.add_argument('--hold-weight', type=float, default=0.35, help='Training weight assigned to hold / low-signal rows.')
    args = parser.parse_args()
    horizons = tuple(int(part.strip()) for part in args.horizons.split(',') if part.strip())
    if not horizons:
        raise ValueError('At least one horizon is required.')
    if args.lookahead not in horizons:
        horizons = tuple(sorted({*horizons, int(args.lookahead)}))

    price_path = resolve_first_existing([PRICE_FEATURES_PATH, PRICE_FEATURES_CSV_FALLBACK, LEGACY_PRICE_FEATURES_PARQUET, LEGACY_PRICE_FEATURES_CSV])
    news_path = resolve_first_existing([NEWS_EMBEDDINGS_RAW_PATH, NEWS_EMBEDDINGS_NPY_PATH, LEGACY_NEWS_EMBEDDINGS_RAW_PATH, LEGACY_NEWS_EMBEDDINGS_NPY_PATH])
    crowd_path = resolve_first_existing([CROWD_EMBEDDINGS_NPY_PATH, LEGACY_CROWD_EMBEDDINGS_NPY_PATH])

    price_frame = load_price_frame(price_path)
    price_block = extract_price_block(price_frame)
    target_artifacts = build_trade_target_artifacts(
        price_frame,
        horizons=horizons,
        primary_horizon=int(args.lookahead),
        atr_multiplier=float(args.atr_multiplier),
        min_abs_return=float(args.min_abs_return),
        hold_weight=float(args.hold_weight),
    )
    targets = target_artifacts.primary_targets
    news_block = np.load(news_path, mmap_mode='r')
    crowd_block = np.load(crowd_path, mmap_mode='r')

    row_count = min(len(price_block), len(targets), len(news_block), len(crowd_block))
    if args.limit_rows > 0:
        row_count = min(row_count, args.limit_rows)

    price_block = np.asarray(price_block[:row_count], dtype=np.float32)
    targets = np.asarray(targets[:row_count], dtype=np.float32)
    news_block = np.asarray(news_block[:row_count], dtype=np.float32)
    crowd_block = np.asarray(crowd_block[:row_count], dtype=np.float32)
    sample_weights = np.asarray(target_artifacts.sample_weights[:row_count], dtype=np.float32)
    hold_mask = np.asarray(target_artifacts.primary_hold_mask[:row_count], dtype=np.float32)

    fused = build_fused_feature_matrix(price_block, news_block, crowd_block)
    save_numpy_artifact(FUSED_FEATURE_MATRIX_PATH, fused)
    save_numpy_artifact(TARGETS_PATH, targets)
    save_numpy_artifact(SAMPLE_WEIGHTS_PATH, sample_weights)
    save_numpy_artifact(TARGET_HOLD_MASK_PATH, hold_mask)
    TARGETS_MULTIHORIZON_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        TARGETS_MULTIHORIZON_PATH,
        **{
            f'forward_return_{horizon}m': np.asarray(values[:row_count], dtype=np.float32)
            for horizon, values in target_artifacts.horizon_returns.items()
        },
        **{
            f'target_{horizon}m': (np.asarray(values[:row_count], dtype=np.float32) > 0.0).astype(np.float32)
            for horizon, values in target_artifacts.horizon_returns.items()
        },
        primary_targets=targets,
        primary_hold_mask=hold_mask,
        sample_weights=sample_weights,
    )

    timestamps = np.asarray(price_frame.index[:row_count].astype(str), dtype='<U32')
    save_numpy_artifact(FUSED_TIMESTAMPS_PATH, timestamps)

    sequence_rows = 0
    if args.materialize_sequences:
        sequence_tensor, sequence_targets = build_sequence_tensor(fused, targets, sequence_len=SEQUENCE_LEN)
        if args.sequence_limit > 0:
            sequence_tensor = sequence_tensor[: args.sequence_limit]
            sequence_targets = sequence_targets[: args.sequence_limit]
        save_numpy_artifact(FUSED_TENSOR_PATH, sequence_tensor)
        save_numpy_artifact(TARGETS_PATH.with_name('targets_sequence.npy'), sequence_targets)
        sequence_rows = int(len(sequence_tensor))

    report = FusionReport(
        rows=int(row_count),
        feature_dim=int(fused.shape[1]),
        target_positive_rate=float(targets.mean()) if len(targets) else 0.0,
        target_hold_rate=float(hold_mask.mean()) if len(hold_mask) else 0.0,
        target_horizon=int(args.lookahead),
        source_price_path=str(price_path),
        source_news_path=str(news_path),
        source_crowd_path=str(crowd_path),
        sequence_rows=sequence_rows,
        sequence_len=SEQUENCE_LEN if args.materialize_sequences else 0,
        source_persona_path=str(PERSONA_OUTPUTS_PATH) if PERSONA_OUTPUTS_PATH.exists() else '',
        target_summary=target_artifacts.summary,
    )
    save_fusion_report(FUSION_REPORT_PATH, report)
    print(report)
    print(
        {
            'sample_weight_path': str(SAMPLE_WEIGHTS_PATH),
            'target_hold_mask_path': str(TARGET_HOLD_MASK_PATH),
            'targets_multihorizon_path': str(TARGETS_MULTIHORIZON_PATH),
            'sample_weight_mean': float(sample_weights.mean()) if len(sample_weights) else 0.0,
            'hold_rate': float(hold_mask.mean()) if len(hold_mask) else 0.0,
            'horizons': list(horizons),
        }
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
