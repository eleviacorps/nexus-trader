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
    FUSED_TIMESTAMPS_PATH,
    LEGACY_CROWD_EMBEDDINGS_NPY_PATH,
    LEGACY_NEWS_EMBEDDINGS_NPY_PATH,
    LEGACY_NEWS_EMBEDDINGS_RAW_PATH,
    LEGACY_PRICE_FEATURES_CSV,
    LEGACY_PRICE_FEATURES_PARQUET,
    NEWS_EMBEDDINGS_NPY_PATH,
    NEWS_EMBEDDINGS_RAW_PATH,
    PRICE_FEATURES_CSV_FALLBACK,
    PRICE_FEATURES_PATH,
    TARGETS_PATH,
)
from src.pipeline.fusion import (  # noqa: E402
    FusionReport,
    build_fused_feature_matrix,
    extract_price_block,
    extract_target_vector,
    load_price_frame,
    normalize_binary_targets,
    save_fusion_report,
    save_numpy_artifact,
)


def resolve_first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    joined = ", ".join(str(path) for path in paths)
    raise FileNotFoundError(f"No artifact found in: {joined}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build fused feature artifacts from price, news, and crowd blocks.")
    parser.add_argument("--limit-rows", type=int, default=0, help="Optional cap for smoke runs.")
    args = parser.parse_args()

    price_path = resolve_first_existing([PRICE_FEATURES_PATH, PRICE_FEATURES_CSV_FALLBACK, LEGACY_PRICE_FEATURES_PARQUET, LEGACY_PRICE_FEATURES_CSV])
    news_path = resolve_first_existing([NEWS_EMBEDDINGS_RAW_PATH, NEWS_EMBEDDINGS_NPY_PATH, LEGACY_NEWS_EMBEDDINGS_RAW_PATH, LEGACY_NEWS_EMBEDDINGS_NPY_PATH])
    crowd_path = resolve_first_existing([CROWD_EMBEDDINGS_NPY_PATH, LEGACY_CROWD_EMBEDDINGS_NPY_PATH])

    price_frame = load_price_frame(price_path)
    price_block = extract_price_block(price_frame)
    targets = normalize_binary_targets(extract_target_vector(price_frame))
    news_block = np.load(news_path, mmap_mode="r")
    crowd_block = np.load(crowd_path, mmap_mode="r")

    row_count = min(len(price_block), len(targets), len(news_block), len(crowd_block))
    if args.limit_rows > 0:
        row_count = min(row_count, args.limit_rows)

    price_block = np.asarray(price_block[:row_count], dtype=np.float32)
    targets = np.asarray(targets[:row_count], dtype=np.float32)
    news_block = np.asarray(news_block[:row_count], dtype=np.float32)
    crowd_block = np.asarray(crowd_block[:row_count], dtype=np.float32)

    fused = build_fused_feature_matrix(price_block, news_block, crowd_block)
    save_numpy_artifact(FUSED_FEATURE_MATRIX_PATH, fused)
    save_numpy_artifact(TARGETS_PATH, targets)

    timestamps = np.asarray(price_frame.index[:row_count].astype(str), dtype="<U32")
    save_numpy_artifact(FUSED_TIMESTAMPS_PATH, timestamps)

    report = FusionReport(
        rows=int(row_count),
        feature_dim=int(fused.shape[1]),
        target_positive_rate=float(targets.mean()) if len(targets) else 0.0,
        source_price_path=str(price_path),
        source_news_path=str(news_path),
        source_crowd_path=str(crowd_path),
    )
    save_fusion_report(FUSION_REPORT_PATH, report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
