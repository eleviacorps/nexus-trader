from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from config.project_config import (  # noqa: E402
    FUSED_FEATURE_MATRIX_PATH,
    TARGETS_PATH,
    V9_MEMORY_BANK_ENCODER_PATH,
    V9_MEMORY_BANK_INDEX_PATH,
    V9_MEMORY_BANK_REPORT_PATH,
)
from src.v9 import (  # noqa: E402
    build_memory_bank_index,
    build_memory_bank_windows,
    save_memory_bank,
    train_memory_bank_encoder,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the V9 memory bank encoder and build the local index.")
    parser.add_argument("--features", default=str(FUSED_FEATURE_MATRIX_PATH))
    parser.add_argument("--targets", default=str(TARGETS_PATH))
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--sample-stride", type=int, default=15)
    parser.add_argument("--max-samples", type=int, default=25000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    features = np.load(Path(args.features), mmap_mode="r")
    targets = np.load(Path(args.targets), mmap_mode="r").astype(np.int64)
    windows, labels = build_memory_bank_windows(
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.int64),
        window_size=args.window_size,
        sample_stride=args.sample_stride,
        max_samples=args.max_samples,
    )
    model, report = train_memory_bank_encoder(
        windows,
        labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device or None,
    )
    embeddings = build_memory_bank_index(model, windows, device=args.device or None)
    save_memory_bank(
        V9_MEMORY_BANK_ENCODER_PATH,
        V9_MEMORY_BANK_INDEX_PATH,
        V9_MEMORY_BANK_REPORT_PATH,
        model,
        embeddings,
        labels,
        report=report,
        window_size=args.window_size,
        sample_stride=args.sample_stride,
    )
    print(V9_MEMORY_BANK_REPORT_PATH.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
