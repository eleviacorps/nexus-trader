from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd  # type: ignore

from config.project_config import (  # noqa: E402
    FUSED_FEATURE_MATRIX_PATH,
    FUSED_TIMESTAMPS_PATH,
    V9_BRANCH_FEATURES_ENRICHED_PATH,
    V9_BRANCH_FEATURES_PATH,
    V9_MEMORY_BANK_ENCODER_PATH,
    V9_MEMORY_BANK_INDEX_PATH,
)
from src.v9 import load_memory_bank, query_memory_bank  # noqa: E402


def _as_naive_datetime64(values: np.ndarray) -> np.ndarray:
    parsed = pd.to_datetime(values, utc=True, errors="coerce").tz_convert(None)
    return parsed.to_numpy(dtype="datetime64[ns]")


def main() -> int:
    parser = argparse.ArgumentParser(description="Append memory-bank signals to a V9 branch feature parquet.")
    parser.add_argument("--features", default=str(V9_BRANCH_FEATURES_PATH))
    parser.add_argument("--output", default=str(V9_BRANCH_FEATURES_ENRICHED_PATH))
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    frame = pd.read_parquet(Path(args.features))
    encoder, bank = load_memory_bank(V9_MEMORY_BANK_ENCODER_PATH, V9_MEMORY_BANK_INDEX_PATH, device=args.device or None)
    if encoder is None or bank is None:
        raise FileNotFoundError("Memory bank artifacts are missing. Train them first.")

    fused = np.load(FUSED_FEATURE_MATRIX_PATH, mmap_mode="r")
    fused_timestamps = np.load(FUSED_TIMESTAMPS_PATH, mmap_mode="r")
    fused_times = _as_naive_datetime64(fused_timestamps)
    sample_rows = frame.groupby("sample_id", sort=False).head(1).copy()
    sample_times = _as_naive_datetime64(sample_rows["timestamp"].to_numpy())

    sample_payloads: dict[int, dict[str, float]] = {}
    for sample_id, sample_time in zip(sample_rows["sample_id"].tolist(), sample_times, strict=False):
        insert_at = int(np.searchsorted(fused_times, sample_time, side="right") - 1)
        if insert_at < args.window_size - 1:
            payload = {
                "memory_bank_confidence": 0.0,
                "memory_bank_bullish_probability": 0.5,
                "memory_bank_mean_distance": 0.0,
                "memory_bank_alignment": 0.5,
            }
        else:
            window = np.asarray(fused[insert_at - args.window_size + 1 : insert_at + 1], dtype=np.float32)
            result = query_memory_bank(encoder, bank, window, device=args.device or None)
            payload = {
                "memory_bank_confidence": float(result.analog_confidence),
                "memory_bank_bullish_probability": float(result.bullish_probability),
                "memory_bank_mean_distance": float(result.mean_distance),
                "memory_bank_alignment": 0.5,
            }
        sample_payloads[int(sample_id)] = payload

    for column in ("memory_bank_confidence", "memory_bank_bullish_probability", "memory_bank_mean_distance", "memory_bank_alignment"):
        frame[column] = frame["sample_id"].map({sample_id: payload[column] for sample_id, payload in sample_payloads.items()}).astype(np.float32)

    branch_direction = frame["branch_direction"].to_numpy(dtype=np.float32)
    bullish_probability = frame["memory_bank_bullish_probability"].to_numpy(dtype=np.float32)
    frame["memory_bank_alignment"] = np.where(branch_direction >= 0.0, bullish_probability, 1.0 - bullish_probability).astype(np.float32)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
