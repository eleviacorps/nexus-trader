from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.project_config import (
    FUSED_FEATURE_MATRIX_PATH,
    FUSED_TIMESTAMPS_PATH,
    OUTPUTS_EVAL_DIR,
    TARGETS_MULTIHORIZON_PATH,
)
from src.backtest.validation import analyze_feature_target_correlations, analyze_timestamp_monotonicity, infer_feature_names


def load_primary_targets(path: Path) -> tuple[np.ndarray, list[str]]:
    bundle = np.load(path)
    keys = [key for key in bundle.files if key.startswith("target_")]
    keys.sort()
    if not keys:
        raise ValueError(f"No target_* arrays found in {path}.")
    primary = np.asarray(bundle[keys[0]], dtype=np.float32)
    return primary, keys


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a lightweight lookahead / leakage audit over fused Nexus artifacts.")
    parser.add_argument("--features", type=Path, default=FUSED_FEATURE_MATRIX_PATH)
    parser.add_argument("--timestamps", type=Path, default=FUSED_TIMESTAMPS_PATH)
    parser.add_argument("--targets", type=Path, default=TARGETS_MULTIHORIZON_PATH)
    parser.add_argument("--suspicious-corr", type=float, default=0.4)
    parser.add_argument("--critical-corr", type=float, default=0.98)
    parser.add_argument("--output", type=Path, default=OUTPUTS_EVAL_DIR / "lookahead_analysis_report.json")
    args = parser.parse_args()

    features = np.load(args.features, mmap_mode="r")
    timestamps = np.load(args.timestamps)
    targets, target_keys = load_primary_targets(args.targets)
    feature_names = infer_feature_names(int(features.shape[-1]))

    report = {
        "features_path": str(args.features),
        "timestamps_path": str(args.timestamps),
        "targets_path": str(args.targets),
        "feature_shape": list(features.shape),
        "timestamp_report": analyze_timestamp_monotonicity(timestamps),
        "target_keys": target_keys,
        "primary_target_report": analyze_feature_target_correlations(
            features,
            targets,
            feature_names=feature_names,
            suspicious_threshold=args.suspicious_corr,
            critical_threshold=args.critical_corr,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
