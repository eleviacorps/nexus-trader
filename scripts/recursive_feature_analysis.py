from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.project_config import FUSED_TENSOR_PATH, OUTPUTS_EVAL_DIR
from src.backtest.validation import analyze_recursive_window_consistency


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze recursive consistency of overlapping fused sequence windows.")
    parser.add_argument("--tensor", type=Path, default=FUSED_TENSOR_PATH)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, default=OUTPUTS_EVAL_DIR / "recursive_feature_analysis_report.json")
    args = parser.parse_args()

    if args.tensor.exists():
        fused_tensor = np.load(args.tensor, mmap_mode="r")
        report = {
            "tensor_path": str(args.tensor),
            "tensor_shape": list(fused_tensor.shape),
            "consistency_report": analyze_recursive_window_consistency(fused_tensor, tolerance=args.tolerance),
            "skipped": False,
        }
    else:
        report = {
            "tensor_path": str(args.tensor),
            "tensor_shape": None,
            "consistency_report": None,
            "skipped": True,
            "reason": "fused_tensor.npy is not present locally; rebuild fused tensor artifacts before running recursive window analysis.",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
