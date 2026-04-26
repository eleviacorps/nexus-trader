from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v19.distillation_dataset import TEACHER_MODELS, build_sjd_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the V19 SJD distillation dataset from packet logs and NIM teachers.")
    parser.add_argument("--target-examples", type=int, default=5000)
    parser.add_argument("--max-teacher-queries", type=int, default=0)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--teacher-models", nargs="*", default=list(TEACHER_MODELS))
    args = parser.parse_args()

    report = build_sjd_dataset(
        target_examples=int(args.target_examples),
        max_teacher_queries=int(args.max_teacher_queries),
        symbol=str(args.symbol),
        teacher_models=args.teacher_models,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
