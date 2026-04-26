from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v22.sjd_augmentation import build_augmented_sjd_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the V22-augmented SJD dataset from month-debug and live-autopsy findings.")
    parser.add_argument("--months", default="2023-12,2024-12")
    parser.add_argument("--output", default="")
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    months = [item.strip() for item in str(args.months).split(",") if item.strip()]
    report = build_augmented_sjd_dataset(
        months,
        output_path=Path(args.output) if args.output else None,
        report_path=Path(args.report) if args.report else None,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
