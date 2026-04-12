from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import OUTPUTS_V22_DIR
from src.v22.month_debugger import run_v22_month_suite


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the V22 debug suite on held-out month slices.")
    parser.add_argument("--months", default="2023-12,2024-12")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    months = [item.strip() for item in str(args.months).split(",") if item.strip()]
    if not months:
        raise SystemExit("At least one month is required.")

    report = run_v22_month_suite(months)
    OUTPUTS_V22_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else (OUTPUTS_V22_DIR / f"month_debug_suite_{'_'.join(months).replace('-', '_')}.json")
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(str(out_path))
    for item in report.get("results", []):
        print(
            json.dumps(
                {
                    "month": item.get("month"),
                    "experiment": item.get("experiment"),
                    "trade_count": item.get("trade_count"),
                    "target_trade_band_met": item.get("target_trade_band_met"),
                    "win_rate": item.get("win_rate"),
                    "avg_return": item.get("avg_return"),
                    "cumulative_return": item.get("cumulative_return"),
                    "sharpe_like": item.get("sharpe_like"),
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
