from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import V21_MT5_TESTER_SIGNALS_PATH, V21_MT5_TESTER_SUMMARY_PATH
from src.v21.mt5_tester_bridge import export_v21_mt5_tester_bridge


def main() -> int:
    parser = argparse.ArgumentParser(description="Export V21 MT5 Strategy Tester bridge signals.")
    parser.add_argument("--month", default="2023-12", help="Month to export in YYYY-MM format.")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--mode", default="precision", choices=["frequency", "precision"])
    parser.add_argument("--equity", type=float, default=1000.0, help="Reference equity for V21 lot sizing.")
    parser.add_argument("--lookback-days", type=int, default=90)
    parser.add_argument("--lookback-bars", type=int, default=240)
    parser.add_argument("--pip-size", type=float, default=0.1)
    parser.add_argument("--csv-path", type=Path, default=V21_MT5_TESTER_SIGNALS_PATH)
    parser.add_argument("--summary-path", type=Path, default=V21_MT5_TESTER_SUMMARY_PATH)
    parser.add_argument("--copy-to-mt5-common", action="store_true", help="Also copy the generated CSV into MT5 Common/Files.")
    args = parser.parse_args()

    summary = export_v21_mt5_tester_bridge(
        month=str(args.month),
        symbol=str(args.symbol),
        mode=str(args.mode),
        lookback_days=int(args.lookback_days),
        lookback_bars=int(args.lookback_bars),
        equity=float(args.equity),
        pip_size=float(args.pip_size),
        csv_path=Path(args.csv_path),
        summary_path=Path(args.summary_path),
        copy_to_common=bool(args.copy_to_mt5_common),
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
