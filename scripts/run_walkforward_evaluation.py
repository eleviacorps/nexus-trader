from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.project_config import BACKTEST_REPORT_PATH, MODEL_MANIFEST_PATH, TEST_YEARS, WALKFORWARD_REPORT_PATH  # noqa: E402
from src.evaluation.walkforward import run_walkforward_evaluation  # noqa: E402


def parse_years(text: str | None) -> list[int]:
    if text is None or not text.strip():
        return [int(year) for year in TEST_YEARS]
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def tagged_path(path: Path, run_tag: str) -> Path:
    if not run_tag:
        return path
    return path.with_name(f"{path.stem}_{run_tag}{path.suffix}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run walk-forward evaluation and directional backtesting.")
    parser.add_argument("--years", default=None, help="Comma-separated evaluation years. Defaults to TEST_YEARS.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-windows-per-year", type=int, default=0, help="Optional cap for smoke runs.")
    parser.add_argument("--max-calibration-windows", type=int, default=12000, help="Optional cap for validation-year calibration windows.")
    parser.add_argument("--run-tag", default="", help="Optional tag used to resolve alternate manifest/report paths.")
    args = parser.parse_args()
    walkforward_path = tagged_path(WALKFORWARD_REPORT_PATH, args.run_tag)
    backtest_path = tagged_path(BACKTEST_REPORT_PATH, args.run_tag)

    report = run_walkforward_evaluation(
        years=parse_years(args.years),
        batch_size=args.batch_size,
        max_windows_per_year=args.max_windows_per_year,
        max_calibration_windows=args.max_calibration_windows,
        manifest_path=tagged_path(MODEL_MANIFEST_PATH, args.run_tag),
    )
    walkforward_path.parent.mkdir(parents=True, exist_ok=True)
    walkforward_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    backtest_payload = {
        "backtest": report.get("overall", {}).get("backtest", {}),
        "event_driven_backtest": report.get("overall", {}).get("event_driven_backtest", {}),
        "event_driven_by_horizon": report.get("overall", {}).get("event_driven_by_horizon", {}),
    }
    backtest_path.write_text(json.dumps(backtest_payload, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
