from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config.project_config import V15_ECI_CALENDAR_DIR, V15_ECI_CALENDAR_PATH


REQUIRED_COLUMNS = ("datetime", "event_type", "importance")


def _load_inputs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        frames.append(frame.copy())
    if not frames:
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS) + ["actual", "forecast", "previous"])
    combined = pd.concat(frames, ignore_index=True)
    combined["datetime"] = pd.to_datetime(combined["datetime"], utc=True, errors="coerce")
    combined = combined.loc[~combined["datetime"].isna()].copy()
    combined["event_type"] = combined["event_type"].astype(str).str.lower().str.replace(" ", "_")
    combined["importance"] = pd.to_numeric(combined["importance"], errors="coerce").fillna(2).astype(int)
    for column in ("actual", "forecast", "previous"):
        if column not in combined.columns:
            combined[column] = pd.NA
    combined = combined.sort_values("datetime").drop_duplicates(subset=["datetime", "event_type"], keep="last")
    return combined


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a normalized historical economic calendar CSV for V15.")
    parser.add_argument("--input-csv", action="append", default=[])
    parser.add_argument("--output", default=str(V15_ECI_CALENDAR_PATH))
    args = parser.parse_args()

    V15_ECI_CALENDAR_DIR.mkdir(parents=True, exist_ok=True)
    input_paths = [Path(value) for value in args.input_csv]
    if not input_paths:
        sources_dir = V15_ECI_CALENDAR_DIR / "sources"
        input_paths = sorted(sources_dir.glob("*.csv")) if sources_dir.exists() else []

    combined = _load_inputs(input_paths)
    output_path = Path(args.output)
    if combined.empty:
        print(json.dumps({"output": str(output_path), "status": "no_input_sources_found", "rows": 0}, indent=2), flush=True)
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    summary = {
        "output": str(output_path),
        "rows": int(len(combined)),
        "event_type_counts": {str(key): int(value) for key, value in combined["event_type"].value_counts().to_dict().items()},
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
