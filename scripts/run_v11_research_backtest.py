from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # type: ignore

from config.project_config import V10_BRANCH_FEATURES_PATH, V11_RESEARCH_BACKTEST_MD_PATH, V11_RESEARCH_BACKTEST_PATH  # noqa: E402
from src.v11 import render_v11_markdown, run_v11_backtest  # noqa: E402


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local V11 research backtest with SETL, PCOP, CESM, and PMWM.")
    parser.add_argument("--features", default=str(V10_BRANCH_FEATURES_PATH))
    parser.add_argument("--report", default=str(V11_RESEARCH_BACKTEST_PATH))
    parser.add_argument("--markdown", default=str(V11_RESEARCH_BACKTEST_MD_PATH))
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()

    feature_path = Path(args.features)
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature artifact not found: {feature_path}")
    frame = _read_frame(feature_path)
    summary = run_v11_backtest(frame, validation_fraction=args.validation_fraction)
    summary["feature_path"] = str(feature_path)

    report_path = Path(args.report)
    markdown_path = Path(args.markdown)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown_path.write_text(render_v11_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
