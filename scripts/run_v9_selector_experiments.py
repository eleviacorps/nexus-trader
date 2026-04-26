from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # type: ignore

from config.project_config import V9_BRANCH_FEATURES_PATH, V9_SELECTOR_RESULTS_MD_PATH, V9_SELECTOR_RESULTS_PATH  # noqa: E402
from src.v9 import run_selector_experiments, save_selector_experiments  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local V9 selector experiments.")
    parser.add_argument("--features", default=str(V9_BRANCH_FEATURES_PATH))
    parser.add_argument("--output-json", default=str(V9_SELECTOR_RESULTS_PATH))
    parser.add_argument("--output-md", default=str(V9_SELECTOR_RESULTS_MD_PATH))
    args = parser.parse_args()

    frame = pd.read_parquet(Path(args.features))
    results = run_selector_experiments(frame)
    save_selector_experiments(Path(args.output_json), Path(args.output_md), results)
    print(Path(args.output_json).read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
