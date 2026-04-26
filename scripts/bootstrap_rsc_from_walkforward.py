from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import (
    V13_BACKTRADER_WALKFORWARD_REPORT_PATH,
    V14_BACKTRADER_WALKFORWARD_REPORT_PATH,
    V15_RSC_BOOTSTRAPPED_PATH,
)
from src.v15.cbwf import build_bootstrapped_rsc


def main() -> int:
    build_bootstrapped_rsc(
        walkforward_paths=[
            V13_BACKTRADER_WALKFORWARD_REPORT_PATH,
            V14_BACKTRADER_WALKFORWARD_REPORT_PATH,
        ],
        save_path=V15_RSC_BOOTSTRAPPED_PATH,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
