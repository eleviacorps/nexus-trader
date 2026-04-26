from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.project_config import CROWD_REPORT_PATH, RAW_CROWD_DIR  # noqa: E402
from src.pipeline.perception import build_crowd_artifacts, save_json  # noqa: E402


def main() -> int:
    report = build_crowd_artifacts(RAW_CROWD_DIR)
    save_json(CROWD_REPORT_PATH, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
