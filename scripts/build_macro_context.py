from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.project_config import MACRO_FEATURES_PATH, MACRO_REPORT_PATH, RAW_MACRO_DIR  # noqa: E402
from src.pipeline.perception import build_macro_artifacts, save_frame, save_json  # noqa: E402


def main() -> int:
    frame = build_macro_artifacts(RAW_MACRO_DIR)
    save_frame(MACRO_FEATURES_PATH, frame)
    report = {
        "rows": int(len(frame)),
        "columns": [column for column in frame.columns if column != "date"],
        "date_min": str(frame["date"].min()) if len(frame) else None,
        "date_max": str(frame["date"].max()) if len(frame) else None,
    }
    save_json(MACRO_REPORT_PATH, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
