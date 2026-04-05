from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import V14_BACKTRADER_WALKFORWARD_REPORT_PATH, V15_PARTICIPATION_AUDIT_PATH
from src.v15.participation_audit import audit_walkforward_report, load_walkforward_report


def main() -> int:
    report = load_walkforward_report(V14_BACKTRADER_WALKFORWARD_REPORT_PATH)
    summary = audit_walkforward_report(report)
    V15_PARTICIPATION_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    V15_PARTICIPATION_AUDIT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(str(V15_PARTICIPATION_AUDIT_PATH), flush=True)
    print(json.dumps({key: value for key, value in summary.items() if key != "months"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
