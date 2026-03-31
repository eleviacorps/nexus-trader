from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.project_config import OUTPUTS_EVAL_DIR  # noqa: E402
from src.pipeline.persona import build_persona_artifacts  # noqa: E402


def main() -> int:
    report = build_persona_artifacts()
    OUTPUTS_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUTS_EVAL_DIR / "persona_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
