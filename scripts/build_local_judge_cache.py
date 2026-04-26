from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import OUTPUTS_DIR
from src.v25.local_judge_cache import LocalJudgeCache


DECISION_LOG_PATH = OUTPUTS_DIR / "live" / "claude_decision_log.jsonl"
CACHE_PATH = OUTPUTS_DIR / "v25" / "local_judge_cache.jsonl"
REPORT_PATH = OUTPUTS_DIR / "v25" / "local_judge_cache_report.json"


def _iter_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def main() -> None:
    OUTPUTS_DIR.joinpath("v25").mkdir(parents=True, exist_ok=True)
    source_rows = _iter_rows(DECISION_LOG_PATH)
    cache = LocalJudgeCache(path=CACHE_PATH, similarity_threshold=0.92)
    cache.entries = []

    total_candidates = 0
    reusable_hits = 0
    available_rows = 0
    for row in source_rows:
        candidate = row.get("candidate", {})
        decision = row.get("decision", {})
        if not isinstance(candidate, Mapping) or not isinstance(decision, Mapping):
            continue
        if not bool(decision.get("available", False)):
            continue
        available_rows += 1
        total_candidates += 1
        prior = cache.lookup(candidate)
        if prior is not None:
            reusable_hits += 1
        cache.add(candidate, decision)
    cache.save()

    report = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "source_log_path": str(DECISION_LOG_PATH),
        "cache_path": str(CACHE_PATH),
        "similarity_threshold": 0.92,
        "source_rows": int(len(source_rows)),
        "available_rows": int(available_rows),
        "cache_entries": int(len(cache.entries)),
        "cache_hit_rate": float(reusable_hits / max(total_candidates, 1)),
        "cache_hits": int(reusable_hits),
        "cache_misses": int(max(total_candidates - reusable_hits, 0)),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
