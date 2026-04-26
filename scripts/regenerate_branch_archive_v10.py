from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # type: ignore

from config.project_config import (  # noqa: E402
    V10_BRANCH_ARCHIVE_PATH,
    V10_BRANCH_ARCHIVE_REPORT_PATH,
)
from src.v8.branch_selector_v8 import summarize_branch_archive  # noqa: E402
from src.v10 import audit_branch_archive, diversify_branch_archive  # noqa: E402


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate a V10 branch archive from an existing branch archive using diversity supervision.")
    parser.add_argument("--archive", required=True, help="Source branch archive parquet/csv.")
    parser.add_argument("--output", default=str(V10_BRANCH_ARCHIVE_PATH))
    parser.add_argument("--report", default=str(V10_BRANCH_ARCHIVE_REPORT_PATH))
    parser.add_argument("--limit-samples", type=int, default=0, help="Optional cap for local smoke runs.")
    args = parser.parse_args()

    archive_path = Path(args.archive)
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    frame = _read_frame(archive_path)
    if args.limit_samples > 0:
        keep_ids = frame["sample_id"].drop_duplicates().head(args.limit_samples).tolist()
        frame = frame.loc[frame["sample_id"].isin(keep_ids)].copy()

    baseline_audit = audit_branch_archive(frame)
    regenerated, reports = diversify_branch_archive(frame)
    regenerated_audit = audit_branch_archive(regenerated)
    archive_summary = summarize_branch_archive(regenerated)

    output_path = Path(args.output)
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        regenerated.to_parquet(output_path, index=False)
        saved_path = output_path
    except Exception:
        saved_path = output_path.with_suffix(".csv")
        regenerated.to_csv(saved_path, index=False)
    payload = {
        "source_archive": str(archive_path),
        "artifact_path": str(saved_path),
        "baseline_audit": baseline_audit.__dict__,
        "regenerated_audit": regenerated_audit.__dict__,
        "archive_summary": archive_summary.__dict__,
        "sample_reports_preview": reports[:25],
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
