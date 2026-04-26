from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # type: ignore

from config.project_config import V10_BRANCH_AUDIT_MD_PATH, V10_BRANCH_AUDIT_PATH  # noqa: E402
from src.v10 import audit_branch_archive, render_audit_markdown  # noqa: E402


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit branch diversity and cone containment for a branch archive.")
    parser.add_argument("--archive", required=True)
    parser.add_argument("--report", default=str(V10_BRANCH_AUDIT_PATH))
    parser.add_argument("--markdown", default=str(V10_BRANCH_AUDIT_MD_PATH))
    parser.add_argument("--title", default="V10 Branch Diversity Audit")
    args = parser.parse_args()

    archive_path = Path(args.archive)
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    frame = _read_frame(archive_path)
    summary = audit_branch_archive(frame)
    payload = {"archive_path": str(archive_path), "summary": summary.__dict__}

    report_path = Path(args.report)
    markdown_path = Path(args.markdown)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown_path.write_text(render_audit_markdown(args.title, summary), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
