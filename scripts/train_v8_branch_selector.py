from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None

from config.project_config import V8_BRANCH_ARCHIVE_PATH, V8_BRANCH_SELECTOR_PATH, V8_BRANCH_SELECTOR_REPORT_PATH  # noqa: E402
from src.v8.branch_selector_v8 import train_branch_selector_v8  # noqa: E402


def tagged_path(path: Path, run_tag: str) -> Path:
    if not run_tag:
        return path
    return path.with_name(f"{path.stem}_{run_tag}{path.suffix}")


def load_archive(path: Path):
    if pd is None:
        raise ImportError("pandas is required for training the v8 branch selector.")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the V8 branch selector on a historical branch archive.")
    parser.add_argument("--archive", default="", help="Optional override for the branch archive path.")
    parser.add_argument("--run-tag", default="")
    args = parser.parse_args()

    archive_path = Path(args.archive) if args.archive else tagged_path(V8_BRANCH_ARCHIVE_PATH, args.run_tag)
    if not archive_path.exists():
        csv_fallback = archive_path.with_suffix(".csv")
        if csv_fallback.exists():
            archive_path = csv_fallback
        else:
            raise FileNotFoundError(f"Branch archive not found: {archive_path}")
    selector_path = tagged_path(V8_BRANCH_SELECTOR_PATH, args.run_tag)
    report_path = tagged_path(V8_BRANCH_SELECTOR_REPORT_PATH, args.run_tag)

    frame = load_archive(archive_path)
    if len(frame) == 0:
        raise ValueError("Branch archive is empty; cannot train selector.")

    payload = train_branch_selector_v8(frame, selector_path)
    report = {
        "archive_path": str(archive_path),
        "selector_path": str(selector_path),
        "provider": str(payload.get("provider", "none")),
        "available": bool(payload.get("available", False)),
        "feature_names": list(payload.get("feature_names", [])),
        "rows": int(len(frame)),
        "samples": int(frame["sample_id"].nunique()) if "sample_id" in frame.columns else 0,
        "winner_positive_rate": float(frame["winner_label"].mean()) if "winner_label" in frame.columns else 0.0,
        "avg_path_error": float(frame["path_error"].mean()) if "path_error" in frame.columns else 0.0,
        "run_tag": args.run_tag,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
