from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # type: ignore

from config.project_config import V9_BRANCH_FEATURES_PATH, V9_BRANCH_LABELS_PATH  # noqa: E402
from src.v9 import (  # noqa: E402
    build_branch_features,
    build_branch_labels,
    summarize_branch_features,
    summarize_branch_labels,
)


def _resolve_output(path_text: str | None, fallback: Path) -> Path:
    if path_text:
        return Path(path_text)
    return fallback


def main() -> int:
    parser = argparse.ArgumentParser(description="Build V9 branch labels and features from an existing V8 branch archive.")
    parser.add_argument("--archive", required=True, help="Path to the input V8 branch archive parquet/csv file.")
    parser.add_argument("--labels-output", default="", help="Optional override for the V9 labels output path.")
    parser.add_argument("--features-output", default="", help="Optional override for the V9 features output path.")
    parser.add_argument("--report-output", default="", help="Optional override for the dataset report path.")
    args = parser.parse_args()

    archive_path = Path(args.archive)
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    if archive_path.suffix.lower() == ".csv":
        archive = pd.read_csv(archive_path)
    else:
        archive = pd.read_parquet(archive_path)

    labels = build_branch_labels(archive)
    features = build_branch_features(labels)

    labels_output = _resolve_output(args.labels_output, V9_BRANCH_LABELS_PATH)
    features_output = _resolve_output(args.features_output, V9_BRANCH_FEATURES_PATH)
    report_output = _resolve_output(args.report_output, features_output.with_suffix(".report.json"))
    labels_output.parent.mkdir(parents=True, exist_ok=True)
    features_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)

    labels.to_parquet(labels_output, index=False)
    features.to_parquet(features_output, index=False)

    label_summary = summarize_branch_labels(labels)
    feature_summary = summarize_branch_features(features)
    report = {
        "archive_path": str(archive_path),
        "labels_path": str(labels_output),
        "features_path": str(features_output),
        "label_summary": label_summary.__dict__,
        "feature_summary": feature_summary.__dict__,
    }
    report_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
