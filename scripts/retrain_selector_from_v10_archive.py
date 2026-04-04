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
    V10_BRANCH_FEATURES_PATH,
    V10_BRANCH_LABELS_PATH,
    V10_SELECTOR_CHECKPOINT_PATH,
    V10_SELECTOR_RESULTS_MD_PATH,
    V10_SELECTOR_RESULTS_PATH,
)
from src.v9 import (  # noqa: E402
    BRANCH_FEATURES_V9,
    build_branch_features,
    build_branch_labels,
    run_selector_experiments,
    save_selector_experiments,
    save_selector_torch,
    train_selector_torch,
)


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build selector-ready V10 features from a regenerated archive and train/evaluate the local selector.")
    parser.add_argument("--archive", required=True)
    parser.add_argument("--labels-output", default=str(V10_BRANCH_LABELS_PATH))
    parser.add_argument("--features-output", default=str(V10_BRANCH_FEATURES_PATH))
    parser.add_argument("--checkpoint", default=str(V10_SELECTOR_CHECKPOINT_PATH))
    parser.add_argument("--report", default=str(V10_SELECTOR_RESULTS_PATH))
    parser.add_argument("--markdown", default=str(V10_SELECTOR_RESULTS_MD_PATH))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    archive = _read_frame(Path(args.archive))
    labels = build_branch_labels(archive)
    features = build_branch_features(labels)
    label_path = Path(args.labels_output)
    feature_path = Path(args.features_output)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(label_path, index=False)
    features.to_parquet(feature_path, index=False)

    model, torch_report = train_selector_torch(
        features,
        feature_names=BRANCH_FEATURES_V9,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        device=args.device or None,
    )
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_selector_torch(model, checkpoint_path, BRANCH_FEATURES_V9)
    report_path = Path(args.report)
    markdown_path = Path(args.markdown)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    experiments = run_selector_experiments(features, validation_fraction=args.validation_fraction)
    save_selector_experiments(report_path, markdown_path, experiments)
    payload = {
        "archive_path": str(args.archive),
        "labels_path": str(label_path),
        "features_path": str(feature_path),
        "checkpoint_path": str(checkpoint_path),
        "torch_report": torch_report.__dict__,
        "selector_experiments_path": str(report_path),
    }
    report_json = json.loads(report_path.read_text(encoding="utf-8"))
    report_json["training"] = payload
    report_path.write_text(json.dumps(report_json, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
