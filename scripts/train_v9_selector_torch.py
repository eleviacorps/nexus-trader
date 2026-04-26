from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # type: ignore

from config.project_config import V9_BRANCH_FEATURES_PATH, V9_SELECTOR_CHECKPOINT_PATH, V9_SELECTOR_RESULTS_PATH  # noqa: E402
from src.v9 import BRANCH_FEATURES_V9, save_selector_torch, train_selector_torch  # noqa: E402


def _resolve_output(path_text: str | None, fallback: Path) -> Path:
    if path_text:
        return Path(path_text)
    return fallback


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the local V9 torch selector on branch features.")
    parser.add_argument("--features", default=str(V9_BRANCH_FEATURES_PATH))
    parser.add_argument("--checkpoint", default=str(V9_SELECTOR_CHECKPOINT_PATH))
    parser.add_argument("--report", default=str(V9_SELECTOR_RESULTS_PATH))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="", help="Optional torch device override such as cuda or cpu.")
    args = parser.parse_args()

    feature_path = Path(args.features)
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature artifact not found: {feature_path}")

    frame = pd.read_parquet(feature_path)
    model, report = train_selector_torch(
        frame,
        feature_names=BRANCH_FEATURES_V9,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        device=args.device or None,
    )

    checkpoint_path = _resolve_output(args.checkpoint, V9_SELECTOR_CHECKPOINT_PATH)
    report_path = _resolve_output(args.report, V9_SELECTOR_RESULTS_PATH)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    save_selector_torch(model, checkpoint_path, BRANCH_FEATURES_V9)

    payload = {
        "feature_path": str(feature_path),
        "checkpoint_path": str(checkpoint_path),
        "report": report.__dict__,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
