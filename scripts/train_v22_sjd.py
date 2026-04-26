from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import OUTPUTS_V22_DIR, V19_SJD_FEATURE_NAMES_PATH
from src.v19.sjd_model import train_sjd_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a V22-specific SJD checkpoint from the augmented dataset.")
    parser.add_argument("--dataset", default=str(OUTPUTS_V22_DIR / "sjd_dataset_v22_augmented.parquet"))
    parser.add_argument("--checkpoint", default=str(OUTPUTS_V22_DIR / "sjd_v22_augmented.pt"))
    parser.add_argument("--npz", default=str(OUTPUTS_V22_DIR / "sjd_v22_augmented.npz"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    report = train_sjd_model(
        dataset_path=Path(args.dataset),
        feature_names_path=V19_SJD_FEATURE_NAMES_PATH,
        checkpoint_path=Path(args.checkpoint),
        npz_output_path=Path(args.npz),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
    )
    report_path = OUTPUTS_V22_DIR / "sjd_v22_training_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
