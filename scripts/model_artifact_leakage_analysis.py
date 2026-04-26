from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.project_config import (  # noqa: E402
    GATE_CONTEXT_PATH,
    MODEL_MANIFEST_PATH,
    META_GATE_PATH,
    OUTPUTS_EVAL_DIR,
    PRECISION_GATE_PATH,
    TARGETS_MULTIHORIZON_PATH,
    TRAINING_SUMMARY_PATH,
    WALKFORWARD_REPORT_PATH,
    FUSED_TIMESTAMPS_PATH,
)
from src.backtest.artifact_audit import audit_model_artifacts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Nexus selector/gate/model artifacts for leakage-like inconsistencies.")
    parser.add_argument("--training-summary", type=Path, default=TRAINING_SUMMARY_PATH)
    parser.add_argument("--walkforward-report", type=Path, default=WALKFORWARD_REPORT_PATH)
    parser.add_argument("--manifest", type=Path, default=MODEL_MANIFEST_PATH)
    parser.add_argument("--precision-gate", type=Path, default=PRECISION_GATE_PATH)
    parser.add_argument("--meta-gate", type=Path, default=META_GATE_PATH)
    parser.add_argument("--gate-context", type=Path, default=GATE_CONTEXT_PATH)
    parser.add_argument("--timestamps", type=Path, default=FUSED_TIMESTAMPS_PATH)
    parser.add_argument("--targets", type=Path, default=TARGETS_MULTIHORIZON_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUTS_EVAL_DIR / "model_artifact_leakage_report.json")
    args = parser.parse_args()

    report = audit_model_artifacts(
        training_summary_path=args.training_summary,
        walkforward_report_path=args.walkforward_report,
        manifest_path=args.manifest,
        precision_gate_path=args.precision_gate,
        meta_gate_path=args.meta_gate,
        gate_context_path=args.gate_context,
        timestamps_path=args.timestamps,
        targets_multihorizon_path=args.targets,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
