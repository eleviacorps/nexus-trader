from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import V12_FEATURE_CONSISTENCY_REPORT_PATH
from src.v12 import (
    render_feature_consistency_summary,
    run_feature_consistency_audit,
    verify_v12_artifacts,
    write_feature_consistency_report,
)


def _status_summary(status: dict) -> str:
    if status.get("all_present", False):
        return (
            "Phase 0 status: all critical V10/V11/TFT artifacts are present. "
            "V12 can build the consistency audit and BCFE on top of the existing stack."
        )
    missing = ", ".join(status.get("missing", []))
    return f"Phase 0 status: some critical artifacts are missing: {missing}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the V12 feature consistency audit over a raw bar window.")
    parser.add_argument("--start", default="2023-10-03 00:00:00+00:00")
    parser.add_argument("--end", default="2024-01-01 00:00:00+00:00")
    parser.add_argument("--warmup-bars", type=int, default=200)
    parser.add_argument("--pass-threshold", type=float, default=0.95)
    parser.add_argument("--report", default=str(V12_FEATURE_CONSISTENCY_REPORT_PATH))
    args = parser.parse_args()

    status = verify_v12_artifacts()
    print("V12 Phase 0 Artifact Check", flush=True)
    for artifact in status["artifacts"]:
        print(f"{artifact['path']} | exists={artifact['exists']} | size_bytes={artifact['size_bytes']}", flush=True)
    print(_status_summary(status), flush=True)

    print("Loading raw bars and running V12 feature consistency audit...", flush=True)
    report = run_feature_consistency_audit(
        start=args.start,
        end=args.end,
        warmup_bars=int(args.warmup_bars),
        pass_threshold=float(args.pass_threshold),
    )
    report["phase_0"] = status
    out_path = write_feature_consistency_report(report, Path(args.report))

    print("", flush=True)
    print("V12 Phase 1 Feature Consistency Summary", flush=True)
    print(render_feature_consistency_summary(report), flush=True)
    legacy = report["legacy_archive_vs_live"]
    print("", flush=True)
    print(
        json.dumps(
            {
                "report_path": str(out_path),
                "pass_count": len(legacy["pass_features"]),
                "fail_count": len(legacy["fail_features"]),
                "bcfe_self_check_fail_count": len(report["bcfe_self_check"]["fail_features"]),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
