from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path("/home/rocm-user/jupyter/nexus")
LOG_DIR = ROOT / "outputs" / "logs"
PYTHON = "python"

os.environ.setdefault("NEXUS_NUM_WORKERS_SERVER", "12")
os.environ.setdefault("NEXUS_PREFETCH_FACTOR", "6")
os.environ.setdefault("NEXUS_PERSISTENT_WORKERS", "1")
os.environ.setdefault("NEXUS_PIN_MEMORY", "1")
os.environ.setdefault("NEXUS_AMP_ENABLED", "1")
os.environ.setdefault("NEXUS_AMP_DTYPE", "bfloat16")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")


def run_step(name: str, args: list[str]) -> None:
    print(f"\n===== {name} =====", flush=True)
    completed = subprocess.run(args, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"{name} failed with exit code {completed.returncode}")


def audit_args(run_tag: str) -> list[str]:
    return [
        PYTHON,
        "scripts/model_artifact_leakage_analysis.py",
        "--training-summary",
        f"outputs/evaluation/training_summary_{run_tag}.json",
        "--walkforward-report",
        f"outputs/evaluation/walkforward_report_{run_tag}.json",
        "--manifest",
        f"models/tft/model_manifest_{run_tag}.json",
        "--precision-gate",
        f"models/tft/precision_gate_{run_tag}.json",
        "--meta-gate",
        f"models/tft/meta_gate_{run_tag}.pkl",
        "--output",
        f"outputs/evaluation/model_artifact_leakage_report_{run_tag}.json",
    ]


def main() -> int:
    os.chdir(ROOT)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    common_eval = [
        "--batch-size",
        "1536",
        "--max-calibration-windows",
        "60000",
    ]
    common_train = [
        "--epochs",
        "10",
        "--batch-size",
        "1536",
        "--num-workers",
        "12",
        "--amp",
        "--amp-dtype",
        "bfloat16",
        "--split-mode",
        "year",
        "--metric",
        "f1",
        "--selection-metric",
        "roc_auc",
    ]

    steps = [
        (
            "walkforward_mh12_full_v7",
            [
                PYTHON,
                "scripts/run_walkforward_evaluation.py",
                "--run-tag",
                "mh12_full_v7",
                "--years",
                "2024,2025,2026",
                *common_eval,
            ],
        ),
        ("audit_mh12_full_v7", audit_args("mh12_full_v7")),
        (
            "train_mh12_recent_v7",
            [
                PYTHON,
                "scripts/train_fused_tft.py",
                *common_train,
                "--train-years",
                "2021,2022,2023,2024",
                "--val-years",
                "2025",
                "--test-years",
                "2026",
                "--run-tag",
                "mh12_recent_v7",
            ],
        ),
        (
            "walkforward_mh12_recent_v7",
            [
                PYTHON,
                "scripts/run_walkforward_evaluation.py",
                "--run-tag",
                "mh12_recent_v7",
                "--years",
                "2026",
                *common_eval,
            ],
        ),
        ("audit_mh12_recent_v7", audit_args("mh12_recent_v7")),
    ]

    for name, args in steps:
        run_step(name, args)

    print("\n===== v7 resume pipeline complete =====", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
