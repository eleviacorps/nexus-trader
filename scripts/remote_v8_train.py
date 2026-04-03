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


def main() -> int:
    os.chdir(ROOT)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    common_train = [
        "--epochs",
        "8",
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
    common_eval = [
        "--batch-size",
        "1536",
        "--max-calibration-windows",
        "60000",
    ]

    steps = [
        ("build_quant_context", [PYTHON, "scripts/build_quant_context.py"]),
        ("build_persona_outputs", [PYTHON, "scripts/build_persona_outputs.py"]),
        ("build_market_dynamics_labels", [PYTHON, "scripts/build_market_dynamics_labels.py"]),
        ("build_fused_artifacts", [PYTHON, "scripts/build_fused_artifacts.py", "--lookahead", "15", "--horizons", "5,10,15,30"]),
        ("build_v8_quant_stack", [PYTHON, "scripts/build_v8_quant_stack.py"]),
        ("tests", [PYTHON, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-q"]),
        (
            "train_mh12_full_v8",
            [
                PYTHON,
                "scripts/train_fused_tft.py",
                *common_train,
                "--train-years",
                "2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020",
                "--val-years",
                "2021,2022,2023",
                "--test-years",
                "2024,2025,2026",
                "--run-tag",
                "mh12_full_v8",
            ],
        ),
        (
            "walkforward_mh12_full_v8",
            [
                PYTHON,
                "scripts/run_walkforward_evaluation.py",
                "--run-tag",
                "mh12_full_v8",
                "--years",
                "2024,2025,2026",
                *common_eval,
            ],
        ),
        (
            "build_branch_archive_train_mh12_full_v8",
            [
                PYTHON,
                "scripts/build_v8_branch_archive.py",
                "--run-tag",
                "mh12_full_v8",
                "--years",
                "2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023",
                "--sample-stride",
                "240",
                "--max-samples",
                "8000",
            ],
        ),
        (
            "build_branch_archive_eval_mh12_full_v8",
            [
                PYTHON,
                "scripts/build_v8_branch_archive.py",
                "--run-tag",
                "mh12_full_v8",
                "--years",
                "2024,2025,2026",
                "--sample-stride",
                "120",
                "--max-samples",
                "4000",
                "--output",
                "outputs/v8/branch_archive_mh12_full_v8_eval.parquet",
                "--report",
                "outputs/evaluation/v8_branch_archive_report_mh12_full_v8_eval.json",
            ],
        ),
        ("train_branch_selector_mh12_full_v8", [PYTHON, "scripts/train_v8_branch_selector.py", "--run-tag", "mh12_full_v8"]),
        (
            "evaluate_branch_selector_mh12_full_v8",
            [
                PYTHON,
                "scripts/run_v8_evaluation.py",
                "--run-tag",
                "mh12_full_v8",
                "--archive",
                "outputs/v8/branch_archive_mh12_full_v8_eval.parquet",
            ],
        ),
        (
            "train_mh12_recent_v8",
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
                "mh12_recent_v8",
            ],
        ),
        (
            "walkforward_mh12_recent_v8",
            [
                PYTHON,
                "scripts/run_walkforward_evaluation.py",
                "--run-tag",
                "mh12_recent_v8",
                "--years",
                "2026",
                *common_eval,
            ],
        ),
        (
            "build_branch_archive_train_mh12_recent_v8",
            [
                PYTHON,
                "scripts/build_v8_branch_archive.py",
                "--run-tag",
                "mh12_recent_v8",
                "--years",
                "2021,2022,2023,2024,2025",
                "--sample-stride",
                "120",
                "--max-samples",
                "6000",
            ],
        ),
        (
            "build_branch_archive_eval_mh12_recent_v8",
            [
                PYTHON,
                "scripts/build_v8_branch_archive.py",
                "--run-tag",
                "mh12_recent_v8",
                "--years",
                "2026",
                "--sample-stride",
                "30",
                "--max-samples",
                "2500",
                "--output",
                "outputs/v8/branch_archive_mh12_recent_v8_eval.parquet",
                "--report",
                "outputs/evaluation/v8_branch_archive_report_mh12_recent_v8_eval.json",
            ],
        ),
        ("train_branch_selector_mh12_recent_v8", [PYTHON, "scripts/train_v8_branch_selector.py", "--run-tag", "mh12_recent_v8"]),
        (
            "evaluate_branch_selector_mh12_recent_v8",
            [
                PYTHON,
                "scripts/run_v8_evaluation.py",
                "--run-tag",
                "mh12_recent_v8",
                "--archive",
                "outputs/v8/branch_archive_mh12_recent_v8_eval.parquet",
            ],
        ),
        ("summarize_v8_results", [PYTHON, "scripts/summarize_v8_results.py", "--run-tags", "mh12_full_v8", "mh12_recent_v8"]),
    ]

    for name, args in steps:
        run_step(name, args)

    print("\n===== v8 pipeline complete =====", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
