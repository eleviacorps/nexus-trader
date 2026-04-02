from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path("/home/rocm-user/jupyter/nexus")
LOG_DIR = ROOT / "outputs" / "logs"
PYTHON = "python"


def run_step(name: str, args: list[str]) -> None:
    print(f"\n===== {name} =====", flush=True)
    completed = subprocess.run(args, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"{name} failed with exit code {completed.returncode}")


def main() -> int:
    os.chdir(ROOT)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    steps = [
        ("build_quant_context", [PYTHON, "scripts/build_quant_context.py"]),
        ("build_persona_outputs", [PYTHON, "scripts/build_persona_outputs.py"]),
        ("build_fused_artifacts", [PYTHON, "scripts/build_fused_artifacts.py", "--lookahead", "15", "--horizons", "5,10,15,30"]),
        ("tests", [PYTHON, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-q"]),
        (
            "train_mh12_full_v5",
            [
                PYTHON,
                "scripts/train_fused_tft.py",
                "--epochs",
                "10",
                "--batch-size",
                "1024",
                "--split-mode",
                "year",
                "--metric",
                "f1",
                "--selection-metric",
                "roc_auc",
                "--train-years",
                "2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020",
                "--val-years",
                "2021,2022,2023",
                "--test-years",
                "2024,2025,2026",
                "--run-tag",
                "mh12_full_v5",
            ],
        ),
        (
            "walkforward_mh12_full_v5",
            [
                PYTHON,
                "scripts/run_walkforward_evaluation.py",
                "--run-tag",
                "mh12_full_v5",
                "--years",
                "2024,2025,2026",
                "--batch-size",
                "1024",
                "--max-calibration-windows",
                "60000",
            ],
        ),
        (
            "train_mh12_recent_v5",
            [
                PYTHON,
                "scripts/train_fused_tft.py",
                "--epochs",
                "10",
                "--batch-size",
                "1024",
                "--split-mode",
                "year",
                "--metric",
                "f1",
                "--selection-metric",
                "roc_auc",
                "--train-years",
                "2021,2022,2023,2024",
                "--val-years",
                "2025",
                "--test-years",
                "2026",
                "--run-tag",
                "mh12_recent_v5",
            ],
        ),
        (
            "walkforward_mh12_recent_v5",
            [
                PYTHON,
                "scripts/run_walkforward_evaluation.py",
                "--run-tag",
                "mh12_recent_v5",
                "--years",
                "2026",
                "--batch-size",
                "1024",
                "--max-calibration-windows",
                "60000",
            ],
        ),
    ]

    for name, args in steps:
        run_step(name, args)

    print("\n===== v5 pipeline complete =====", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
