from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run_step(name: str, command: list[str]) -> None:
    print(f"===== {name} =====", flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"{name} failed with exit code {completed.returncode}")


def main() -> int:
    steps = [
        (
            "regenerate_gate_mh12_full_v8",
            [PYTHON, "scripts/regenerate_gate_artifacts.py", "--run-tag", "mh12_full_v8"],
        ),
        (
            "walkforward_mh12_full_v8",
            [PYTHON, "scripts/run_walkforward_evaluation.py", "--run-tag", "mh12_full_v8", "--years", "2024,2025,2026"],
        ),
        (
            "regenerate_gate_mh12_recent_v8",
            [PYTHON, "scripts/regenerate_gate_artifacts.py", "--run-tag", "mh12_recent_v8"],
        ),
        (
            "walkforward_mh12_recent_v8",
            [PYTHON, "scripts/run_walkforward_evaluation.py", "--run-tag", "mh12_recent_v8", "--years", "2026"],
        ),
        (
            "summarize_v8_results",
            [PYTHON, "scripts/summarize_v8_results.py", "--run-tags", "mh12_full_v8", "mh12_recent_v8"],
        ),
    ]
    for name, command in steps:
        run_step(name, command)
    print("===== v8 gate refresh complete =====", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
