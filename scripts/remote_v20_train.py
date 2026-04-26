from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_V20_DIR = ROOT / "outputs" / "v20"
OUTPUTS_V20_DIR.mkdir(parents=True, exist_ok=True)


def run_step(command: list[str], summary: list[dict[str, object]]) -> None:
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    summary.append(
        {
            "command": command,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
    )


def main() -> int:
    python = sys.executable
    summary: list[dict[str, object]] = []

    steps = [
        [python, "-m", "pip", "install", "pytest", "einops", "lightning", "torchmetrics", "hmmlearn", "statsmodels", "arch", "pywavelets", "optuna", "--quiet"],
        [python, "scripts/build_macro_features.py", "--interval", "15min", "--start", "2019-01-01", "--end", "2024-01-15"],
        [python, "scripts/train_regime_hmm.py", "--interval", "15min", "--start", "2019-01-01", "--end", "2024-01-15"],
        [python, "scripts/build_v20_features.py", "--interval", "15min", "--start", "2019-01-01", "--end", "2024-01-15"],
        [python, "scripts/build_branch_pairs_1m.py", "--max-pairs", "100000"],
        [python, "scripts/train_mamba_backbone.py", "--epochs", "2", "--batch-size", "128", "--sequence-len", "32"],
        [python, "scripts/train_branch_decoder.py", "--epochs", "2", "--batch-size", "256"],
        [python, "scripts/train_cabr_v20.py", "--epochs", "2", "--batch-size", "512"],
        [python, "scripts/generate_sjd_dataset_v20.py", "--target-rows", "50000"],
        [python, "scripts/train_sjd_v20.py", "--epochs", "3", "--batch-size", "512"],
        [python, "scripts/train_rl_executor.py"],
        [python, "scripts/run_v20_backtrader_month.py", "--month", "2023-12", "--mode", "frequency"],
        [python, "scripts/run_walkforward_v20.py", "--mode", "frequency"],
        [python, "scripts/build_v20_summary.py"],
        [python, "-m", "pytest", "tests/test_v20_conformal.py", "tests/test_v20_runtime.py", "tests/test_v20_sjd.py", "-v"],
    ]
    for command in steps:
        run_step(command, summary)

    summary_path = OUTPUTS_V20_DIR / "remote_v20_train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print({"summary_path": str(summary_path), "steps": len(summary)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
