from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_V21_DIR = ROOT / "outputs" / "v21"
OUTPUTS_V21_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = OUTPUTS_V21_DIR / "phase1_features_summary.json"

ARCHIVE_PRIMARY = ROOT / "data" / "raw" / "xauusd_1min_2007_2024.parquet"
ARCHIVE_FALLBACK = ROOT / "data_store" / "processed" / "XAUUSD_1m_full.parquet"
LEGACY_RAW_TARGET = ROOT / "data_store" / "processed" / "XAUUSD_1m_full.parquet"


def run_step(command: list[str]) -> dict[str, object]:
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def ensure_archive() -> dict[str, object]:
    archive = ARCHIVE_PRIMARY if ARCHIVE_PRIMARY.exists() else ARCHIVE_FALLBACK
    if not archive.exists():
        return {
            "step": "ensure_archive",
            "returncode": 1,
            "error": f"Missing remote archive at {ARCHIVE_PRIMARY} and fallback {ARCHIVE_FALLBACK}",
        }
    LEGACY_RAW_TARGET.parent.mkdir(parents=True, exist_ok=True)
    if archive.resolve() != LEGACY_RAW_TARGET.resolve():
        shutil.copy2(archive, LEGACY_RAW_TARGET)
    return {
        "step": "ensure_archive",
        "returncode": 0,
        "archive": str(archive),
        "legacy_target": str(LEGACY_RAW_TARGET),
        "size_bytes": archive.stat().st_size,
    }


def collect_outputs() -> dict[str, object]:
    import pandas as pd

    features_path = ROOT / "data" / "features" / "v20_features.parquet"
    macro_path = ROOT / "data" / "features" / "macro_features.parquet"
    denoised_path = ROOT / "data" / "features" / "v20_ohlcv_denoised.parquet"
    regime_path = ROOT / "data" / "features" / "regime_labels.parquet"
    copied_features = ROOT / "data" / "features" / "v21_features.parquet"
    copied_meta = ROOT / "data" / "features" / "v21_features_metadata.json"
    copied_macro = ROOT / "data" / "features" / "v21_macro_features.parquet"
    copied_denoised = ROOT / "data" / "features" / "v21_ohlcv_denoised.parquet"
    copied_regime = ROOT / "data" / "features" / "regime_labels_full.parquet"
    copied_hmm = ROOT / "checkpoints" / "v21" / "hmm_6state.pkl"
    hmm_source = ROOT / "checkpoints" / "v20" / "hmm_6state.pkl"

    if features_path.exists():
        shutil.copy2(features_path, copied_features)
    if (ROOT / "data" / "features" / "v20_features_metadata.json").exists():
        copied_meta.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "data" / "features" / "v20_features_metadata.json", copied_meta)
    if macro_path.exists():
        shutil.copy2(macro_path, copied_macro)
    if denoised_path.exists():
        shutil.copy2(denoised_path, copied_denoised)
    if regime_path.exists():
        shutil.copy2(regime_path, copied_regime)
    if hmm_source.exists():
        copied_hmm.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(hmm_source, copied_hmm)

    frame = pd.read_parquet(copied_features) if copied_features.exists() else pd.DataFrame()
    return {
        "v21_features_exists": copied_features.exists(),
        "v21_features_rows": int(len(frame)),
        "v21_feature_count": int(len(frame.columns)) if not frame.empty else 0,
        "macro_exists": copied_macro.exists(),
        "denoised_exists": copied_denoised.exists(),
        "regime_exists": copied_regime.exists(),
        "hmm_exists": copied_hmm.exists(),
    }


def main() -> int:
    python = sys.executable
    steps: list[dict[str, object]] = []

    steps.append(ensure_archive())
    if steps[-1]["returncode"] != 0:
        SUMMARY_PATH.write_text(json.dumps({"steps": steps}, indent=2), encoding="utf-8")
        print({"summary_path": str(SUMMARY_PATH), "failed": "ensure_archive"})
        return 1

    install_commands = [
        [python, "-m", "pip", "install", "mamba-ssm", "--quiet"],
        [python, "-m", "pip", "install", "einops", "lightning", "torchmetrics", "--quiet"],
        [python, "-m", "pip", "install", "hmmlearn", "statsmodels", "arch", "pywavelets", "--quiet"],
        [python, "-m", "pip", "install", "faiss-cpu", "--quiet"],
        [python, "-m", "pip", "install", "xgboost", "--quiet"],
        [python, "-m", "pip", "install", "optuna", "--quiet"],
    ]
    for command in install_commands:
        step = run_step(command)
        if step["returncode"] != 0 and command[4] == "mamba-ssm":
            step["non_blocking_note"] = "mamba-ssm unavailable, using manual SSM fallback"
            step["returncode"] = 0
        steps.append(step)

    steps.append(run_step([python, "scripts/build_macro_features.py", "--interval", "15min"]))
    steps.append(run_step([python, "scripts/train_regime_hmm.py", "--interval", "15min"]))
    steps.append(run_step([python, "scripts/build_v20_features.py", "--interval", "15min"]))

    outputs = collect_outputs()
    payload = {"steps": steps, "outputs": outputs}
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print({"summary_path": str(SUMMARY_PATH), **outputs})
    return 0 if all(int(step.get("returncode", 1)) == 0 for step in steps) else 1


if __name__ == "__main__":
    raise SystemExit(main())
