from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_V21_DIR = ROOT / "outputs" / "v21"
OUTPUTS_V21_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = OUTPUTS_V21_DIR / "phase2_training_summary.json"


def launch(command: list[str], log_path: Path) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )


def main() -> int:
    python = sys.executable
    xlstm_log = ROOT / "outputs" / "logs" / "remote_v21_xlstm.log"
    bimamba_log = ROOT / "outputs" / "logs" / "remote_v21_bimamba.log"

    processes = {
        "xlstm": launch(
            [python, "scripts/train_xlstm_backbone.py", "--epochs", "5", "--batch-size", "256", "--sequence-len", "240", "--max-rows", "200000"],
            xlstm_log,
        ),
        "bimamba": launch(
            [python, "scripts/train_bimamba_backbone_v21.py", "--epochs", "5", "--batch-size", "128", "--sequence-len", "240", "--max-rows", "200000"],
            bimamba_log,
        ),
    }

    status: dict[str, dict[str, object]] = {}
    while processes:
        for name, process in list(processes.items()):
            code = process.poll()
            if code is None:
                continue
            status[name] = {
                "returncode": int(code),
                "log_path": str(xlstm_log if name == "xlstm" else bimamba_log),
            }
            processes.pop(name)
        if processes:
            time.sleep(10)

    xlstm_report = ROOT / "outputs" / "v21" / "xlstm_training_report.json"
    bimamba_report = ROOT / "outputs" / "v21" / "bimamba_training_report.json"
    summary = {
        "status": status,
        "xlstm_report_exists": xlstm_report.exists(),
        "bimamba_report_exists": bimamba_report.exists(),
        "xlstm_log_tail": xlstm_log.read_text(encoding="utf-8", errors="replace")[-4000:] if xlstm_log.exists() else "",
        "bimamba_log_tail": bimamba_log.read_text(encoding="utf-8", errors="replace")[-4000:] if bimamba_log.exists() else "",
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(SUMMARY_PATH), **summary}))
    return 0 if all(int(item.get("returncode", 1)) == 0 for item in status.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
