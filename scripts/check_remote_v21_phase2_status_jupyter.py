from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/home/rocm-user/jupyter/nexus")
LOG_PATH = ROOT / "outputs" / "logs" / "remote_v21_phase2.log"
PID_PATH = ROOT / "outputs" / "logs" / "remote_v21_phase2.pid"
SUMMARY_PATH = ROOT / "outputs" / "v21" / "phase2_training_summary.json"

payload = {
    "log_exists": LOG_PATH.exists(),
    "pid_exists": PID_PATH.exists(),
    "summary_exists": SUMMARY_PATH.exists(),
    "pid": PID_PATH.read_text(encoding="utf-8").strip() if PID_PATH.exists() else None,
    "tail": LOG_PATH.read_text(encoding="utf-8", errors="replace")[-4000:].splitlines()[-20:] if LOG_PATH.exists() else [],
}
if SUMMARY_PATH.exists():
    payload["summary_tail"] = SUMMARY_PATH.read_text(encoding="utf-8", errors="replace")[-4000:]
print(json.dumps(payload, indent=2))
