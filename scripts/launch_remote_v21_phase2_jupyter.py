from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path("/home/rocm-user/jupyter/nexus")
LOG_PATH = ROOT / "outputs" / "logs" / "remote_v21_phase2.log"
PID_PATH = ROOT / "outputs" / "logs" / "remote_v21_phase2.pid"

ROOT.joinpath("outputs", "logs").mkdir(parents=True, exist_ok=True)
log_handle = LOG_PATH.open("ab")
process = subprocess.Popen(
    ["python", "scripts/remote_v21_phase2_train.py"],
    cwd=ROOT,
    stdout=log_handle,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
PID_PATH.write_text(str(process.pid), encoding="utf-8")
print(json.dumps({"pid": process.pid, "log_path": str(LOG_PATH), "pid_path": str(PID_PATH)}))
