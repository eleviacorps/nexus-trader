from pathlib import Path
import psutil


root = Path("/home/rocm-user/jupyter/nexus")
log_path = root / "outputs" / "logs" / "remote_v20_pipeline.log"
pid_path = root / "outputs" / "logs" / "remote_v20_pipeline.pid"
info = {"log_exists": log_path.exists(), "pid_exists": pid_path.exists()}
if pid_path.exists():
    pid = int(pid_path.read_text(encoding="utf-8").strip())
    info["pid"] = pid
    info["running"] = psutil.pid_exists(pid)
if log_path.exists():
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-30:]
    info["tail"] = lines
print(info)
