from pathlib import Path
import psutil


root = Path("/home/rocm-user/jupyter/nexus")
log_path = root / "outputs" / "logs" / "remote_v21_phase1.log"
pid_path = root / "outputs" / "logs" / "remote_v21_phase1.pid"
summary_path = root / "outputs" / "v21" / "phase1_features_summary.json"
info = {
    "log_exists": log_path.exists(),
    "pid_exists": pid_path.exists(),
    "summary_exists": summary_path.exists(),
}
if pid_path.exists():
    pid = int(pid_path.read_text(encoding="utf-8").strip())
    info["pid"] = pid
    info["running"] = psutil.pid_exists(pid)
if log_path.exists():
    info["tail"] = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-30:]
if summary_path.exists():
    info["summary_tail"] = summary_path.read_text(encoding="utf-8", errors="ignore")[-2000:]
print(info)
