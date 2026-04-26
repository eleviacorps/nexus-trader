from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
BACKUP_ROOT = OUTPUTS_DIR / "backups"


def _interval_from_env(name: str, default_seconds: int) -> int:
    raw = os.getenv(name, str(default_seconds)).strip()
    try:
        value = int(raw)
    except Exception:
        value = default_seconds
    return max(30, value)


def _python_executable() -> str:
    return os.getenv("NEXUS_PYTHON", sys.executable)


def _run_checked(command: list[str]) -> int:
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    return int(completed.returncode)


def run_api() -> int:
    host = os.getenv("NEXUS_HOST", "0.0.0.0").strip() or "0.0.0.0"
    port = os.getenv("NEXUS_PORT", "8000").strip() or "8000"
    command = [_python_executable(), "-m", "uvicorn", "src.service.app:create_app", "--factory", "--host", host, "--port", port]
    return _run_checked(command)


def run_frontend() -> int:
    frontend_dist = PROJECT_ROOT / "ui" / "frontend" / "dist"
    if not frontend_dist.exists():
        return 2
    command = [_python_executable(), "-m", "http.server", "4173", "--bind", "0.0.0.0", "--directory", str(frontend_dist)]
    return _run_checked(command)


def run_trader_once() -> int:
    command = [_python_executable(), "scripts/run_v25_proxy_paper.py"]
    return _run_checked(command)


def run_monitor_once() -> int:
    deployment_cmd = [_python_executable(), "scripts/deployment_readiness_check.py"]
    rc = _run_checked(deployment_cmd)
    validation_script = PROJECT_ROOT / "scripts" / "run_final_v25_validation.py"
    if validation_script.exists():
        rc = max(rc, _run_checked([_python_executable(), str(validation_script)]))
    return rc


def run_backup_once() -> int:
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    target = BACKUP_ROOT / timestamp
    target.mkdir(parents=True, exist_ok=True)
    include_roots = [
        OUTPUTS_DIR / "live",
        OUTPUTS_DIR / "v25",
        OUTPUTS_DIR / "deployment",
    ]
    copied: list[str] = []
    for root in include_roots:
        if not root.exists():
            continue
        for file in root.rglob("*"):
            if not file.is_file():
                continue
            relative = file.relative_to(OUTPUTS_DIR)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, destination)
            copied.append(str(relative))
    manifest = {
        "created_at": datetime.now(tz=UTC).isoformat(),
        "file_count": len(copied),
        "files": copied,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"backup_dir": str(target), "file_count": len(copied)}, indent=2))
    return 0


def _run_loop(worker: Callable[[], int], interval_seconds: int, label: str) -> int:
    while True:
        started = datetime.now(tz=UTC).isoformat()
        rc = worker()
        print(json.dumps({"service": label, "started_at": started, "exit_code": rc}, indent=2))
        time.sleep(interval_seconds)


def _spawn_supervised_service(name: str, loop: bool) -> subprocess.Popen[str]:
    args = [_python_executable(), "scripts/start_production.py", "--service", name]
    if loop:
        args.append("--loop")
    return subprocess.Popen(args, cwd=str(PROJECT_ROOT), text=True)


def run_all_supervisor(loop: bool) -> int:
    services = ["api", "frontend", "trader", "monitor", "backup"]
    children = {name: _spawn_supervised_service(name, loop=(name not in {"api", "frontend"} and loop)) for name in services}
    try:
        while True:
            for name, process in list(children.items()):
                rc = process.poll()
                if rc is None:
                    continue
                print(json.dumps({"service": name, "event": "restarting", "exit_code": rc}, indent=2))
                children[name] = _spawn_supervised_service(name, loop=(name != "api" and loop))
            time.sleep(5)
    except KeyboardInterrupt:
        for process in children.values():
            process.terminate()
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Start Nexus Trader production services.")
    parser.add_argument("--service", choices=["api", "frontend", "trader", "monitor", "backup", "all"], default="all")
    parser.add_argument("--loop", action="store_true", help="Run service(s) continuously.")
    args = parser.parse_args()

    trader_interval = _interval_from_env("NEXUS_TRADER_INTERVAL_SECONDS", 300)
    monitor_interval = _interval_from_env("NEXUS_MONITOR_INTERVAL_SECONDS", 600)
    backup_interval = _interval_from_env("NEXUS_BACKUP_INTERVAL_SECONDS", 900)

    if args.service == "api":
        raise SystemExit(run_api())
    if args.service == "frontend":
        raise SystemExit(run_frontend())
    if args.service == "trader":
        if args.loop:
            raise SystemExit(_run_loop(run_trader_once, trader_interval, "trader"))
        raise SystemExit(run_trader_once())
    if args.service == "monitor":
        if args.loop:
            raise SystemExit(_run_loop(run_monitor_once, monitor_interval, "monitor"))
        raise SystemExit(run_monitor_once())
    if args.service == "backup":
        if args.loop:
            raise SystemExit(_run_loop(run_backup_once, backup_interval, "backup"))
        raise SystemExit(run_backup_once())
    raise SystemExit(run_all_supervisor(loop=True))


if __name__ == "__main__":
    main()
