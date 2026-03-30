from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

SYNC_PATHS = ["notebooks", "src", "config", "scripts", "tests"]
EXCLUDES = [".git/", "__pycache__/", ".ipynb_checkpoints/", "data_store/raw/"]


def build_rsync_command(server: str, remote_root: str, local_root: Path) -> list[str]:
    command = [
        "rsync",
        "-av",
        "--delete",
    ]
    for exclude in EXCLUDES:
        command.extend(["--exclude", exclude])
    for sync_path in SYNC_PATHS:
        command.append(str(local_root / sync_path))
    command.append(f"{server}:{remote_root}")
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync local Nexus Trader code to the ROCm server.")
    parser.add_argument("--server", required=True, help="Server host or user@host target")
    parser.add_argument("--remote-root", required=True, help="Remote Nexus root directory")
    parser.add_argument("--local-root", default=str(Path(__file__).resolve().parents[1]), help="Local project root")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without executing it")
    args = parser.parse_args()

    local_root = Path(args.local_root).resolve()
    command = build_rsync_command(args.server, args.remote_root, local_root)
    if args.dry_run:
        print(" ".join(command))
        return 0

    if shutil.which("rsync") is None:
        raise SystemExit("rsync is required for sync_local_to_server.py")

    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
