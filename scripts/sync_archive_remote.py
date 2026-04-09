from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import V21_REMOTE_ARCHIVE_PATH
from scripts.jupyter_remote_exec import _build_session, _contents_url, execute_code


def _ensure_remote_parents(base_url: str, token: str, remote_file: str) -> None:
    session = _build_session(base_url, token)
    parts = PurePosixPath(remote_file).parts
    accum = ""
    for part in parts[:-1]:
        if part == "/":
            accum = "/"
            continue
        accum = f"{accum.rstrip('/')}/{part}" if accum else part
        response = session.put(_contents_url(base_url, accum), json={"type": "directory"}, timeout=60)
        response.raise_for_status()


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload the local XAUUSD 1-minute archive to the remote Jupyter workspace.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--local-path", default=str(PROJECT_ROOT / "data_store" / "processed" / "XAUUSD_1m_full.parquet"))
    parser.add_argument("--remote-path", default="data/raw/xauusd_1min_2007_2024.parquet")
    args = parser.parse_args()

    local_path = Path(args.local_path).resolve()
    if not local_path.exists():
        raise SystemExit(f"Local archive not found: {local_path}")

    remote_file = f"{args.remote_root.rstrip('/')}/{args.remote_path.lstrip('/')}"
    _ensure_remote_parents(args.base_url, args.token, remote_file)
    session = _build_session(args.base_url, args.token)
    payload = {
        "type": "file",
        "format": "base64",
        "content": base64.b64encode(local_path.read_bytes()).decode("ascii"),
    }
    response = session.put(_contents_url(args.base_url, remote_file), json=payload, timeout=7200)
    response.raise_for_status()

    remote_abs_path = f"/home/rocm-user/jupyter/{args.remote_root.strip('/').lstrip('/')}/{args.remote_path.lstrip('/')}"
    code = f"""
from pathlib import Path
path = Path(r"{remote_abs_path}")
print({{"exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0}})
"""
    verify_exit = execute_code(args.base_url, args.token, code, timeout=300)
    summary = {
        "local_path": str(local_path),
        "remote_file": remote_file,
        "remote_abs_path": remote_abs_path,
        "local_size_bytes": local_path.stat().st_size,
        "verify_exit_code": verify_exit,
        "success": verify_exit == 0,
    }
    print(json.dumps(summary, indent=2))
    return 0 if verify_exit == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
