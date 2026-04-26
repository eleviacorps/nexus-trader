from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.jupyter_remote_exec import _build_session, _contents_url


def download_file(base_url: str, token: str, remote_path: str, local_path: Path) -> dict[str, object]:
    session = _build_session(base_url, token)
    response = session.get(_contents_url(base_url, remote_path), timeout=3600)
    response.raise_for_status()
    payload = response.json()
    file_format = str(payload.get("format", "text"))
    content = payload.get("content", "")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if file_format == "base64":
        local_path.write_bytes(base64.b64decode(str(content)))
    else:
        local_path.write_text(str(content), encoding="utf-8")
    return {
        "remote_path": remote_path,
        "local_path": str(local_path),
        "bytes": local_path.stat().st_size,
        "format": file_format,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Download files from the remote Jupyter workspace.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--remote-paths", nargs="+", required=True)
    parser.add_argument("--local-root", default=str(PROJECT_ROOT))
    args = parser.parse_args()

    local_root = Path(args.local_root).resolve()
    results: list[dict[str, object]] = []
    for relative in args.remote_paths:
        normalized = relative.replace("\\", "/").lstrip("/")
        remote_path = f"{args.remote_root.rstrip('/')}/{normalized}"
        local_path = local_root / PurePosixPath(normalized)
        results.append(download_file(args.base_url, args.token, remote_path, local_path))
        print(f"downloaded {normalized} -> {local_path}")
    print(json.dumps({"downloads": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
