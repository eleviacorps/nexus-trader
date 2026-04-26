from __future__ import annotations

import argparse
import base64
from pathlib import Path
from typing import Iterable

import requests


DEFAULT_PATHS = [
    "models",
    "checkpoints",
    "outputs",
    "data/branches",
    "data/features",
    "data/processed",
    "data/embeddings",
]


def api_url(base_url: str, remote_path: str) -> str:
    normalized_base = base_url.rstrip("/")
    normalized_path = remote_path.strip("/").replace("\\", "/")
    return f"{normalized_base}/api/contents/{normalized_path}"


def to_local_relative(remote_path: str, remote_root: str) -> Path:
    normalized_remote = remote_path.replace("\\", "/").strip("/")
    normalized_root = remote_root.replace("\\", "/").strip("/")
    if normalized_remote.startswith(normalized_root + "/"):
        normalized_remote = normalized_remote[len(normalized_root) + 1 :]
    elif normalized_remote == normalized_root:
        normalized_remote = ""
    return Path(normalized_remote)


def download_file(session: requests.Session, base_url: str, remote_path: str, local_root: Path, remote_root: str) -> None:
    response = session.get(api_url(base_url, remote_path), params={"content": 1}, timeout=120)
    response.raise_for_status()
    payload = response.json()
    relative_path = to_local_relative(remote_path, remote_root)
    target = local_root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)

    fmt = payload.get("format")
    content = payload.get("content")
    if fmt == "base64":
        target.write_bytes(base64.b64decode(content))
    else:
        target.write_text(content or "", encoding="utf-8")
    print(f"downloaded {remote_path}")


def walk_and_download(session: requests.Session, base_url: str, remote_path: str, local_root: Path, remote_root: str) -> None:
    response = session.get(api_url(base_url, remote_path), params={"content": 1}, timeout=120)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, list):
        return
    kind = payload.get("type")
    if kind == "file":
        download_file(session, base_url, remote_path, local_root, remote_root)
        return
    if kind != "directory":
        return
    for item in payload.get("content", []):
        path = item.get("path")
        if not path:
            continue
        if item.get("type") == "directory":
            walk_and_download(session, base_url, path, local_root, remote_root)
        elif item.get("type") == "file":
            download_file(session, base_url, path, local_root, remote_root)


def sync_paths(base_url: str, token: str, local_root: Path, remote_root: str, paths: Iterable[str]) -> None:
    session = requests.Session()
    session.params = {"token": token}
    for relative in paths:
        remote_path = f"{remote_root.rstrip('/')}/{relative.strip('/')}"
        walk_and_download(session, base_url, remote_path, local_root, remote_root)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Nexus server artifacts from a Jupyter server to the local machine.")
    parser.add_argument("--base-url", required=True, help="Example: http://129.212.178.105")
    parser.add_argument("--token", required=True, help="Jupyter token")
    parser.add_argument("--remote-root", required=True, help="Remote Nexus root, e.g. /home/rocm-user/jupyter/nexus")
    parser.add_argument("--local-root", default=str(Path(__file__).resolve().parents[1]), help="Local project root")
    parser.add_argument("--paths", nargs="*", default=DEFAULT_PATHS, help="Relative project paths to download")
    args = parser.parse_args()

    sync_paths(
        base_url=args.base_url,
        token=args.token,
        local_root=Path(args.local_root).resolve(),
        remote_root=args.remote_root,
        paths=args.paths,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
