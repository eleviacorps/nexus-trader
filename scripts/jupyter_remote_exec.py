from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Iterable

import requests
import websocket


def _contents_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/api/contents/{path.strip('/')}"


def ensure_remote_directory(session: requests.Session, base_url: str, remote_path: str) -> None:
    response = session.put(
        _contents_url(base_url, remote_path),
        json={"type": "directory"},
        timeout=60,
    )
    response.raise_for_status()


def upload_text_file(session: requests.Session, base_url: str, remote_path: str, local_path: Path) -> None:
    content = local_path.read_text(encoding="utf-8")
    response = session.put(
        _contents_url(base_url, remote_path),
        json={"type": "file", "format": "text", "content": content},
        timeout=120,
    )
    response.raise_for_status()


def upload_paths(base_url: str, token: str, local_root: Path, remote_root: str, paths: Iterable[str]) -> None:
    session = requests.Session()
    session.params = {"token": token}
    for relative in paths:
        local_path = (local_root / relative).resolve()
        remote_path = f"{remote_root.rstrip('/')}/{relative.replace('\\', '/')}"
        parts = PurePosixPath(remote_path).parts
        accum = ""
        for part in parts[:-1]:
            if part == "/":
                accum = "/"
                continue
            accum = f"{accum.rstrip('/')}/{part}" if accum else part
            ensure_remote_directory(session, base_url, accum)
        upload_text_file(session, base_url, remote_path, local_path)
        print(f"uploaded {relative}")


def execute_code(base_url: str, token: str, code: str, timeout: int = 3600) -> int:
    session = requests.Session()
    session.params = {"token": token}
    kernel_response = session.post(f"{base_url.rstrip('/')}/api/kernels", timeout=60)
    kernel_response.raise_for_status()
    kernel_id = kernel_response.json()["id"]
    ws_url = f"ws://{base_url.rstrip('/').removeprefix('http://').removeprefix('https://')}/api/kernels/{kernel_id}/channels?token={token}"
    cookie_header = "; ".join(f"{key}={value}" for key, value in session.cookies.get_dict().items())
    ws = websocket.create_connection(
        ws_url,
        timeout=30,
        header=[f"Cookie: {cookie_header}"] if cookie_header else None,
    )
    msg_id = uuid.uuid4().hex
    message = {
        "header": {
            "msg_id": msg_id,
            "username": "codex",
            "session": uuid.uuid4().hex,
            "msg_type": "execute_request",
            "version": "5.3",
        },
        "parent_header": {},
        "metadata": {},
        "content": {
            "code": code,
            "silent": False,
            "store_history": False,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": True,
        },
        "channel": "shell",
    }
    ws.send(json.dumps(message))
    deadline = time.time() + timeout
    exit_code = 0
    while time.time() < deadline:
        raw = ws.recv()
        payload = json.loads(raw)
        msg_type = payload.get("msg_type")
        content = payload.get("content", {})
        parent = payload.get("parent_header", {})
        if parent.get("msg_id") != msg_id:
            continue
        if msg_type == "stream":
            print(content.get("text", ""), end="")
        elif msg_type in {"execute_result", "display_data"}:
            data = content.get("data", {})
            text = data.get("text/plain")
            if text:
                print(text)
        elif msg_type == "error":
            print("\n".join(content.get("traceback", [])))
            exit_code = 1
            break
        elif msg_type == "status" and content.get("execution_state") == "idle":
            break
    ws.close()
    try:
        session.delete(f"{base_url.rstrip('/')}/api/kernels/{kernel_id}", timeout=30)
    except Exception:
        pass
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload files to and execute code on a Jupyter server.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--upload", nargs="*", default=[])
    parser.add_argument("--code-file", default="")
    parser.add_argument("--local-root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()

    local_root = Path(args.local_root).resolve()
    if args.upload:
        upload_paths(args.base_url, args.token, local_root, args.remote_root, args.upload)
    if args.code_file:
        code = Path(args.code_file).read_text(encoding="utf-8")
        return execute_code(args.base_url, args.token, code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
