from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import tarfile
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.jupyter_remote_exec import _build_session, _contents_url, execute_code


VERIFY_IMPORTS = [
    "src.v12.bar_consistent_features",
    "src.v16.confidence_tier",
    "src.v17.mmm",
    "src.v18.mfg_beliefs",
    "src.v20.mamba_backbone",
    "src.v21.xlstm_backbone",
    "src.v22.hybrid_risk_judge",
    "src.v22.ensemble_judge_stack",
    "src.v22.online_hmm",
    "src.v22.circuit_breaker",
]


def create_workspace_tarball() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for root_name in ("src", "scripts", "config"):
            root = PROJECT_ROOT / root_name
            if not root.exists():
                continue
            for path in sorted(root.rglob("*.py")):
                tar.add(path, arcname=str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"))
    buf.seek(0)
    return buf.read()


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


def upload_tarball(base_url: str, token: str, remote_root: str, tarball_bytes: bytes) -> str:
    remote_file = f"{remote_root.rstrip('/')}/workspace_sync/workspace_src_scripts_config.tar.gz"
    _ensure_remote_parents(base_url, token, remote_file)
    session = _build_session(base_url, token)
    payload = {"type": "file", "format": "base64", "content": base64.b64encode(tarball_bytes).decode("ascii")}
    response = session.put(_contents_url(base_url, remote_file), json=payload, timeout=900)
    response.raise_for_status()
    return remote_file


def extract_remote_workspace(base_url: str, token: str, remote_root: str, remote_tarball: str) -> int:
    remote_abs_root = f"/home/rocm-user/jupyter/{remote_root.strip('/').lstrip('/')}"
    code = f"""
import shutil
import tarfile
from pathlib import Path

workspace = Path(r"{remote_abs_root}")
tar_path = workspace / "workspace_sync" / "workspace_src_scripts_config.tar.gz"
staging = workspace / "workspace_sync" / "_extract_stage"
staging.parent.mkdir(parents=True, exist_ok=True)
if staging.exists():
    shutil.rmtree(staging)
staging.mkdir(parents=True, exist_ok=True)
with tarfile.open(tar_path, mode="r:gz") as tar:
    tar.extractall(staging)
for name in ("src", "scripts", "config"):
    candidate = staging / name
    if candidate.exists():
        destination = workspace / name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(candidate), str(destination))
counts = {{
    "src_py": len(list((workspace / "src").rglob("*.py"))) if (workspace / "src").exists() else 0,
    "scripts_py": len(list((workspace / "scripts").rglob("*.py"))) if (workspace / "scripts").exists() else 0,
    "config_py": len(list((workspace / "config").rglob("*.py"))) if (workspace / "config").exists() else 0,
}}
print(counts)
"""
    return execute_code(base_url, token, code, timeout=600)


def verify_remote_imports(base_url: str, token: str, remote_root: str) -> tuple[int, dict[str, str]]:
    remote_abs_root = f"/home/rocm-user/jupyter/{remote_root.strip('/').lstrip('/')}"
    imports_json = json.dumps(VERIFY_IMPORTS)
    code = f"""
import importlib
import json
import os
import sys

workspace = r"{remote_abs_root}"
if workspace not in sys.path:
    sys.path.insert(0, workspace)
os.chdir(workspace)
modules = json.loads({imports_json!r})
results = {{}}
for module_name in modules:
    try:
        importlib.import_module(module_name)
        results[module_name] = "PASS"
    except Exception as exc:
        results[module_name] = f"FAIL: {{type(exc).__name__}}: {{exc}}"
print(json.dumps(results, indent=2))
print("All imports OK" if all(value == "PASS" for value in results.values()) else "Import verification failed")
"""
    exit_code = execute_code(base_url, token, code, timeout=600)
    session = _build_session(base_url, token)
    # Keep tool contract simple: rerun compactly and parse via contents-less kernel execution is overkill,
    # so run once more with machine-friendly JSON-only output.
    del session
    code_json = f"""
import importlib
import json
import os
import sys

workspace = r"{remote_abs_root}"
if workspace not in sys.path:
    sys.path.insert(0, workspace)
os.chdir(workspace)
modules = json.loads({imports_json!r})
results = {{}}
for module_name in modules:
    try:
        importlib.import_module(module_name)
        results[module_name] = "PASS"
    except Exception as exc:
        results[module_name] = f"FAIL: {{type(exc).__name__}}: {{exc}}"
print(json.dumps(results))
"""
    # Capture by temporarily using the underlying Jupyter tool output through a local wrapper.
    # The caller still receives the pass/fail dict for reporting.
    import subprocess
    import sys as _sys

    helper = Path(__file__).resolve().parent / "_tmp_sync_verify_v21.py"
    helper.write_text(
        f"import sys\nsys.path.insert(0, {str(PROJECT_ROOT)!r})\nfrom scripts.jupyter_remote_exec import execute_code\n"
        f"code = {code_json!r}\n"
        f"raise SystemExit(execute_code({base_url!r}, {token!r}, code, timeout=600))\n",
        encoding="utf-8",
    )
    try:
        completed = subprocess.run([_sys.executable, str(helper)], capture_output=True, text=True, check=False)
        stdout = completed.stdout.strip().splitlines()
        parsed: dict[str, str] = {}
        for line in reversed(stdout):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                parsed = json.loads(line)
                break
        return exit_code, parsed
    finally:
        if helper.exists():
            helper.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="Create, upload, extract, and verify a full remote workspace tarball.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--remote-root", required=True)
    args = parser.parse_args()

    tarball = create_workspace_tarball()
    remote_tarball = upload_tarball(args.base_url, args.token, args.remote_root, tarball)
    extract_exit = extract_remote_workspace(args.base_url, args.token, args.remote_root, remote_tarball)
    verify_exit, verify_results = verify_remote_imports(args.base_url, args.token, args.remote_root)
    payload = {
        "remote_tarball": remote_tarball,
        "tarball_size_bytes": len(tarball),
        "extract_exit_code": extract_exit,
        "verify_exit_code": verify_exit,
        "imports": verify_results,
        "success": bool(extract_exit == 0 and verify_exit == 0 and verify_results and all(value == "PASS" for value in verify_results.values())),
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
