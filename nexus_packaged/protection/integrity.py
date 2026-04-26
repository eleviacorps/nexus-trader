"""Binary integrity checks for packaged executable."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def compute_exe_hash(exe_path: str) -> str:
    """SHA-256 digest of an executable or script file."""
    path = Path(exe_path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_integrity(exe_path: str, expected_hash_path: str) -> bool:
    """Verify executable hash against expected hash file.

    If running from a Python script (non-executable launch), return True.
    """
    runtime_path = Path(exe_path)
    suffix = runtime_path.suffix.lower()
    if suffix in {".py", ".pyw"}:
        return True
    expected_path = Path(expected_hash_path)
    if not expected_path.exists():
        return False
    expected_line = expected_path.read_text(encoding="utf-8").strip()
    expected_hash = expected_line.split()[0] if expected_line else ""
    if not expected_hash:
        return False
    actual_hash = compute_exe_hash(str(runtime_path))
    return actual_hash == expected_hash


def current_runtime_path() -> str:
    """Return runtime executable/script path used for integrity checks."""
    return str(Path(sys.argv[0]).resolve())

