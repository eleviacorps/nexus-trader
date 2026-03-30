from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from config.project_config import DATA_ROOT, LOCAL_PROJECT_ROOT, PROJECT_ROOT


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def data_path(*parts: str) -> Path:
    return DATA_ROOT.joinpath(*parts)


def local_path(*parts: str) -> Path:
    return LOCAL_PROJECT_ROOT.joinpath(*parts)
