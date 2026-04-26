from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class FileRow:
    path: str
    dir_path: str
    name: str
    ext: str
    top_level: str
    category: str
    size_bytes: int
    mtime_utc: str
    tracked: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a SQLite file-location index for the repository.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root directory.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("meta/file_index.sqlite"),
        help="Output SQLite database path (relative to --root if not absolute).",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[".git", "__pycache__"],
        help="Directory name to exclude. Can be passed multiple times.",
    )
    return parser.parse_args()


def resolve_db_path(root: Path, db_path: Path) -> Path:
    if db_path.is_absolute():
        return db_path
    return root / db_path


def get_tracked_files(root: Path) -> set[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def classify(top_level: str, rel_path: str) -> str:
    if rel_path.startswith("scripts/evaluation/"):
        return "evaluation"
    mapping = {
        "src": "source",
        "scripts": "scripts",
        "tests": "tests",
        "docs": "docs",
        "nexus_old": "archive",
        "nexus_packaged": "packaged-runtime",
        "notebooks": "notebooks",
        "config": "config",
        "infra": "infra",
        "ui": "ui",
    }
    return mapping.get(top_level, top_level or "root")


def iter_files(root: Path, exclude_dirs: set[str], tracked_files: set[str]) -> list[FileRow]:
    rows: list[FileRow] = []
    for current_root, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        current_root_path = Path(current_root)
        for name in files:
            full_path = current_root_path / name
            rel = full_path.relative_to(root).as_posix()
            stat = full_path.stat()
            top_level = rel.split("/", 1)[0] if "/" in rel else ""
            rows.append(
                FileRow(
                    path=rel,
                    dir_path=str(Path(rel).parent).replace("\\", "/"),
                    name=name,
                    ext=Path(name).suffix.lower(),
                    top_level=top_level,
                    category=classify(top_level, rel),
                    size_bytes=int(stat.st_size),
                    mtime_utc=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                    tracked=1 if rel in tracked_files else 0,
                )
            )
    return rows


def build_database(db_path: Path, rows: list[FileRow], root: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")

        conn.executescript(
            """
            DROP TABLE IF EXISTS files;
            DROP TABLE IF EXISTS meta;
            DROP TABLE IF EXISTS files_fts;

            CREATE TABLE files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL UNIQUE,
                dir_path TEXT NOT NULL,
                name TEXT NOT NULL,
                ext TEXT NOT NULL,
                top_level TEXT NOT NULL,
                category TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                mtime_utc TEXT NOT NULL,
                tracked INTEGER NOT NULL
            );

            CREATE INDEX idx_files_name ON files(name);
            CREATE INDEX idx_files_ext ON files(ext);
            CREATE INDEX idx_files_top_level ON files(top_level);
            CREATE INDEX idx_files_category ON files(category);
            CREATE INDEX idx_files_tracked ON files(tracked);

            CREATE TABLE meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE VIRTUAL TABLE files_fts USING fts5(
                path,
                name,
                category,
                content='files',
                content_rowid='id'
            );
            """
        )

        conn.executemany(
            """
            INSERT INTO files (
                path, dir_path, name, ext, top_level, category, size_bytes, mtime_utc, tracked
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row.path,
                    row.dir_path,
                    row.name,
                    row.ext,
                    row.top_level,
                    row.category,
                    row.size_bytes,
                    row.mtime_utc,
                    row.tracked,
                )
                for row in rows
            ],
        )

        conn.execute(
            """
            INSERT INTO files_fts(rowid, path, name, category)
            SELECT id, path, name, category FROM files
            """
        )

        now_utc = datetime.now(tz=timezone.utc).isoformat()
        total_files = str(len(rows))
        tracked_files = str(sum(row.tracked for row in rows))
        conn.executemany(
            "INSERT INTO meta(key, value) VALUES (?, ?)",
            [
                ("indexed_at_utc", now_utc),
                ("root", str(root.resolve())),
                ("total_files", total_files),
                ("tracked_files", tracked_files),
            ],
        )

        conn.commit()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    db_path = resolve_db_path(root, args.db).resolve()
    exclude_dirs = set(args.exclude_dir or [])

    tracked = get_tracked_files(root)
    rows = iter_files(root, exclude_dirs, tracked)
    build_database(db_path, rows, root)

    print(f"File index written: {db_path}")
    print(f"Indexed files: {len(rows)}")
    print(f"Tracked files: {sum(row.tracked for row in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
