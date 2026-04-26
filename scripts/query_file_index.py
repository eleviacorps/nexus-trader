from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the repository file-location SQLite index.")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("meta/file_index.sqlite"),
        help="Path to the SQLite index database.",
    )
    parser.add_argument("--name", type=str, default="", help="Name contains filter (case-insensitive).")
    parser.add_argument("--path", type=str, default="", help="Path contains filter (case-insensitive).")
    parser.add_argument("--ext", type=str, default="", help="File extension filter, e.g. .py")
    parser.add_argument("--category", type=str, default="", help="Category filter.")
    parser.add_argument("--top-level", type=str, default="", help="Top-level directory filter.")
    parser.add_argument(
        "--tracked",
        choices=["any", "yes", "no"],
        default="any",
        help="Filter by git-tracked status.",
    )
    parser.add_argument("--limit", type=int, default=50, help="Max rows to return.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    db_path = args.db.resolve()
    if not db_path.exists():
        print(f"Index database not found: {db_path}")
        print("Build it first with: python scripts/build_file_index.py")
        return 1

    clauses: list[str] = []
    params: list[object] = []

    if args.name:
        clauses.append("LOWER(name) LIKE ?")
        params.append(f"%{args.name.lower()}%")
    if args.path:
        clauses.append("LOWER(path) LIKE ?")
        params.append(f"%{args.path.lower()}%")
    if args.ext:
        clauses.append("ext = ?")
        params.append(args.ext.lower())
    if args.category:
        clauses.append("category = ?")
        params.append(args.category)
    if args.top_level:
        clauses.append("top_level = ?")
        params.append(args.top_level)
    if args.tracked == "yes":
        clauses.append("tracked = 1")
    elif args.tracked == "no":
        clauses.append("tracked = 0")

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
        SELECT path, size_bytes, tracked, category
        FROM files
        {where_sql}
        ORDER BY path
        LIMIT ?
    """
    params.append(max(1, int(args.limit)))

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()

    if not rows:
        print("No rows matched.")
        return 0

    for path, size_bytes, tracked, category in rows:
        tracked_flag = "tracked" if tracked else "untracked"
        print(f"{path} | {size_bytes} bytes | {tracked_flag} | {category}")
    print(f"\nRows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
