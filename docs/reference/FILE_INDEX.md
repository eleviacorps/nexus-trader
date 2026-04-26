# File Index Database

The repo includes a local SQLite file index so future agents can find files quickly without a full recursive search.

## Build / Refresh

```bash
python scripts/build_file_index.py
```

Default output:

- `meta/file_index.sqlite`

## Query Examples

```bash
python scripts/query_file_index.py --name evaluate_v26
python scripts/query_file_index.py --ext .py --category evaluation
python scripts/query_file_index.py --path nexus_packaged/v30 --limit 20
```

## Database Contents

- Table: `files`
  - `path`, `dir_path`, `name`, `ext`, `top_level`, `category`
  - `size_bytes`, `mtime_utc`, `tracked`
- Table: `meta`
  - index timestamp and file counts
- Virtual table: `files_fts`
  - full-text search index for path/name/category
