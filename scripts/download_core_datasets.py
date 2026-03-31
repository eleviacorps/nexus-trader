from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from shutil import which
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.project_config import (  # noqa: E402
    DATASET_MANIFEST_PATH,
    DOWNLOAD_REPORT_PATH,
    RAW_CROWD_DIR,
    RAW_MACRO_DIR,
    RAW_NEWS_DIR,
)

try:
    import requests  # type: ignore  # noqa: E402
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"requests is required for dataset download: {exc}")

try:
    import yfinance as yf  # type: ignore  # noqa: E402
except ImportError:  # pragma: no cover
    yf = None


def load_manifest(path: Path) -> dict[str, list[dict[str, Any]]]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def output_path_for(category: str, filename: str) -> Path:
    base = {
        "macro": RAW_MACRO_DIR,
        "news": RAW_NEWS_DIR,
        "crowd": RAW_CROWD_DIR,
    }[category]
    return base / filename


def download_with_aria2(url: str, destination: Path) -> None:
    ensure_parent(destination)
    if which("aria2c") is None:
        raise FileNotFoundError("aria2c is not installed.")
    command = [
        "aria2c",
        "-x",
        "16",
        "-s",
        "16",
        "-k",
        "1M",
        "-c",
        url,
        "-d",
        str(destination.parent),
        "-o",
        destination.name,
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or f"aria2c failed for {url}")


def download_fred_csv(series_id: str, destination: Path) -> dict[str, Any]:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    ensure_parent(destination)
    destination.write_text(response.text, encoding="utf-8")
    return {"url": url, "bytes": len(response.content)}


def download_gdelt_doc(entry: dict[str, Any], destination: Path) -> dict[str, Any]:
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": entry["query"],
        "mode": entry.get("mode", "ArtList"),
        "format": entry.get("format", "CSV"),
        "timespan": entry.get("timespan", "365d"),
        "maxrecords": entry.get("maxrecords", 250),
    }
    response = requests.get(url, params=params, timeout=120)
    response.raise_for_status()
    ensure_parent(destination)
    destination.write_text(response.text, encoding="utf-8")
    return {"url": response.url, "bytes": len(response.content)}


def download_reddit_search(entry: dict[str, Any], destination: Path) -> dict[str, Any]:
    subreddit = entry["subreddit"]
    url = f"https://www.reddit.com/r/{subreddit}/search.json"
    params = {
        "q": entry["query"],
        "restrict_sr": "1",
        "sort": "new",
        "t": "year",
        "limit": entry.get("limit", 100),
    }
    headers = {"User-Agent": "nexus-trader/0.1 dataset bootstrap"}
    response = requests.get(url, params=params, headers=headers, timeout=60)
    response.raise_for_status()
    payload = response.json()
    ensure_parent(destination)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    children = payload.get("data", {}).get("children", [])
    return {"url": response.url, "posts": len(children), "bytes": len(response.content)}


def download_yfinance_csv(entry: dict[str, Any], destination: Path) -> dict[str, Any]:
    if yf is None:
        raise ImportError("yfinance is required for yfinance_csv downloads.")
    ticker = entry["ticker"]
    frame = yf.download(
        ticker,
        period=entry.get("period", "max"),
        interval=entry.get("interval", "1d"),
        auto_adjust=False,
        progress=False,
    )
    if frame.empty:
        raise ValueError(f"No rows returned for ticker {ticker}")
    ensure_parent(destination)
    frame.to_csv(destination)
    return {"ticker": ticker, "rows": int(len(frame))}


def download_json_api(entry: dict[str, Any], destination: Path) -> dict[str, Any]:
    response = requests.get(entry["url"], timeout=60)
    response.raise_for_status()
    ensure_parent(destination)
    destination.write_text(json.dumps(response.json(), indent=2), encoding="utf-8")
    return {"url": entry["url"], "bytes": len(response.content)}


def download_entry(category: str, entry: dict[str, Any], force: bool) -> dict[str, Any]:
    destination = output_path_for(category, entry["filename"])
    if destination.exists() and not force:
        return {
            "name": entry["name"],
            "category": category,
            "status": "skipped",
            "path": str(destination),
            "reason": "exists",
        }

    kind = entry["kind"]
    if kind == "fred_csv":
        detail = download_fred_csv(entry["series_id"], destination)
    elif kind == "gdelt_doc":
        detail = download_gdelt_doc(entry, destination)
    elif kind == "reddit_search_json":
        detail = download_reddit_search(entry, destination)
    elif kind == "yfinance_csv":
        detail = download_yfinance_csv(entry, destination)
    elif kind == "json_api":
        detail = download_json_api(entry, destination)
    elif kind == "direct":
        url = entry["url"]
        if which("aria2c") is not None:
            download_with_aria2(url, destination)
            detail = {"url": url, "transport": "aria2c"}
        else:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            ensure_parent(destination)
            destination.write_bytes(response.content)
            detail = {"url": url, "transport": "requests", "bytes": len(response.content)}
    else:
        raise ValueError(f"Unsupported dataset kind: {kind}")

    return {
        "name": entry["name"],
        "category": category,
        "status": "downloaded",
        "path": str(destination),
        **detail,
    }


def selected_entries(manifest: dict[str, list[dict[str, Any]]], category: str) -> list[tuple[str, dict[str, Any]]]:
    if category == "all":
        categories = ["macro", "news", "crowd"]
    else:
        categories = [category]
    output = []
    for current in categories:
        output.extend((current, entry) for entry in manifest.get(current, []))
    return output


def print_plan(entries: list[tuple[str, dict[str, Any]]]) -> None:
    for category, entry in entries:
        print(f"[{category}] {entry['name']} -> {entry['filename']} ({entry['kind']}, priority={entry.get('priority', 'n/a')})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download core Nexus Trader datasets as quickly as possible.")
    parser.add_argument("--category", choices=["all", "macro", "news", "crowd"], default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--allow-errors", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(DATASET_MANIFEST_PATH)
    entries = selected_entries(manifest, args.category)
    if args.plan:
        print_plan(entries)
        return 0

    parallel_entries = [(category, entry) for category, entry in entries if entry["kind"] != "yfinance_csv"]
    sequential_entries = [(category, entry) for category, entry in entries if entry["kind"] == "yfinance_csv"]

    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {
            executor.submit(download_entry, category, entry, args.force): (category, entry["name"])
            for category, entry in parallel_entries
        }
        for future in as_completed(future_map):
            category, name = future_map[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover
                result = {"name": name, "category": category, "status": "error", "error": str(exc)}
            results.append(result)
            print(json.dumps(result, indent=2))

    for category, entry in sequential_entries:
        try:
            result = download_entry(category, entry, args.force)
        except Exception as exc:  # pragma: no cover
            result = {"name": entry["name"], "category": category, "status": "error", "error": str(exc)}
        results.append(result)
        print(json.dumps(result, indent=2))

    write_json(DOWNLOAD_REPORT_PATH, {"category": args.category, "results": results})
    failures = [row for row in results if row.get("status") == "error"]
    if args.allow_errors:
        return 0
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
