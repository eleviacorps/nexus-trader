from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
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


SESSION = requests.Session()


def get_with_retry(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 60,
    retries: int = 3,
) -> Any:
    last_error = None
    for attempt in range(retries):
        try:
            response = SESSION.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except Exception as exc:  # pragma: no cover
            last_error = exc
            if attempt == retries - 1:
                raise
    raise last_error


def load_manifest(path: Path) -> dict[str, list[dict[str, Any]]]:
    return json.loads(path.read_text(encoding="utf-8"))


def extend_manifest(manifest: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    output = {key: list(value) for key, value in manifest.items()}
    current_year = datetime.now(timezone.utc).year

    existing_macro_names = {entry.get("name") for entry in output.get("macro", [])}
    extra_macro = [
        ("two_year_yield", "DGS2"),
        ("yield_curve_10y2y", "T10Y2Y"),
        ("high_yield_spread", "BAMLH0A0HYM2"),
        ("financial_conditions", "NFCI"),
    ]
    for name, series_id in extra_macro:
        if name in existing_macro_names:
            continue
        output.setdefault("macro", []).append(
            {
                "name": name,
                "kind": "fred_csv",
                "series_id": series_id,
                "filename": f"fred/{series_id}.csv",
                "priority": "high",
            }
        )

    extra_direct_macro = [
        (
            "treasury_par_yield_curve_archive",
            "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rate-archives/par-yield-curve-rates-1990-2023.csv",
            "treasury/par_yield_curve_rates_1990_2023.csv",
            "high",
        ),
        (
            "treasury_par_real_yield_curve_archive",
            "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rate-archives/par-real-yield-curve-rates-2003-2023.csv",
            "treasury/par_real_yield_curve_rates_2003_2023.csv",
            "high",
        ),
        (
            "treasury_real_long_term_archive",
            "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rate-archives/real-long-term-rates-2000-2023.csv",
            "treasury/real_long_term_rates_2000_2023.csv",
            "medium",
        ),
    ]
    for name, url, filename, priority in extra_direct_macro:
        if name in existing_macro_names:
            continue
        output.setdefault("macro", []).append(
            {
                "name": name,
                "kind": "direct",
                "url": url,
                "filename": filename,
                "priority": priority,
            }
        )

    existing_news_names = {entry.get("name") for entry in output.get("news", [])}
    extra_news = [
        (
            "central_bank_gold_headlines",
            "(gold OR bullion OR central bank gold reserves) AND (central bank OR reserve diversification OR de-dollarization)",
            "gdelt/central_bank_gold_headlines.csv",
            "medium",
        ),
        (
            "rates_shock_headlines",
            "(Treasury yields OR yield curve OR real yields OR bond market) AND (gold OR dollar OR XAUUSD)",
            "gdelt/rates_shock_headlines.csv",
            "medium",
        ),
        (
            "energy_inflation_headlines",
            "(oil OR crude OR energy prices) AND (inflation OR CPI OR Fed OR gold)",
            "gdelt/energy_inflation_headlines.csv",
            "medium",
        ),
    ]
    for name, query, filename, priority in extra_news:
        if name in existing_news_names:
            continue
        output.setdefault("news", []).append(
            {
                "name": name,
                "kind": "gdelt_doc",
                "query": query,
                "timespan": "365d",
                "maxrecords": 250,
                "mode": "ArtList",
                "format": "CSV",
                "filename": filename,
                "priority": priority,
            }
        )
    direct_news = [
        (
            "fomc_calendars",
            "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
            "fed/fomc_calendars.html",
            "high",
        ),
        (
            "world_gold_council_data_page",
            "https://www.gold.org/goldhub/data",
            "gold/world_gold_council_data.html",
            "medium",
        ),
    ]
    for year in range(max(2020, current_year - 3), current_year + 1):
        direct_news.append(
            (
                f"fed_monetary_releases_{year}",
                f"https://www.federalreserve.gov/newsevents/pressreleases/{year}-press-fomc.htm",
                f"fed/monetary_{year}.html",
                "medium",
            )
        )
    for name, url, filename, priority in direct_news:
        if name in existing_news_names:
            continue
        output.setdefault("news", []).append(
            {
                "name": name,
                "kind": "direct",
                "url": url,
                "filename": filename,
                "priority": priority,
            }
        )

    existing_crowd_names = {entry.get("name") for entry in output.get("crowd", [])}
    for year in range(2010, current_year + 1):
        financial_name = f"cftc_financial_futures_{year}"
        if financial_name not in existing_crowd_names:
            output.setdefault("crowd", []).append(
                {
                    "name": financial_name,
                    "kind": "direct",
                    "url": f"https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip",
                    "filename": f"cftc/fut_fin_txt_{year}.zip",
                    "priority": "high",
                }
            )
        combined_name = f"cftc_financial_combined_{year}"
        if combined_name not in existing_crowd_names:
            output.setdefault("crowd", []).append(
                {
                    "name": combined_name,
                    "kind": "direct",
                    "url": f"https://www.cftc.gov/files/dea/history/com_fin_txt_{year}.zip",
                    "filename": f"cftc/com_fin_txt_{year}.zip",
                    "priority": "medium",
                }
            )
        disagg_name = f"cftc_disaggregated_{year}"
        if disagg_name not in existing_crowd_names:
            output.setdefault("crowd", []).append(
                {
                    "name": disagg_name,
                    "kind": "direct",
                    "url": f"https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip",
                    "filename": f"cftc/fut_disagg_txt_{year}.zip",
                    "priority": "medium",
                }
            )
    return output


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
    response = get_with_retry(url, timeout=60)
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
    response = get_with_retry(url, params=params, timeout=120)
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
    response = get_with_retry(url, params=params, headers=headers, timeout=60)
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
    response = get_with_retry(entry["url"], timeout=60)
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
            response = get_with_retry(url, timeout=120)
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

    manifest = extend_manifest(load_manifest(DATASET_MANIFEST_PATH))
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
