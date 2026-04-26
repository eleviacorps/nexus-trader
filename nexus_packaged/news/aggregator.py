"""Async RSS news aggregation and relevance scoring."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

try:
    import feedparser  # type: ignore
except Exception:  # noqa: BLE001
    feedparser = None

try:
    import httpx  # type: ignore
except Exception:  # noqa: BLE001
    httpx = None


KEYWORDS = [
    "gold",
    "xau",
    "fed",
    "rate",
    "inflation",
    "dollar",
    "usd",
    "treasury",
    "fomc",
    "powell",
    "yield",
    "commodities",
    "oil",
    "geopolit",
    "war",
    "sanction",
]


@dataclass
class NewsItem:
    """Normalized news item."""

    title: str
    source: str
    category: str
    published_at: datetime
    summary: str
    url: str
    relevance_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "source": self.source,
            "category": self.category,
            "published_at": self.published_at.isoformat(),
            "summary": self.summary,
            "url": self.url,
            "relevance_score": float(self.relevance_score),
        }


class NewsAggregator:
    """Aggregates RSS feeds with in-memory TTL cache."""

    def __init__(self, config: dict):
        self.config = dict(config.get("news", {}))
        self.sources = list(self.config.get("rss_sources", []))
        self.ttl = int(self.config.get("cache_ttl_seconds", 300))
        self.timeout = int(self.config.get("fetch_timeout_seconds", 10))
        self.min_score = float(self.config.get("min_relevance_score", 0.1))
        self.max_items = int(self.config.get("max_display_items", 15))
        self._cache: list[NewsItem] = []
        self._cache_ts: datetime | None = None
        self.logger = logging.getLogger("nexus.system")
        self.error_logger = logging.getLogger("nexus.errors")

    def _score(self, title: str, summary: str) -> float:
        text = f"{title} {summary}".lower()
        hits = sum(text.count(keyword) for keyword in KEYWORDS)
        length = max(1, len(text.split()))
        score = hits / length
        return float(min(1.0, score * 10.0))

    async def _fetch_source(self, client: httpx.AsyncClient, source: dict[str, Any]) -> list[NewsItem]:
        url = str(source.get("url", ""))
        name = str(source.get("name", "unknown"))
        category = str(source.get("category", "macro"))
        try:
            response = await client.get(url, timeout=self.timeout)
            response.raise_for_status()
            if feedparser is None:
                return []
            parsed = feedparser.parse(response.text)
            items: list[NewsItem] = []
            for entry in parsed.entries[:40]:
                title = str(entry.get("title", "")).strip()
                summary = str(entry.get("summary", "")).strip()
                link = str(entry.get("link", "")).strip()
                published = entry.get("published") or entry.get("updated")
                if published:
                    try:
                        published_dt = parsedate_to_datetime(str(published))
                        if published_dt.tzinfo is None:
                            published_dt = published_dt.replace(tzinfo=timezone.utc)
                        else:
                            published_dt = published_dt.astimezone(timezone.utc)
                    except Exception:  # noqa: BLE001
                        published_dt = datetime.now(timezone.utc)
                else:
                    published_dt = datetime.now(timezone.utc)
                score = self._score(title, summary)
                if score <= self.min_score:
                    continue
                items.append(
                    NewsItem(
                        title=title,
                        source=name,
                        category=category,
                        published_at=published_dt,
                        summary=summary,
                        url=link,
                        relevance_score=score,
                    )
                )
            return items
        except Exception as exc:  # noqa: BLE001
            self.error_logger.warning("News source fetch failed (%s): %s", name, exc)
            return []

    async def fetch_all(self) -> list[NewsItem]:
        """Fetch and merge all sources with TTL cache."""
        now = datetime.now(timezone.utc)
        if self._cache_ts is not None and (now - self._cache_ts).total_seconds() <= self.ttl:
            return list(self._cache)
        if not self.sources:
            self._cache = []
            self._cache_ts = now
            return []
        if httpx is None:
            self.error_logger.warning("httpx is unavailable; returning cached news only.")
            return list(self._cache)
        async with httpx.AsyncClient(follow_redirects=True) as client:
            tasks = [self._fetch_source(client, source) for source in self.sources]
            batches = await asyncio.gather(*tasks, return_exceptions=False)
        merged = [item for batch in batches for item in batch]
        merged.sort(key=lambda item: item.published_at, reverse=True)
        self._cache = merged[: self.max_items]
        self._cache_ts = now
        return list(self._cache)

    def get_cached(self) -> list[NewsItem]:
        """Return cached items."""
        return list(self._cache)
