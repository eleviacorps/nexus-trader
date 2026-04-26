"""News panel widget."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from textual.widget import Widget
from textual.widgets import Static


class NewsPanel(Widget):
    """Scrollable news list with category coloring."""

    def __init__(self, news_aggregator, refresh_seconds: int = 300, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.news_aggregator = news_aggregator
        self.refresh_seconds = max(30, int(refresh_seconds))
        self._static = Static("Loading news...")
        self._task: asyncio.Task | None = None

    def compose(self):
        yield self._static

    async def on_mount(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="nexus_news_panel")

    async def on_unmount(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        while True:
            items = await self.news_aggregator.fetch_all()
            lines = []
            for item in items[:15]:
                ts = item.published_at.strftime("%H:%M")
                source = item.source[:12]
                lines.append(f"[{ts}] [{source}] {item.title}")
            if not lines:
                lines = ["No relevant news items"]
            self._static.update("\n".join(lines))
            await asyncio.sleep(self.refresh_seconds)

