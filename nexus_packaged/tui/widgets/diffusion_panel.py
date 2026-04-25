"""Diffusion paths panel widget."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from textual.widget import Widget
from textual.widgets import Static


def _palette_64() -> list[str]:
    base = [
        "cyan",
        "green",
        "red",
        "yellow",
        "blue",
        "magenta",
        "white",
        "orange",
    ]
    colors = []
    while len(colors) < 64:
        colors.extend(base)
    return colors[:64]


class DiffusionPanel(Widget):
    """Displays up to 64 generated paths in ASCII form using plotext."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._static = Static("Generated Paths: 0 | Updated: --:--:--")
        self._palette = _palette_64()

    def compose(self):
        yield self._static

    def update_paths(self, paths: np.ndarray) -> None:
        if paths.ndim != 2:
            self._static.update("Generated Paths: 0 | Updated: --:--:--")
            return
        try:
            import plotext as plt  # type: ignore

            plt.clear_figure()
            plt.theme("clear")
            horizon = paths.shape[1]
            x = list(range(horizon))
            paths_to_plot = min(64, paths.shape[0])
            for idx in range(paths_to_plot):
                color = self._palette[idx]
                plt.plot(x, paths[idx].tolist(), color=color, marker=None)
            median = np.median(paths[:paths_to_plot], axis=0)
            p10 = np.percentile(paths[:paths_to_plot], 10, axis=0)
            p90 = np.percentile(paths[:paths_to_plot], 90, axis=0)
            plt.plot(x, median.tolist(), color="white")
            plt.plot(x, p10.tolist(), color="gray")
            plt.plot(x, p90.tolist(), color="gray")
            plt.title(f"Generated Paths: {paths_to_plot} | Updated: {datetime.now().strftime('%H:%M:%S')}")
            rendered = plt.build()
            self._static.update(rendered)
        except Exception:
            self._static.update(
                f"Generated Paths: {paths.shape[0]} | Updated: {datetime.now().strftime('%H:%M:%S')}"
            )

