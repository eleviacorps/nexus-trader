from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MT5Bridge:
    export_path: Path

    def export_signal(self, signal: dict[str, Any]) -> Path:
        self.export_path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(signal)
        payload.setdefault("transport", "json_signal_feed")
        self.export_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self.export_path

    def export_bulk_signals(self, signals: list[dict[str, Any]]) -> Path:
        self.export_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"transport": "manual_export", "signals": [dict(item) for item in signals]}
        self.export_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self.export_path

