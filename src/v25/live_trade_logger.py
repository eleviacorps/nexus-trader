from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class LiveTradeLogger:
    def __init__(self, decision_log_path: Path, csv_log_path: Path):
        self.decision_log_path = Path(decision_log_path)
        self.csv_log_path = Path(csv_log_path)
        self.decision_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_log_path.exists():
            header = [
                "timestamp",
                "trade_id",
                "symbol",
                "direction",
                "regime",
                "reason",
                "claude_approve",
                "claude_confidence",
                "result",
                "pnl",
            ]
            with self.csv_log_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=header)
                writer.writeheader()

    def log_decision(self, payload: dict[str, Any]) -> None:
        with self.decision_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def log_trade(self, row: dict[str, Any]) -> None:
        header = [
            "timestamp",
            "trade_id",
            "symbol",
            "direction",
            "regime",
            "reason",
            "claude_approve",
            "claude_confidence",
            "result",
            "pnl",
        ]
        write_header = not self.csv_log_path.exists()
        with self.csv_log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": row.get("timestamp"),
                    "trade_id": row.get("trade_id"),
                    "symbol": row.get("symbol", "XAUUSD"),
                    "direction": row.get("direction"),
                    "regime": row.get("regime"),
                    "reason": row.get("reason"),
                    "claude_approve": row.get("claude_approve"),
                    "claude_confidence": row.get("claude_confidence"),
                    "result": row.get("result"),
                    "pnl": row.get("pnl"),
                }
            )
