from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.project_config import V13_PAPER_TRADE_LOG_PATH


@dataclass
class PaperTrade:
    timestamp: str
    symbol: str
    direction: str
    uts_score: float
    cabr_score: float
    regime: str
    entry_price: float
    exit_price: float | None
    entry_time: str
    exit_time: str
    status: str
    outcome: str | None = None
    pnl_pips: float | None = None
    skip_reason: str | None = None


class PaperTradeAccumulator:
    def __init__(self, path: Path = V13_PAPER_TRADE_LOG_PATH) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._pending: dict[str, dict[str, Any]] = {}

    def log_trade(
        self,
        *,
        symbol: str,
        direction: str,
        uts_score: float,
        cabr_score: float,
        regime: str,
        entry_price: float,
        entry_time: str,
        exit_time: str,
        trade_id: str | None = None,
    ) -> str:
        identifier = trade_id or f'{symbol}:{entry_time}:{direction}:{len(self._pending)}'
        trade = PaperTrade(
            timestamp=datetime.now(timezone.utc).isoformat(),
            symbol=symbol,
            direction=direction,
            uts_score=float(uts_score),
            cabr_score=float(cabr_score),
            regime=str(regime),
            entry_price=float(entry_price),
            exit_price=None,
            entry_time=str(entry_time),
            exit_time=str(exit_time),
            status='pending',
        )
        self._pending[identifier] = asdict(trade)
        self._append(self._pending[identifier])
        return identifier

    def complete_trade(self, trade_id: str, *, exit_price: float) -> dict[str, Any] | None:
        if trade_id not in self._pending:
            return None
        trade = dict(self._pending.pop(trade_id))
        entry = float(trade['entry_price'])
        direction = str(trade['direction']).upper()
        sign = 1.0 if direction == 'BUY' else -1.0
        pnl_pips = ((float(exit_price) - entry) * sign) / 0.1
        trade['exit_price'] = float(exit_price)
        trade['pnl_pips'] = float(pnl_pips)
        trade['outcome'] = 'win' if pnl_pips > 0.0 else 'loss'
        trade['status'] = 'complete'
        self._append(trade)
        return trade

    def log_completed_trade(
        self,
        *,
        symbol: str,
        direction: str,
        uts_score: float,
        cabr_score: float,
        regime: str,
        entry_price: float,
        exit_price: float,
        entry_time: str,
        exit_time: str,
        skip_reason: str | None = None,
    ) -> dict[str, Any]:
        sign = 1.0 if str(direction).upper() == 'BUY' else -1.0
        pnl_pips = ((float(exit_price) - float(entry_price)) * sign) / 0.1
        trade = PaperTrade(
            timestamp=datetime.now(timezone.utc).isoformat(),
            symbol=symbol,
            direction=str(direction).upper(),
            uts_score=float(uts_score),
            cabr_score=float(cabr_score),
            regime=str(regime),
            entry_price=float(entry_price),
            exit_price=float(exit_price),
            entry_time=str(entry_time),
            exit_time=str(exit_time),
            status='complete',
            outcome='win' if pnl_pips > 0.0 else 'loss',
            pnl_pips=float(pnl_pips),
            skip_reason=skip_reason,
        )
        payload = asdict(trade)
        self._append(payload)
        return payload

    def _append(self, payload: dict[str, Any]) -> None:
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(payload) + '\n')

    def load_records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
        return records

    def load_completed_trades(self) -> list[dict[str, Any]]:
        return [record for record in self.load_records() if record.get('status') == 'complete' and record.get('outcome') in {'win', 'loss'}]

    def summary(self) -> dict[str, Any]:
        trades = self.load_completed_trades()
        pending = [record for record in self.load_records() if record.get('status') == 'pending']
        if not trades:
            return {
                'count': 0,
                'win_rate': None,
                'avg_pnl_pips': None,
                'pending': len(pending),
                'ready_for_calibration': False,
                'ready_for_stage3_report': False,
            }
        wins = sum(1 for trade in trades if trade.get('outcome') == 'win')
        avg_pnl = sum(float(trade.get('pnl_pips') or 0.0) for trade in trades) / len(trades)
        return {
            'count': len(trades),
            'win_rate': wins / len(trades),
            'avg_pnl_pips': avg_pnl,
            'pending': len(pending),
            'ready_for_calibration': len(trades) >= 40,
            'ready_for_stage3_report': len(trades) >= 100,
        }
