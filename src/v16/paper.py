from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.project_config import V16_PAPER_TRADE_STATE_PATH
from src.v13.daps import maximum_lot_for_leverage
from src.v16.confidence_tier import ConfidenceTier
from src.v16.sel import sel_lot_size


SYMBOL_SPECS = {
    "XAUUSD": {"contract_size": 100.0, "pip_size": 0.1, "pip_value_per_lot": 10.0},
    "EURUSD": {"contract_size": 100000.0, "pip_size": 0.0001, "pip_value_per_lot": 10.0},
    "BTCUSD": {"contract_size": 1.0, "pip_size": 1.0, "pip_value_per_lot": 1.0},
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _symbol_spec(symbol: str) -> dict[str, float]:
    return dict(SYMBOL_SPECS.get(str(symbol).upper(), SYMBOL_SPECS["XAUUSD"]))


@dataclass(frozen=True)
class PaperSnapshot:
    balance: float
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    total_trades: int
    open_positions: int
    win_rate: float | None


class PaperTradingEngine:
    def __init__(self, path: Path = V16_PAPER_TRADE_STATE_PATH, starting_balance: float = 1000.0) -> None:
        self.path = Path(path)
        self.starting_balance = float(starting_balance)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write_state(self._default_state())

    def _default_state(self) -> dict[str, Any]:
        return {
            "version": "v16_paper_1",
            "starting_balance": self.starting_balance,
            "cash_balance": self.starting_balance,
            "realized_pnl": 0.0,
            "closed_trades": [],
            "open_positions": [],
            "updated_at": _utc_now(),
        }

    def _read_state(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._default_state()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            payload = self._default_state()
        payload.setdefault("closed_trades", [])
        payload.setdefault("open_positions", [])
        payload.setdefault("cash_balance", self.starting_balance)
        payload.setdefault("realized_pnl", 0.0)
        return payload

    def _write_state(self, payload: dict[str, Any]) -> None:
        payload["updated_at"] = _utc_now()
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def reset(self, starting_balance: float | None = None) -> dict[str, Any]:
        if starting_balance is not None:
            self.starting_balance = float(starting_balance)
        payload = self._default_state()
        self._write_state(payload)
        return self.state()

    def _position_unrealized(self, position: dict[str, Any], current_price: float) -> tuple[float, float]:
        direction_sign = 1.0 if str(position.get("direction", "BUY")).upper() == "BUY" else -1.0
        spec = _symbol_spec(str(position.get("symbol", "XAUUSD")))
        entry_price = _safe_float(position.get("entry_price"), current_price)
        pip_size = spec["pip_size"]
        pip_value_per_lot = spec["pip_value_per_lot"]
        lot = _safe_float(position.get("lot"), 0.0)
        pnl_pips = ((float(current_price) - entry_price) * direction_sign) / max(pip_size, 1e-9)
        pnl_usd = pnl_pips * pip_value_per_lot * lot
        return float(pnl_pips), float(pnl_usd)

    def state(self, current_prices: dict[str, float] | None = None) -> dict[str, Any]:
        payload = self._read_state()
        current_prices = {str(key).upper(): _safe_float(value) for key, value in (current_prices or {}).items()}
        open_positions = []
        unrealized_pnl = 0.0
        for position in payload.get("open_positions", []):
            symbol = str(position.get("symbol", "XAUUSD")).upper()
            current_price = current_prices.get(symbol, _safe_float(position.get("entry_price"), 0.0))
            pnl_pips, pnl_usd = self._position_unrealized(position, current_price)
            unrealized_pnl += pnl_usd
            open_positions.append(
                position
                | {
                    "current_price": round(current_price, 5),
                    "unrealized_pnl_pips": round(pnl_pips, 2),
                    "unrealized_pnl_usd": round(pnl_usd, 2),
                }
            )
        closed_trades = list(payload.get("closed_trades", []))
        wins = sum(1 for trade in closed_trades if _safe_float(trade.get("pnl_usd"), 0.0) > 0.0)
        snapshot = PaperSnapshot(
            balance=float(payload.get("cash_balance", self.starting_balance)),
            equity=float(payload.get("cash_balance", self.starting_balance)) + float(unrealized_pnl),
            realized_pnl=float(payload.get("realized_pnl", 0.0)),
            unrealized_pnl=float(unrealized_pnl),
            total_trades=len(closed_trades),
            open_positions=len(open_positions),
            win_rate=(wins / len(closed_trades)) if closed_trades else None,
        )
        return {
            "summary": {
                "balance": round(snapshot.balance, 2),
                "equity": round(snapshot.equity, 2),
                "realized_pnl": round(snapshot.realized_pnl, 2),
                "unrealized_pnl": round(snapshot.unrealized_pnl, 2),
                "total_trades": snapshot.total_trades,
                "open_positions": snapshot.open_positions,
                "win_rate": None if snapshot.win_rate is None else round(snapshot.win_rate, 4),
            },
            "open_positions": open_positions,
            "closed_trades": closed_trades[-100:],
            "updated_at": payload.get("updated_at"),
        }

    def open_position(
        self,
        *,
        symbol: str,
        direction: str,
        entry_price: float,
        confidence_tier: str,
        sqt_label: str,
        mode: str,
        leverage: float = 200.0,
        stop_pips: float = 20.0,
        take_profit_pips: float = 30.0,
        note: str = "",
    ) -> dict[str, Any]:
        payload = self._read_state()
        state = self.state()
        equity = _safe_float(state["summary"].get("equity"), self.starting_balance)
        symbol_upper = str(symbol).upper()
        spec = _symbol_spec(symbol_upper)
        tier = ConfidenceTier(str(confidence_tier).lower())
        raw_lot = sel_lot_size(
            equity=equity,
            confidence_tier=tier,
            sqt_label=sqt_label,
            mode=mode,
            stop_pips=stop_pips,
            pip_value_per_lot=spec["pip_value_per_lot"],
        )
        leverage_cap = maximum_lot_for_leverage(
            current_equity=equity,
            max_account_leverage=float(leverage),
            price_per_ounce=float(entry_price),
            contract_size_oz=spec["contract_size"],
        )
        lot = float(min(raw_lot, leverage_cap)) if leverage_cap is not None else float(raw_lot)
        if lot <= 0.0:
            raise ValueError("Leverage cap and risk sizing produced a zero lot.")
        direction_upper = str(direction).upper()
        direction_sign = 1.0 if direction_upper == "BUY" else -1.0
        notional_usd = lot * spec["contract_size"] * float(entry_price)
        margin_used = notional_usd / max(float(leverage), 1.0)
        pip_size = spec["pip_size"]
        stop_distance = stop_pips * pip_size
        take_distance = take_profit_pips * pip_size
        trade_id = uuid.uuid4().hex[:12]
        position = {
            "trade_id": trade_id,
            "symbol": symbol_upper,
            "direction": direction_upper,
            "mode": str(mode).lower(),
            "confidence_tier": tier.value,
            "sqt_label": str(sqt_label).upper(),
            "entry_price": round(float(entry_price), 5),
            "entry_time": _utc_now(),
            "lot": round(lot, 2),
            "leverage": float(leverage),
            "stop_pips": float(stop_pips),
            "take_profit_pips": float(take_profit_pips),
            "stop_price": round(float(entry_price) - (direction_sign * stop_distance), 5),
            "take_profit_price": round(float(entry_price) + (direction_sign * take_distance), 5),
            "notional_usd": round(notional_usd, 2),
            "margin_used": round(margin_used, 2),
            "note": str(note),
        }
        payload["open_positions"].append(position)
        self._write_state(payload)
        return position

    def close_position(self, trade_id: str, *, exit_price: float) -> dict[str, Any]:
        payload = self._read_state()
        open_positions = list(payload.get("open_positions", []))
        match = None
        remaining = []
        for position in open_positions:
            if str(position.get("trade_id")) == str(trade_id):
                match = position
            else:
                remaining.append(position)
        if match is None:
            raise KeyError(f"Unknown paper trade id: {trade_id}")
        pnl_pips, pnl_usd = self._position_unrealized(match, float(exit_price))
        closed = match | {
            "exit_price": round(float(exit_price), 5),
            "exit_time": _utc_now(),
            "pnl_pips": round(pnl_pips, 2),
            "pnl_usd": round(pnl_usd, 2),
            "status": "closed",
        }
        payload["open_positions"] = remaining
        payload["closed_trades"] = (payload.get("closed_trades", []) + [closed])[-500:]
        payload["cash_balance"] = round(_safe_float(payload.get("cash_balance"), self.starting_balance) + pnl_usd, 2)
        payload["realized_pnl"] = round(_safe_float(payload.get("realized_pnl"), 0.0) + pnl_usd, 2)
        self._write_state(payload)
        return closed
