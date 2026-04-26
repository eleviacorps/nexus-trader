"""Native backtest engine for packaged runtime."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nexus_packaged.core.diffusion_loader import BaseModelLoader
from nexus_packaged.core.regime_detector import derive_signal_from_paths
from nexus_packaged.trading.lot_calculator import LotCalculator


@dataclass
class BacktestConfig:
    """Backtest controls and risk/trade parameters."""

    start_date: str
    end_date: str
    timeframe_minutes: int
    mode: str = "selective"
    confidence_threshold: float = 0.60
    frequency_every_n_cycles: int = 1
    interval_minutes: int = 15
    max_trade_count: int = 0
    min_trades_for_valid_result: int = 30
    lot_mode: str = "fixed"
    fixed_lot_size: float = 0.01
    lot_min: float = 0.01
    lot_max: float = 0.10
    lot_range_mode: str = "confidence"
    risk_pct_per_trade: float = 1.0
    kelly_fraction: float = 0.25
    leverage: int = 200
    contract_size: float = 100.0
    pip_value_per_lot: float = 1.0
    risk_reward: float = 2.0
    sl_atr_multiplier: float = 1.5
    trade_expiry_bars: int = 10
    allowed_directions: list = field(default_factory=lambda: ["BUY", "SELL"])
    initial_equity: float = 10000.0
    commission_per_lot_round_trip: float = 7.0
    spread_pips: float = 0.3
    max_open_trades: int = 3
    max_daily_trades: int = 0
    max_daily_loss_pct: float = 0.0
    max_drawdown_pct: float = 0.0


@dataclass
class TradeRecord:
    """Backtest trade record."""

    trade_id: str
    entry_time: datetime
    exit_time: datetime
    direction: str
    lot_size: float
    entry_price: float
    exit_price: float
    sl: float
    tp: float
    pnl_usd: float
    pnl_pips: float
    exit_reason: str
    confidence: float
    margin_used: float
    commission: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict representation."""
        return {
            "trade_id": self.trade_id,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "direction": self.direction,
            "lot_size": self.lot_size,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "sl": self.sl,
            "tp": self.tp,
            "pnl_usd": self.pnl_usd,
            "pnl_pips": self.pnl_pips,
            "exit_reason": self.exit_reason,
            "confidence": self.confidence,
            "margin_used": self.margin_used,
            "commission": self.commission,
            "source": self.source,
        }


@dataclass
class BacktestResult:
    """Backtest outputs."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    expectancy_usd: float
    expectancy_pips: float
    max_drawdown_pct: float
    max_drawdown_usd: float
    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float
    total_return_pct: float
    total_return_usd: float
    avg_trade_duration_bars: int
    avg_lot_size: float
    total_commission_paid: float
    margin_call_skips: int
    equity_curve: np.ndarray
    drawdown_curve: np.ndarray
    trade_log: list[TradeRecord]
    config: BacktestConfig


class BacktestEngine:
    """Walk-forward backtest with margin/leverage simulation."""

    def __init__(
        self,
        config: BacktestConfig,
        features: np.ndarray,
        ohlcv: pd.DataFrame,
        model_loader: BaseModelLoader | None = None,
    ) -> None:
        self.config = config
        self.features = np.asarray(features, dtype=np.float32)
        self.ohlcv = ohlcv.copy()
        self.model_loader = model_loader
        self.logger = logging.getLogger("nexus.trades")
        self.error_logger = logging.getLogger("nexus.errors")
        self.lot_calculator = LotCalculator()
        self._json_log_path = Path("nexus_packaged/logs/trades.log")
        self._json_log_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
        hl = frame["high"] - frame["low"]
        hc = (frame["high"] - frame["close"].shift(1)).abs()
        lc = (frame["low"] - frame["close"].shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _resample(self, frame: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
        if timeframe_minutes <= 1:
            return frame
        rule = f"{int(timeframe_minutes)}min"
        sampled = frame.resample(rule).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        return sampled.dropna()

    def _signal_from_features(self, window: np.ndarray) -> tuple[str, float, np.ndarray]:
        """Generate signal from model or heuristic fallback."""
        num_paths = 64
        horizon = 20
        if self.model_loader is not None and self.model_loader.is_loaded:
            paths = self.model_loader.predict(window)
            signal, confidence, _ = derive_signal_from_paths(paths, confidence_threshold=0.0)
            return signal, float(confidence), paths

        # Heuristic fallback from momentum in real features.
        momentum = float(np.nanmean(window[-5:, 0]))
        confidence = float(np.clip(abs(momentum) * 100.0, 0.0, 1.0))
        signal = "BUY" if momentum > 0 else "SELL"
        if abs(momentum) < 1e-6:
            signal = "HOLD"
        drift = np.linspace(0.0, momentum * 5.0, horizon, dtype=np.float32)
        paths = np.tile(drift[None, :], (num_paths, 1))
        noise = np.random.normal(loc=0.0, scale=max(1e-4, abs(momentum)), size=paths.shape).astype(np.float32)
        return signal, confidence, paths + noise

    def _mode_allows(self, cycle_index: int, timestamp: pd.Timestamp, confidence: float, trade_count: int, last_trade_ts: pd.Timestamp | None) -> bool:
        cfg = self.config
        mode = cfg.mode.lower()
        if mode == "forced":
            return True
        if mode == "selective":
            return confidence >= float(cfg.confidence_threshold)
        if mode == "frequency":
            n = max(1, int(cfg.frequency_every_n_cycles))
            return cycle_index % n == 0
        if mode == "interval":
            interval = pd.Timedelta(minutes=max(1, int(cfg.interval_minutes)))
            if last_trade_ts is None:
                return True
            return (timestamp - last_trade_ts) >= interval
        if mode == "count":
            if cfg.max_trade_count <= 0:
                return True
            return trade_count < int(cfg.max_trade_count)
        return False

    def _log_trade_event(self, payload: dict[str, Any]) -> None:
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()
        encoded = json.dumps(payload)
        self.logger.info(encoded)
        try:
            with self._json_log_path.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
        except Exception:  # noqa: BLE001
            self.error_logger.exception("Failed to append trade JSON log line.")

    def run(self) -> BacktestResult:
        """Execute backtest synchronously."""
        frame = self.ohlcv.copy()
        if not isinstance(frame.index, pd.DatetimeIndex):
            frame.index = pd.to_datetime(frame.index, utc=True)
        frame = frame.sort_index()
        frame = frame.loc[self.config.start_date : self.config.end_date]
        frame = self._resample(frame, self.config.timeframe_minutes)
        if frame.empty:
            raise ValueError("Backtest frame is empty for selected date range.")

        atr14 = self._atr(frame, 14).bfill().ffill()
        base_index = self.ohlcv.index
        idx_map = np.searchsorted(base_index.values, frame.index.values, side="left")

        equity = float(self.config.initial_equity)
        equity_peak = equity
        daily_start_equity: dict[datetime.date, float] = {}
        daily_trades: dict[datetime.date, int] = {}
        open_positions: list[dict[str, Any]] = []
        closed: list[TradeRecord] = []
        margin_call_skips = 0
        total_commission = 0.0
        cycle_idx = 0
        last_trade_ts: pd.Timestamp | None = None
        returns_for_ratio: list[float] = []

        equity_curve: list[float] = []
        drawdown_curve: list[float] = []

        for bar_i, (ts, row) in enumerate(frame.iterrows()):
            price = float(row["close"])
            high = float(row["high"])
            low = float(row["low"])
            current_day = ts.date()
            if current_day not in daily_start_equity:
                daily_start_equity[current_day] = equity
                daily_trades[current_day] = 0

            # Exit checks.
            still_open: list[dict[str, Any]] = []
            for pos in open_positions:
                age = bar_i - pos["entry_bar"]
                direction = pos["direction"]
                exit_reason = None
                exit_price = price
                if direction == "BUY":
                    if low <= pos["sl"]:
                        exit_reason = "SL"
                        exit_price = pos["sl"]
                    elif high >= pos["tp"]:
                        exit_reason = "TP"
                        exit_price = pos["tp"]
                else:
                    if high >= pos["sl"]:
                        exit_reason = "SL"
                        exit_price = pos["sl"]
                    elif low <= pos["tp"]:
                        exit_reason = "TP"
                        exit_price = pos["tp"]
                if exit_reason is None and age >= int(self.config.trade_expiry_bars):
                    exit_reason = "EXPIRY"
                    exit_price = price

                if exit_reason is None:
                    still_open.append(pos)
                    continue

                pip_size = 0.01
                sign = 1.0 if direction == "BUY" else -1.0
                pnl_pips = sign * (exit_price - pos["entry_price"]) / pip_size
                pnl_usd = pnl_pips * float(self.config.pip_value_per_lot) * float(pos["lot"])
                equity += pnl_usd
                returns_for_ratio.append(pnl_usd)
                record = TradeRecord(
                    trade_id=pos["trade_id"],
                    entry_time=pos["entry_time"].to_pydatetime(),
                    exit_time=ts.to_pydatetime(),
                    direction=direction,
                    lot_size=float(pos["lot"]),
                    entry_price=float(pos["entry_price"]),
                    exit_price=float(exit_price),
                    sl=float(pos["sl"]),
                    tp=float(pos["tp"]),
                    pnl_usd=float(pnl_usd),
                    pnl_pips=float(pnl_pips),
                    exit_reason=exit_reason,
                    confidence=float(pos["confidence"]),
                    margin_used=float(pos["margin_used"]),
                    commission=float(pos["commission"]),
                    source="backtest",
                )
                closed.append(record)
                self._log_trade_event(
                    {
                        "event": "CLOSE",
                        "trade_id": record.trade_id,
                        "source": "backtest",
                        "direction": record.direction,
                        "lot": record.lot_size,
                        "entry": record.entry_price,
                        "sl": record.sl,
                        "tp": record.tp,
                        "confidence": record.confidence,
                        "margin_used": record.margin_used,
                        "reason": record.exit_reason,
                    }
                )
            open_positions = still_open

            # Floating pnl for margin check.
            pip_size = 0.01
            floating = 0.0
            used_margin = 0.0
            for pos in open_positions:
                sign = 1.0 if pos["direction"] == "BUY" else -1.0
                floating += sign * (price - pos["entry_price"]) / pip_size * self.config.pip_value_per_lot * pos["lot"]
                used_margin += (pos["lot"] * self.config.contract_size * pos["entry_price"]) / max(1, self.config.leverage)
            free_margin = equity + floating - used_margin

            # Inference and possible entry.
            row_idx = int(idx_map[bar_i]) if bar_i < len(idx_map) else -1
            if row_idx >= self.features.shape[0]:
                row_idx = self.features.shape[0] - 1
            lookback = 128
            if row_idx > lookback:
                window = self.features[row_idx - lookback : row_idx]
                signal, confidence, _paths = self._signal_from_features(window)
            else:
                signal, confidence = "HOLD", 0.0

            cfg = self.config
            allowed, reason = True, ""
            if len(open_positions) >= int(cfg.max_open_trades):
                allowed, reason = False, "MAX_OPEN_TRADES"
            if cfg.max_daily_trades > 0 and daily_trades[current_day] >= int(cfg.max_daily_trades):
                allowed, reason = False, "MAX_DAILY_TRADES"
            day_pnl_pct = ((equity - daily_start_equity[current_day]) / max(1e-9, daily_start_equity[current_day])) * 100.0
            if cfg.max_daily_loss_pct > 0 and day_pnl_pct <= -abs(float(cfg.max_daily_loss_pct)):
                allowed, reason = False, "MAX_DAILY_LOSS"
            drawdown_pct = ((equity_peak - equity) / max(1e-9, equity_peak)) * 100.0
            if cfg.max_drawdown_pct > 0 and drawdown_pct >= abs(float(cfg.max_drawdown_pct)):
                allowed, reason = False, "MAX_DRAWDOWN"
            if signal not in set(str(x).upper() for x in cfg.allowed_directions):
                allowed, reason = False, "DIRECTION_FILTER"
            if not self._mode_allows(cycle_idx, ts, confidence, len(closed), last_trade_ts):
                allowed, reason = False, "MODE_FILTER"

            if allowed and signal in {"BUY", "SELL"}:
                sl_distance = max(1e-6, float(atr14.iloc[bar_i]) * float(cfg.sl_atr_multiplier))
                sl_pips = sl_distance / pip_size
                lot = self.lot_calculator.calculate(
                    mode=cfg.lot_mode,
                    config=cfg,
                    account_equity=equity,
                    sl_pips=sl_pips,
                    pip_value=float(cfg.pip_value_per_lot),
                    confidence=float(confidence),
                    win_rate_history=(len([t for t in closed if t.pnl_usd > 0]) / len(closed)) if closed else 0.5,
                    broker_lot_min=float(cfg.lot_min),
                    broker_lot_max=float(cfg.lot_max),
                    broker_lot_step=0.01,
                    risk_reward=float(cfg.risk_reward),
                )
                spread_price = float(cfg.spread_pips) * pip_size
                entry = price + spread_price if signal == "BUY" else price - spread_price
                sl = entry - sl_distance if signal == "BUY" else entry + sl_distance
                tp = entry + sl_distance * float(cfg.risk_reward) if signal == "BUY" else entry - sl_distance * float(cfg.risk_reward)
                required_margin = (lot * cfg.contract_size * entry) / max(1, cfg.leverage)
                if required_margin > free_margin:
                    margin_call_skips += 1
                    self._log_trade_event(
                        {
                            "event": "SKIP",
                            "trade_id": "",
                            "source": "backtest",
                            "direction": signal,
                            "lot": lot,
                            "entry": entry,
                            "sl": sl,
                            "tp": tp,
                            "confidence": confidence,
                            "margin_used": required_margin,
                            "reason": "MARGIN_CALL_SKIP",
                        }
                    )
                else:
                    commission = float(cfg.commission_per_lot_round_trip) * lot
                    equity -= commission
                    total_commission += commission
                    pos = {
                        "trade_id": str(uuid.uuid4()),
                        "entry_bar": bar_i,
                        "entry_time": ts,
                        "direction": signal,
                        "lot": lot,
                        "entry_price": entry,
                        "sl": sl,
                        "tp": tp,
                        "confidence": confidence,
                        "margin_used": required_margin,
                        "commission": commission,
                    }
                    open_positions.append(pos)
                    daily_trades[current_day] += 1
                    last_trade_ts = ts
                    self._log_trade_event(
                        {
                            "event": "OPEN",
                            "trade_id": pos["trade_id"],
                            "source": "backtest",
                            "direction": signal,
                            "lot": lot,
                            "entry": entry,
                            "sl": sl,
                            "tp": tp,
                            "confidence": confidence,
                            "margin_used": required_margin,
                            "reason": "OPEN",
                        }
                    )
            elif not allowed and reason:
                self._log_trade_event(
                    {
                        "event": "SKIP",
                        "trade_id": "",
                        "source": "backtest",
                        "direction": signal,
                        "lot": 0.0,
                        "entry": price,
                        "sl": 0.0,
                        "tp": 0.0,
                        "confidence": confidence,
                        "margin_used": used_margin,
                        "reason": reason,
                    }
                )

            equity_peak = max(equity_peak, equity)
            dd_usd = equity_peak - equity
            dd_pct = (dd_usd / max(1e-9, equity_peak)) * 100.0
            equity_curve.append(equity)
            drawdown_curve.append(dd_pct)
            cycle_idx += 1

            if self.config.max_trade_count > 0 and len(closed) >= int(self.config.max_trade_count):
                break

        # Force-close remaining open positions at last close.
        if len(frame) > 0:
            final_ts = frame.index[-1]
            final_close = float(frame.iloc[-1]["close"])
            for pos in open_positions:
                sign = 1.0 if pos["direction"] == "BUY" else -1.0
                pnl_pips = sign * (final_close - pos["entry_price"]) / 0.01
                pnl_usd = pnl_pips * self.config.pip_value_per_lot * pos["lot"]
                equity += pnl_usd
                returns_for_ratio.append(pnl_usd)
                closed.append(
                    TradeRecord(
                        trade_id=pos["trade_id"],
                        entry_time=pos["entry_time"].to_pydatetime(),
                        exit_time=final_ts.to_pydatetime(),
                        direction=pos["direction"],
                        lot_size=float(pos["lot"]),
                        entry_price=float(pos["entry_price"]),
                        exit_price=float(final_close),
                        sl=float(pos["sl"]),
                        tp=float(pos["tp"]),
                        pnl_usd=float(pnl_usd),
                        pnl_pips=float(pnl_pips),
                        exit_reason="MANUAL",
                        confidence=float(pos["confidence"]),
                        margin_used=float(pos["margin_used"]),
                        commission=float(pos["commission"]),
                        source="backtest",
                    )
                )

        total_trades = len(closed)
        wins = len([t for t in closed if t.pnl_usd > 0])
        losses = len([t for t in closed if t.pnl_usd <= 0])
        pnl_usd = np.array([t.pnl_usd for t in closed], dtype=np.float64) if closed else np.array([], dtype=np.float64)
        pnl_pips = np.array([t.pnl_pips for t in closed], dtype=np.float64) if closed else np.array([], dtype=np.float64)
        expectancy_usd = float(np.mean(pnl_usd)) if pnl_usd.size else 0.0
        expectancy_pips = float(np.mean(pnl_pips)) if pnl_pips.size else 0.0
        std_ret = float(np.std(pnl_usd)) if pnl_usd.size else 0.0
        downside = pnl_usd[pnl_usd < 0]
        std_downside = float(np.std(downside)) if downside.size else 0.0
        sharpe = expectancy_usd / (std_ret + 1e-12)
        sortino = expectancy_usd / (std_downside + 1e-12)
        gross_profit = float(np.sum(pnl_usd[pnl_usd > 0])) if pnl_usd.size else 0.0
        gross_loss = float(np.abs(np.sum(pnl_usd[pnl_usd < 0]))) if pnl_usd.size else 0.0
        profit_factor = gross_profit / (gross_loss + 1e-12)
        total_return_usd = float(equity - self.config.initial_equity)
        total_return_pct = (total_return_usd / max(1e-9, self.config.initial_equity)) * 100.0
        max_dd_pct = float(np.max(drawdown_curve)) if drawdown_curve else 0.0
        max_dd_usd = float(max_dd_pct / 100.0 * equity_peak)
        avg_duration = int(np.mean([(t.exit_time - t.entry_time).total_seconds() / 60 / max(1, self.config.timeframe_minutes) for t in closed])) if closed else 0
        avg_lot = float(np.mean([t.lot_size for t in closed])) if closed else 0.0

        return BacktestResult(
            total_trades=total_trades,
            winning_trades=wins,
            losing_trades=losses,
            win_rate=(wins / total_trades) if total_trades else 0.0,
            expectancy_usd=expectancy_usd,
            expectancy_pips=expectancy_pips,
            max_drawdown_pct=max_dd_pct / 100.0,
            max_drawdown_usd=max_dd_usd,
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            profit_factor=float(profit_factor),
            total_return_pct=float(total_return_pct),
            total_return_usd=float(total_return_usd),
            avg_trade_duration_bars=avg_duration,
            avg_lot_size=avg_lot,
            total_commission_paid=float(total_commission),
            margin_call_skips=int(margin_call_skips),
            equity_curve=np.asarray(equity_curve, dtype=np.float32),
            drawdown_curve=np.asarray(drawdown_curve, dtype=np.float32),
            trade_log=closed,
            config=self.config,
        )

    def run_async(self) -> asyncio.Task:
        """Run backtest in thread-backed asyncio task."""
        loop = asyncio.get_running_loop()
        return loop.create_task(asyncio.to_thread(self.run))
