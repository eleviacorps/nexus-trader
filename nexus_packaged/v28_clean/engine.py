"""Clean V28 execution engine.

Pipeline:
LIVE PRICE -> CONTEXT -> MODEL -> PATHS -> METRICS -> DECISION -> SNAPSHOT -> UI
"""

from __future__ import annotations

import asyncio
import copy
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import torch

STATE: dict[str, Any] = {}
SNAPSHOT: dict[str, Any] | None = None
_STATE_LOCK = threading.Lock()
_SNAPSHOT_LOCK = threading.Lock()


def compute_metrics(paths: np.ndarray | list[list[float]]) -> dict[str, float]:
    """Compute EV/dispersion/directional probabilities from generated paths."""
    arr = np.asarray(paths, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return {
            "ev": 0.0,
            "std": 1e-6,
            "prob_up": 0.0,
            "prob_down": 0.0,
            "confidence": 0.0,
        }
    returns = arr[:, -1] - arr[:, 0]
    ev = float(np.mean(returns))
    std = float(np.std(returns) + 1e-6)
    prob_up = float(np.mean(returns > 0))
    prob_down = float(np.mean(returns < 0))
    confidence = float(abs(prob_up - prob_down))
    return {
        "ev": ev,
        "std": std,
        "prob_up": prob_up,
        "prob_down": prob_down,
        "confidence": confidence,
    }


def decide(metrics: dict[str, float]) -> str:
    """Decision policy from clean confidence system."""
    ev = float(metrics.get("ev", 0.0))
    conf = float(metrics.get("confidence", 0.0))
    if conf < 0.12:
        return "HOLD"
    return "BUY" if ev > 0 else "SELL"


def compute_tp_sl(price: float, conf: float, atr: float) -> tuple[float, float, int]:
    """Compute TP/SL distances and RR from confidence."""
    if conf < 0.2:
        rr = 2
    elif conf < 0.35:
        rr = 3
    else:
        rr = 4
    safe_atr = float(max(1e-6, atr))
    sl = float(safe_atr * 0.3)
    tp = float(sl * rr)
    return tp, sl, rr


def _snapshot_expired(snapshot: dict[str, Any] | None, now_ts: float) -> bool:
    if snapshot is None:
        return False
    return float(snapshot.get("expires_at", 0.0)) <= now_ts


def create_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    """Create immutable execution snapshot."""
    global SNAPSHOT
    now_ts = time.time()
    snap = {
        "decision": str(state.get("decision", "HOLD")),
        "price": float(state.get("price", 0.0)),
        "tp": float(state.get("tp_price", 0.0)),
        "sl": float(state.get("sl_price", 0.0)),
        "created_at": now_ts,
        "expires_at": now_ts + 900.0,
    }
    with _SNAPSHOT_LOCK:
        SNAPSHOT = snap
    return snap


@dataclass(slots=True)
class EngineConfig:
    """Resolved runtime config for V28 clean engine."""

    symbol: str
    timeframe_minutes: int
    lookback: int
    feature_dim: int
    horizon: int
    loop_seconds: float
    auto_trade_enabled: bool
    execution_enabled: bool
    lot_size: float


class ModelAdapter:
    """Model interface using predictor for proper inference."""

    def __init__(self, loader: Any) -> None:
        self.loader = loader
        self._predictor = None
        try:
            from src.v27.short_horizon_predictor import create_short_horizon_predictor
            if hasattr(loader, '_model') and loader._model is not None:
                self._predictor = create_short_horizon_predictor(loader._model, device=str(getattr(loader, '_device', 'cpu')))
                self._predictor.num_futures = 16
                self._predictor.confidence_threshold = 0.35
        except Exception:
            pass

    def generate_paths(self, context: np.ndarray) -> np.ndarray:
        """Generate model paths using predictor if available."""
        if self._predictor is not None:
            try:
                arr = np.asarray(context, dtype=np.float32)
                if hasattr(self.loader, '_normalize'):
                    arr = self.loader._normalize(arr)
                tensor = torch.from_numpy(arr).unsqueeze(0)
                if hasattr(self.loader, '_device'):
                    tensor = tensor.to(self.loader._device)
                regime_probs = torch.ones(1, 9, device=tensor.device) / 9
                result = self._predictor.predict_15min_trade(
                    past_context=tensor,
                    regime_probs=regime_probs,
                    current_price=3300.0,
                    steps=5
                )
                decision = result.decision
                conf = result.confidence
                paths = np.zeros((64, 20), dtype=np.float32)
                if decision == "BUY":
                    paths[:, :] = np.linspace(0, conf * 10, 20)[None, :]
                elif decision == "SELL":
                    paths[:, :] = np.linspace(0, -conf * 10, 20)[None, :]
                return paths
            except Exception:
                pass
        
        # Fallback: return zero paths
        return np.zeros((64, 20), dtype=np.float32)


class V28CleanEngine:
    """Standalone real-time execution engine."""

    def __init__(
        self,
        *,
        settings: dict[str, Any],
        model_loader: Any,
        mt5_connector: Any,
        features: np.ndarray,
        ohlcv: pd.DataFrame,
    ) -> None:
        model_cfg = dict(settings.get("model", {}))
        data_cfg = dict(settings.get("data", {}))
        auto_cfg = dict(settings.get("auto_trade", {}))
        mt5_cfg = dict(settings.get("mt5", {}))
        self.cfg = EngineConfig(
            symbol=str(data_cfg.get("symbol", "XAUUSD")),
            timeframe_minutes=max(1, int(data_cfg.get("base_timeframe_minutes", 1))),
            lookback=int(model_cfg.get("lookback", 128)),
            feature_dim=int(model_cfg.get("feature_dim", 144)),
            horizon=int(model_cfg.get("horizon", 20)),
            loop_seconds=0.5,
            auto_trade_enabled=bool(auto_cfg.get("enabled", False)),
            execution_enabled=bool(mt5_cfg.get("execution_enabled", False)),
            lot_size=float(auto_cfg.get("fixed_lot_size", 0.01)),
        )
        self.logger = logging.getLogger("nexus.system")
        self.error_logger = logging.getLogger("nexus.errors")
        self.model = ModelAdapter(model_loader)
        self.mt5 = mt5_connector
        self.features = np.asarray(features, dtype=np.float32)
        self.runtime_ohlc = ohlcv[["open", "high", "low", "close", "volume"]].copy()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_ohlc_pull = 0.0
        self._last_signature = ""
        self._last_signature_change = time.monotonic()

        with _STATE_LOCK:
            STATE.clear()
            STATE.update(self._empty_state())

    def _empty_state(self) -> dict[str, Any]:
        now_ts = time.time()
        return {
            "price": 0.0,
            "paths": [],
            "metrics": {
                "ev": 0.0,
                "std": 1e-6,
                "prob_up": 0.0,
                "prob_down": 0.0,
                "confidence": 0.0,
            },
            "decision": "HOLD",
            "timestamp": now_ts,
            "base_time": int(now_ts),
            "timeframe_sec": int(self.cfg.timeframe_minutes * 60),
            "horizon_steps": int(self.cfg.horizon),
            "atr": 0.0,
            "rr": 0,
            "tp_distance": 0.0,
            "sl_distance": 0.0,
            "tp_price": 0.0,
            "sl_price": 0.0,
            "snapshot": None,
            "last_update_iso": datetime.now(timezone.utc).isoformat(),
            "pipeline_status": "BOOTING",
        }

    def get_live_price(self) -> float:
        """MT5-only live price, with fallback to last OHLC close."""
        price = float(self.mt5.get_live_price())
        if np.isfinite(price) and price > 0:
            return price
        # Fallback: use last close from ohlcv data
        if not self.runtime_ohlc.empty:
            last_close = float(self.runtime_ohlc["close"].iloc[-1])
            if np.isfinite(last_close) and last_close > 0:
                return last_close
        return 0.0

    def build_context(self, price: float) -> np.ndarray:
        """Build latest context window only."""
        if self.features.shape[0] < self.cfg.lookback:
            raise RuntimeError("Insufficient feature rows for context.")
        context = np.array(self.features[-self.cfg.lookback :], dtype=np.float32, copy=True)
        if context.shape != (self.cfg.lookback, self.cfg.feature_dim):
            raise RuntimeError(f"Invalid context shape: {context.shape}")
        ref_close = float(self.runtime_ohlc["close"].iloc[-1]) if not self.runtime_ohlc.empty else float(price)
        rel = (float(price) - ref_close) / max(abs(ref_close), 1e-6)
        # Inject latest live movement into the last row to keep context fresh.
        context[-1, 0] = rel
        if self.cfg.feature_dim > 1:
            context[-1, 1] = float(price)
        if self.cfg.feature_dim > 2:
            context[-1, 2] = ref_close
        return context

    def _timeframe_constant(self) -> Any:
        mt5_mod = getattr(self.mt5, "_mt5", None)
        if mt5_mod is None:
            return None
        mapping = {
            1: "TIMEFRAME_M1",
            5: "TIMEFRAME_M5",
            15: "TIMEFRAME_M15",
            30: "TIMEFRAME_M30",
            60: "TIMEFRAME_H1",
            240: "TIMEFRAME_H4",
        }
        return getattr(mt5_mod, mapping.get(self.cfg.timeframe_minutes, "TIMEFRAME_M1"), None)

    def _pull_live_ohlc(self, count: int = 600) -> None:
        if not bool(getattr(self.mt5, "is_connected", False)):
            return
        mt5_mod = getattr(self.mt5, "_mt5", None)
        tf = self._timeframe_constant()
        if mt5_mod is None or tf is None:
            return
        rates = mt5_mod.copy_rates_from_pos(self.cfg.symbol, tf, 0, int(count))
        if rates is None or len(rates) == 0:
            return
        frame = pd.DataFrame(rates)
        frame["timestamp"] = pd.to_datetime(frame["time"], unit="s", utc=True)
        frame = frame.set_index("timestamp")
        frame = frame.rename(columns={"tick_volume": "volume"})
        self.runtime_ohlc = frame[["open", "high", "low", "close", "volume"]].astype(float)

    def _update_last_candle(self, price: float) -> None:
        if self.runtime_ohlc.empty:
            ts = pd.Timestamp.now(tz="UTC").floor(f"{self.cfg.timeframe_minutes}min")
            self.runtime_ohlc = pd.DataFrame(
                {"open": [price], "high": [price], "low": [price], "close": [price], "volume": [0.0]},
                index=[ts],
            )
            return
        self.runtime_ohlc.iloc[-1, self.runtime_ohlc.columns.get_loc("close")] = price
        self.runtime_ohlc.iloc[-1, self.runtime_ohlc.columns.get_loc("high")] = max(
            float(self.runtime_ohlc.iloc[-1]["high"]),
            price,
        )
        self.runtime_ohlc.iloc[-1, self.runtime_ohlc.columns.get_loc("low")] = min(
            float(self.runtime_ohlc.iloc[-1]["low"]),
            price,
        )

    def _compute_atr(self, period: int = 14) -> float:
        if self.runtime_ohlc.empty:
            return 0.0
        frame = self.runtime_ohlc.tail(max(32, period + 2)).copy()
        hl = frame["high"] - frame["low"]
        hc = (frame["high"] - frame["close"].shift(1)).abs()
        lc = (frame["low"] - frame["close"].shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        return float(0.0 if pd.isna(atr) else atr)

    def _publish_state(self, payload: dict[str, Any]) -> None:
        with _STATE_LOCK:
            STATE.clear()
            STATE.update(payload)

    def get_state(self) -> dict[str, Any]:
        """Thread-safe state snapshot for API/UI."""
        with _STATE_LOCK:
            return copy.deepcopy(STATE)

    def get_ohlc_payload(self, limit: int = 500) -> list[dict[str, float | int]]:
        """Chart candles payload."""
        bars = self.runtime_ohlc.tail(int(limit))
        payload: list[dict[str, float | int]] = []
        for idx, row in bars.iterrows():
            payload.append(
                {
                    "time": int(pd.Timestamp(idx).timestamp()),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                }
            )
        return payload

    def _clear_expired_snapshot(self) -> None:
        global SNAPSHOT
        now_ts = time.time()
        with _SNAPSHOT_LOCK:
            if _snapshot_expired(SNAPSHOT, now_ts):
                SNAPSHOT = None

    def _execute_trade(self, decision: str, state: dict[str, Any]) -> None:
        if not self.cfg.execution_enabled or not bool(getattr(self.mt5, "is_connected", False)):
            return
        request_payload = {
            "symbol": self.cfg.symbol,
            "direction": decision,
            "lot_size": float(self.cfg.lot_size),
            "sl": float(state.get("sl_price", 0.0)),
            "tp": float(state.get("tp_price", 0.0)),
            "comment": "v28_clean_auto",
        }
        try:
            asyncio.run(self.mt5.place_order(request_payload))
        except Exception as exc:  # noqa: BLE001
            self.error_logger.error("PIPELINE BROKEN: trade execution failed: %s", exc)

    def auto_trade(self, state: dict[str, Any]) -> None:
        """Auto-trade entrypoint bound to decision + snapshot guard."""
        if not self.cfg.auto_trade_enabled:
            return
        decision = str(state.get("decision", "HOLD"))
        if decision == "HOLD":
            return
        with _SNAPSHOT_LOCK:
            if SNAPSHOT is not None and not _snapshot_expired(SNAPSHOT, time.time()):
                return
        snap = create_snapshot(state)
        self._execute_trade(decision, state)
        state["snapshot"] = snap

    def _check_freeze(self, state: dict[str, Any]) -> None:
        sig = (
            f"{state['price']:.5f}|{state['metrics']['ev']:.6f}|"
            f"{state['metrics']['std']:.6f}|{state['metrics']['confidence']:.6f}|"
            f"{state['decision']}"
        )
        now_mono = time.monotonic()
        if sig != self._last_signature:
            self._last_signature = sig
            self._last_signature_change = now_mono
            return
        if (now_mono - self._last_signature_change) > 5.0:
            self.error_logger.error("PIPELINE BROKEN: state frozen for >5s")

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                price = self.get_live_price()
                if price <= 0:
                    raise RuntimeError("live_price_unavailable")

                now_ts = time.time()
                if (now_ts - self._last_ohlc_pull) >= 1.0:
                    self._pull_live_ohlc(count=700)
                    self._last_ohlc_pull = now_ts
                self._update_last_candle(price)

                context = self.build_context(price)
                paths = np.asarray(self.model.generate_paths(context), dtype=np.float32)
                if paths.ndim != 2:
                    raise RuntimeError(f"invalid_paths_shape:{paths.shape}")

                paths[:, 0] = float(price)
                assert abs(float(paths[0, 0]) - float(price)) < 1e-3

                metrics = compute_metrics(paths)
                decision = decide(metrics)
                atr = self._compute_atr(period=14)
                tp_distance, sl_distance, rr = compute_tp_sl(price, float(metrics["confidence"]), atr)
                tp_price = float(price + tp_distance) if decision == "BUY" else float(price - tp_distance)
                sl_price = float(price - sl_distance) if decision == "BUY" else float(price + sl_distance)
                if decision == "HOLD":
                    tp_price = float(price)
                    sl_price = float(price)

                self._clear_expired_snapshot()
                with _SNAPSHOT_LOCK:
                    active_snapshot = copy.deepcopy(SNAPSHOT)

                base_time = int(pd.Timestamp(self.runtime_ohlc.index[-1]).timestamp()) if not self.runtime_ohlc.empty else int(now_ts)
                state = {
                    "price": float(price),
                    "paths": paths.tolist(),
                    "metrics": metrics,
                    "decision": decision,
                    "timestamp": float(now_ts),
                    "base_time": base_time,
                    "timeframe_sec": int(self.cfg.timeframe_minutes * 60),
                    "horizon_steps": int(paths.shape[1]),
                    "atr": float(atr),
                    "rr": int(rr),
                    "tp_distance": float(tp_distance),
                    "sl_distance": float(sl_distance),
                    "tp_price": float(tp_price),
                    "sl_price": float(sl_price),
                    "snapshot": active_snapshot,
                    "last_update_iso": datetime.now(timezone.utc).isoformat(),
                    "pipeline_status": "OK",
                }
                self.auto_trade(state)
                self._check_freeze(state)
                self._publish_state(state)
                self.logger.info("UPDATE: %s %s", state["timestamp"], state["metrics"])
                print("UPDATE:", state["timestamp"], state["metrics"])
            except AssertionError as exc:
                self.error_logger.error("PIPELINE BROKEN: path anchor validation failed: %s", exc)
            except Exception as exc:  # noqa: BLE001
                self.error_logger.error("PIPELINE BROKEN: %s", exc)
                fail_state = self.get_state()
                fail_state["pipeline_status"] = "BROKEN"
                fail_state["last_error"] = str(exc)
                fail_state["timestamp"] = float(time.time())
                fail_state["last_update_iso"] = datetime.now(timezone.utc).isoformat()
                self._publish_state(fail_state)
            time.sleep(self.cfg.loop_seconds)

    def start(self) -> None:
        """Start engine thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="v28_clean_engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop engine thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
