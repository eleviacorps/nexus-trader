"""Asynchronous inference coordination."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd

from nexus_packaged.core.diffusion_loader import BaseModelLoader
from nexus_packaged.core.model_guard import get_inference_guard
from nexus_packaged.core.regime_detector import build_signal_snapshot
from nexus_packaged.v27_execution.execution_engine import ExecutionDecision, SnapshotExecutionEngine
from nexus_packaged.v27_execution.path_processing import prepare_chart_payload


@dataclass
class InferenceEvent:
    """Single inference output event."""

    timestamp: datetime
    bar_timestamp: datetime
    paths: np.ndarray
    median_path: np.ndarray
    band_10: np.ndarray
    band_90: np.ndarray
    signal: str
    confidence: float
    regime: str
    latency_ms: float
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "bar_timestamp": self.bar_timestamp.isoformat(),
            "num_paths": int(self.paths.shape[0]),
            "horizon_bars": int(self.paths.shape[1]),
            "paths": self.paths.tolist(),
            "median_path": self.median_path.tolist(),
            "confidence_band_10": self.band_10.tolist(),
            "confidence_band_90": self.band_90.tolist(),
            "signal": self.signal,
            "confidence": float(self.confidence),
            "regime": self.regime,
            "latency_ms": float(self.latency_ms),
            "meta": self.meta,
        }


class InferenceRunner:
    """Maintains rolling context and emits model predictions."""

    def __init__(
        self,
        *,
        model_loader: BaseModelLoader,
        features: np.ndarray,
        ohlcv: pd.DataFrame,
        settings: dict[str, Any],
        live_price_provider: Callable[[], float] | None = None,
    ) -> None:
        self._logger = logging.getLogger("nexus.system")
        self._model_loader = model_loader
        self._features = np.asarray(features, dtype=np.float32)
        self._ohlcv = ohlcv.copy()
        self._settings = settings
        self._live_price_provider = live_price_provider

        model_cfg = dict(settings.get("model", {}))
        self.lookback = int(model_cfg.get("lookback", 128))
        self.interval_seconds = max(1, int(model_cfg.get("inference_interval_seconds", 60)))
        self.horizon_steps = int(model_cfg.get("horizon", 20))
        self.timeframe_sec = max(60, int(settings.get("data", {}).get("base_timeframe_minutes", 1)) * 60)
        self._queue: asyncio.Queue[InferenceEvent] = asyncio.Queue(maxsize=512)
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._latest_event: InferenceEvent | None = None
        # Start from the most recent available context instead of replaying from
        # the beginning of history, so live UI/paths align with current candles.
        self._index = max(self.lookback, len(self._features) - 1)
        self._last_infer_ts: float = 0.0
        self._lock = asyncio.Lock()
        self._inference_logger = logging.getLogger("nexus.inference")
        execution_cfg = dict(settings.get("execution", {}))
        self.state_interval_seconds = max(0.2, float(execution_cfg.get("state_update_interval_seconds", 0.5)))
        self.state_stuck_after_seconds = max(2.0, float(execution_cfg.get("state_stuck_after_seconds", 5.0)))
        self._last_state_signature = ""
        self._last_state_change_monotonic = 0.0
        self._state_stuck_logged = False
        self._global_state: dict[str, Any] = self._initial_global_state()
        bar_seconds = self.timeframe_sec
        self._decision_engine = SnapshotExecutionEngine(
            ev_k=float(execution_cfg.get("ev_k", 0.02)),
            conf_min=float(execution_cfg.get("conf_min", 0.10)),
            min_tick=float(execution_cfg.get("min_tick", 0.01)),
            snapshot_bars=int(execution_cfg.get("snapshot_bars", 10)),
            bar_seconds=bar_seconds,
        )

    @property
    def event_queue(self) -> asyncio.Queue[InferenceEvent]:
        """Inference event queue consumed by TUI/API/trading."""
        return self._queue

    @property
    def latest_event(self) -> InferenceEvent | None:
        """Most recent inference event."""
        return self._latest_event

    @property
    def is_running(self) -> bool:
        """Runner loop state."""
        return self._running

    def current_price(self) -> float:
        """Single source of truth: live MT5 price with OHLC fallback."""
        if self._live_price_provider is None:
            return self._fallback_price_from_ohlc()
        try:
            price = float(self._live_price_provider())
            if np.isfinite(price) and price > 0:
                return price
        except Exception:  # noqa: BLE001
            pass
        return self._fallback_price_from_ohlc()

    def _fallback_price_from_ohlc(self) -> float:
        """Fallback to last known OHLC close price."""
        if not self._ohlcv.empty:
            return float(self._ohlcv.iloc[-1]["close"])
        return 0.0

    def _initial_global_state(self) -> dict[str, Any]:
        """Build the default state payload returned before first inference."""
        now = datetime.now(timezone.utc)
        base_time = int(now.timestamp())
        return {
            "timestamp": now.isoformat(),
            "timestamp_epoch_ms": int(now.timestamp() * 1000),
            "price": float(self.current_price()),
            "paths": [],
            "decision": "HOLD",
            "signal": "HOLD",
            "confidence": 0.0,
            "ev": 0.0,
            "std": 0.0,
            "skew": 0.0,
            "ev_threshold": 0.0,
            "regime": "UNKNOWN",
            "base_time": base_time,
            "timeframe_sec": int(self.timeframe_sec),
            "horizon_steps": int(self.horizon_steps),
            "live": {
                "signal": "HOLD",
                "confidence": 0.0,
                "ev": 0.0,
                "std": 0.0,
                "ev_threshold": 0.0,
                "regime": "UNKNOWN",
                "paths": [],
            },
            "snapshot": None,
            "current_signal_source": "live",
            "last_update_time": now.isoformat(),
            "last_update_unix": float(now.timestamp()),
        }

    def _update_global_state(self, event: InferenceEvent) -> None:
        """Update global state from one event and emit state diagnostics."""
        now = datetime.now(timezone.utc)
        # Get fresh live price, not from potentially stale event meta
        fresh_price = float(self.current_price())
        live_signal = event.meta.get("live_signal") if isinstance(event.meta.get("live_signal"), dict) else {}
        snapshot_signal = event.meta.get("snapshot_signal") if isinstance(event.meta.get("snapshot_signal"), dict) else None
        live_paths = event.meta.get("live_paths") if isinstance(event.meta.get("live_paths"), list) else event.paths.tolist()
        snapshot_paths = event.meta.get("snapshot_paths") if isinstance(event.meta.get("snapshot_paths"), list) else None
        current_signal = snapshot_signal if snapshot_signal else live_signal
        current_paths = snapshot_paths if snapshot_paths is not None else live_paths
        current_price = fresh_price  # Always use fresh live price
        self._logger.debug("STATE_UPDATE: fresh_price=%.5f, event.current_price=%.5f", fresh_price, event.meta.get("current_price", 0))
        base_time = int(event.meta.get("base_time", int(now.timestamp())))
        timeframe_sec = int(event.meta.get("timeframe_sec", self.timeframe_sec))
        horizon_steps = int(event.meta.get("horizon_steps", self.horizon_steps))
        state = {
            "timestamp": now.isoformat(),
            "timestamp_epoch_ms": int(now.timestamp() * 1000),
            "price": current_price,
            "paths": current_paths,
            "decision": str(current_signal.get("decision", event.signal)),
            "signal": str(current_signal.get("decision", event.signal)),
            "confidence": float(current_signal.get("confidence", event.confidence)),
            "ev": float(current_signal.get("ev", event.meta.get("ev", 0.0))),
            "std": float(current_signal.get("std", event.meta.get("std", 0.0))),
            "skew": float(current_signal.get("skew", event.meta.get("skew", 0.0))),
            "ev_threshold": float(current_signal.get("ev_threshold", event.meta.get("ev_threshold", 0.0))),
            "regime": str(current_signal.get("regime", event.regime)),
            "base_time": base_time,
            "timeframe_sec": timeframe_sec,
            "horizon_steps": horizon_steps,
            "live": {
                "signal": str(live_signal.get("decision", "HOLD")),
                "confidence": float(live_signal.get("confidence", 0.0)),
                "ev": float(live_signal.get("ev", 0.0)),
                "std": float(live_signal.get("std", 0.0)),
                "ev_threshold": float(live_signal.get("ev_threshold", 0.0)),
                "regime": str(live_signal.get("regime", "UNKNOWN")),
                "entry": float(live_signal.get("entry", current_price)),
                "paths": live_paths,
            },
            "snapshot": (
                {
                    "signal": str(snapshot_signal.get("decision", "HOLD")),
                    "confidence": float(snapshot_signal.get("confidence", 0.0)),
                    "ev": float(snapshot_signal.get("ev", 0.0)),
                    "std": float(snapshot_signal.get("std", 0.0)),
                    "ev_threshold": float(snapshot_signal.get("ev_threshold", 0.0)),
                    "regime": str(snapshot_signal.get("regime", "UNKNOWN")),
                    "entry": float(snapshot_signal.get("entry", 0.0)),
                    "snapshot_id": str(snapshot_signal.get("snapshot_id", "")),
                    "active": bool(snapshot_signal.get("snapshot_active", False)),
                    "expires_at": str(snapshot_signal.get("expires_at", "")),
                    "paths": snapshot_paths,
                }
                if snapshot_signal is not None
                else None
            ),
            "current_signal_source": str(event.meta.get("current_signal_source", "live")),
            "last_update_time": now.isoformat(),
            "last_update_unix": float(now.timestamp()),
        }
        self._global_state = state

        # Freeze detection for stale runtime values.
        signature = (
            f"{state['price']:.6f}|{state['ev']:.8f}|{state['std']:.8f}|"
            f"{state['confidence']:.8f}|{state['decision']}"
        )
        monotonic_now = asyncio.get_running_loop().time()
        if signature != self._last_state_signature:
            self._last_state_signature = signature
            self._last_state_change_monotonic = monotonic_now
            self._state_stuck_logged = False
        elif (
            self._last_state_change_monotonic > 0
            and (monotonic_now - self._last_state_change_monotonic) >= self.state_stuck_after_seconds
            and not self._state_stuck_logged
        ):
            self._state_stuck_logged = True
            self._logger.warning("STATE_STUCK: runtime state unchanged for > %.1fs", self.state_stuck_after_seconds)

        self._inference_logger.info(
            json.dumps(
                {
                    "timestamp": now.isoformat(),
                    "event": "STATE_UPDATED",
                    "price": float(state["price"]),
                    "ev": float(state["ev"]),
                    "std": float(state["std"]),
                    "confidence": float(state["confidence"]),
                    "decision": str(state["decision"]),
                }
            )
        )

    def get_global_state(self) -> dict[str, Any]:
        """Return latest runtime state object."""
        return dict(self._global_state)

    @staticmethod
    def _decision_payload(decision: ExecutionDecision | None) -> dict[str, Any] | None:
        if decision is None:
            return None
        return {
            "decision": str(decision.decision),
            "confidence": float(decision.confidence),
            "ev": float(decision.ev),
            "std": float(decision.std),
            "skew": float(decision.skew),
            "regime": str(decision.regime),
            "ev_threshold": float(decision.ev_threshold),
            "rr": int(decision.rr),
            "sl_distance": float(decision.sl_distance),
            "tp_distance": float(decision.tp_distance),
            "entry": float(decision.entry),
            "sl": float(decision.sl),
            "tp": float(decision.tp),
            "snapshot_id": str(decision.snapshot_id),
            "snapshot_active": bool(decision.snapshot_active),
            "created_at": str(decision.created_at),
            "expires_at": str(decision.expires_at),
            "hold_reason": str(decision.hold_reason),
        }

    @staticmethod
    def _payload_to_decision(payload: dict[str, Any]) -> ExecutionDecision:
        return ExecutionDecision(
            decision=str(payload.get("decision", "HOLD")),
            confidence=float(payload.get("confidence", 0.0)),
            ev=float(payload.get("ev", 0.0)),
            std=float(payload.get("std", 0.0)),
            skew=float(payload.get("skew", 0.0)),
            regime=str(payload.get("regime", "UNKNOWN")),
            ev_threshold=float(payload.get("ev_threshold", 0.0)),
            rr=int(payload.get("rr", 0)),
            sl_distance=float(payload.get("sl_distance", 0.0)),
            tp_distance=float(payload.get("tp_distance", 0.0)),
            entry=float(payload.get("entry", 0.0)),
            sl=float(payload.get("sl", 0.0)),
            tp=float(payload.get("tp", 0.0)),
            snapshot_id=str(payload.get("snapshot_id", "")),
            snapshot_active=bool(payload.get("snapshot_active", False)),
            created_at=str(payload.get("created_at", "")),
            expires_at=str(payload.get("expires_at", "")),
            hold_reason=str(payload.get("hold_reason", "")),
        )

    def create_snapshot_from_event(self, event: InferenceEvent) -> dict[str, Any] | None:
        """Create snapshot from live decision for trade execution lock."""
        live_payload = event.meta.get("live_signal")
        if not isinstance(live_payload, dict):
            return None
        live_decision = self._payload_to_decision(live_payload)
        if live_decision.decision not in {"BUY", "SELL"}:
            return None
        snap = self._decision_engine.create_snapshot(
            live_decision=live_decision,
            paths=np.asarray(event.meta.get("live_paths") or event.paths.tolist(), dtype=np.float32),
            bar_timestamp=event.bar_timestamp,
        )
        snap_payload = self._decision_payload(snap)
        if snap_payload is not None:
            snap_paths = self._decision_engine.get_active_snapshot_paths(bar_timestamp=event.bar_timestamp)
            event.meta["snapshot_signal"] = snap_payload
            event.meta["snapshot_paths"] = snap_paths.tolist() if snap_paths is not None else None
            event.meta["current_signal_source"] = "snapshot"
            event.signal = str(snap_payload.get("decision", event.signal))
            event.confidence = float(snap_payload.get("confidence", event.confidence))
            event.regime = str(snap_payload.get("regime", event.regime))
            self._latest_event = event
            self._update_global_state(event)
        return snap_payload

    def reload_state(self) -> None:
        """Reset runtime state for reload endpoint."""
        self._decision_engine.reset_state()
        self._latest_event = None
        self._index = max(self.lookback, len(self._features) - 1)
        self._last_state_signature = ""
        self._last_state_change_monotonic = 0.0
        self._state_stuck_logged = False
        self._global_state = self._initial_global_state()

    def _sync_runtime_ohlc_with_live_price(self, *, current_price: float, now: datetime) -> None:
        """Keep runtime OHLC aligned with live MT5 ticks for ATR/chart scaling."""
        if self._ohlcv.empty or current_price <= 0:
            return
        ts_now = pd.Timestamp(now).tz_convert("UTC") if pd.Timestamp(now).tzinfo else pd.Timestamp(now, tz="UTC")
        last_idx = pd.Timestamp(self._ohlcv.index[-1])
        last_idx = last_idx.tz_convert("UTC") if last_idx.tzinfo else last_idx.tz_localize("UTC")

        if (ts_now - last_idx).total_seconds() >= self.timeframe_sec:
            prev_close = float(self._ohlcv.iloc[-1]["close"])
            new_row = pd.DataFrame(
                {
                    "open": [prev_close],
                    "high": [max(prev_close, current_price)],
                    "low": [min(prev_close, current_price)],
                    "close": [current_price],
                    "volume": [0.0],
                },
                index=[ts_now.floor(f"{max(1, int(self.timeframe_sec // 60))}min")],
            )
            self._ohlcv = pd.concat([self._ohlcv, new_row]).sort_index().tail(10000)
            return

        self._ohlcv.iloc[-1, self._ohlcv.columns.get_loc("close")] = current_price
        self._ohlcv.iloc[-1, self._ohlcv.columns.get_loc("high")] = max(
            float(self._ohlcv.iloc[-1]["high"]),
            current_price,
        )
        self._ohlcv.iloc[-1, self._ohlcv.columns.get_loc("low")] = min(
            float(self._ohlcv.iloc[-1]["low"]),
            current_price,
        )

    async def append_bar(self, bar: dict[str, Any]) -> None:
        """Append a new bar to runtime buffers.

        This method is intended for MT5 live-bar integration.
        """
        async with self._lock:
            ts = pd.to_datetime(bar.get("timestamp"), utc=True, errors="coerce")
            if pd.isna(ts):
                ts = pd.Timestamp.now(tz="UTC")
            row = pd.DataFrame(
                {
                    "open": [float(bar.get("open", 0.0))],
                    "high": [float(bar.get("high", 0.0))],
                    "low": [float(bar.get("low", 0.0))],
                    "close": [float(bar.get("close", 0.0))],
                    "volume": [float(bar.get("volume", 0.0))],
                },
                index=[ts],
            )
            self._ohlcv = pd.concat([self._ohlcv, row]).sort_index()
            # Feature recomputation for a single bar is not available in isolation.
            # Keep fixed feature matrix for now and rely on periodic historical replay.
            self._logger.warning("append_bar received; live feature expansion is not enabled in this build.")

    async def _infer_once(self) -> InferenceEvent | None:
        if len(self._features) <= self.lookback:
            return None
        # Keep inference pinned to the latest available bar when no new features
        # are appended, instead of idling forever past the array tail.
        self._index = min(max(self._index, self.lookback), len(self._features) - 1)
        guard = get_inference_guard()
        now = datetime.now(timezone.utc)
        # Runtime state is real-time; keep bar timestamp aligned to wall clock.
        bar_ts = now
        if not guard.enabled:
            empty_paths = np.zeros((int(self._settings["model"]["num_paths"]), int(self._settings["model"]["horizon"])), dtype=np.float32)
            event = InferenceEvent(
                timestamp=now,
                bar_timestamp=bar_ts,
                paths=empty_paths,
                median_path=np.median(empty_paths, axis=0),
                band_10=np.percentile(empty_paths, 10, axis=0),
                band_90=np.percentile(empty_paths, 90, axis=0),
                signal="HOLD",
                confidence=0.0,
                regime="UNKNOWN",
                latency_ms=0.0,
                meta={
                    "error": guard.reason or "inference_disabled",
                    "base_time": int(now.timestamp()),
                    "timeframe_sec": int(self.timeframe_sec),
                    "horizon_steps": int(self.horizon_steps),
                },
            )
            return event

        start = asyncio.get_running_loop().time()
        window = self._features[self._index - self.lookback : self._index]
        raw_paths = await asyncio.to_thread(self._model_loader.predict, window)
        latency_ms = (asyncio.get_running_loop().time() - start) * 1000.0
        current_price = float(self.current_price())
        self._logger.info("INFER: _index=%d, price=%.5f, window[-1,0]=%.5f", self._index, current_price, window[-1, 0] if window.size > 0 else 0)
        if current_price <= 0:
            if not self._ohlcv.empty:
                current_price = float(self._ohlcv.iloc[-1]["close"])
                self._logger.info("Using fallback price from OHLC: %.5f", current_price)
            else:
                return InferenceEvent(
                    timestamp=now,
                    bar_timestamp=bar_ts,
                    paths=np.zeros((int(self._settings["model"]["num_paths"]), int(self._settings["model"]["horizon"])), dtype=np.float32),
                    median_path=np.zeros((int(self._settings["model"]["horizon"]),), dtype=np.float32),
                    band_10=np.zeros((int(self._settings["model"]["horizon"]),), dtype=np.float32),
                    band_90=np.zeros((int(self._settings["model"]["horizon"]),), dtype=np.float32),
                    signal="HOLD",
                    confidence=0.0,
                    regime="UNKNOWN",
                    latency_ms=float(latency_ms),
                    meta={
                        "error": "live_price_unavailable_and_no_ohlc",
                        "current_price": 0.0,
                        "base_time": int(now.timestamp()),
                        "timeframe_sec": int(self.timeframe_sec),
                        "horizon_steps": int(self.horizon_steps),
                    },
                )
        self._sync_runtime_ohlc_with_live_price(current_price=current_price, now=now)
        bar_step_seconds = self.timeframe_sec
        execution_cfg = dict(self._settings.get("execution", {}))
        live_chart_payload = prepare_chart_payload(
            paths=raw_paths,
            current_price=current_price,
            bar_timestamp=pd.Timestamp(bar_ts),
            base_step_seconds=bar_step_seconds,
            ohlc=self._ohlcv,
            output_normalized=bool(execution_cfg.get("output_normalized", False)),
            output_mean=float(execution_cfg.get("output_mean", 0.0)),
            output_std=float(execution_cfg.get("output_std", 1.0)),
        )
        live_paths = np.asarray(live_chart_payload.get("paths_matrix", raw_paths), dtype=np.float32)

        recent_prices = self._ohlcv["close"].iloc[max(0, self._index - self.lookback) : self._index]
        signal_snapshot = build_signal_snapshot(
            live_paths,
            recent_prices=recent_prices,
            confidence_threshold=float(self._settings["regime"]["signal_confidence_threshold"]),
            hurst_window=int(self._settings["regime"]["hurst_window"]),
            trending_threshold=float(self._settings["regime"]["trending_threshold"]),
            ranging_threshold=float(self._settings["regime"]["ranging_threshold"]),
        )
        live_decision = self._decision_engine.evaluate_live(
            paths=live_paths,
            entry_price=current_price,
            regime=signal_snapshot.regime,
            ohlc=self._ohlcv.iloc[max(0, self._index - 200) : self._index],
            bar_timestamp=bar_ts,
        )
        snapshot_decision = self._decision_engine.get_active_snapshot(bar_timestamp=bar_ts)
        snapshot_paths = self._decision_engine.get_active_snapshot_paths(bar_timestamp=bar_ts)
        current_decision = snapshot_decision if snapshot_decision is not None else live_decision
        current_paths = snapshot_paths if snapshot_paths is not None else live_paths
        event = InferenceEvent(
            timestamp=now,
            bar_timestamp=bar_ts,
            paths=current_paths.astype(np.float32, copy=False),
            median_path=np.median(current_paths, axis=0).astype(np.float32),
            band_10=np.percentile(current_paths, 10, axis=0).astype(np.float32),
            band_90=np.percentile(current_paths, 90, axis=0).astype(np.float32),
            signal=current_decision.decision,
            confidence=float(current_decision.confidence),
            regime=current_decision.regime,
            latency_ms=float(latency_ms),
            meta={
                "hurst": float(signal_snapshot.hurst_exponent),
                "positive_ratio": float(np.mean(((live_paths[:, -1] - current_price) / max(current_price, 1e-9)) > 0.0)),
                "negative_ratio": float(np.mean(((live_paths[:, -1] - current_price) / max(current_price, 1e-9)) < 0.0)),
                "ev_threshold": float(current_decision.ev_threshold),
                "hold_reason": str(current_decision.hold_reason),
                "current_price": current_price,
                "ev": float(current_decision.ev),
                "std": float(current_decision.std),
                "skew": float(current_decision.skew),
                "rr": int(current_decision.rr),
                "sl_distance": float(current_decision.sl_distance),
                "tp_distance": float(current_decision.tp_distance),
                "entry": float(current_decision.entry),
                "sl": float(current_decision.sl),
                "tp": float(current_decision.tp),
                "snapshot_id": str(current_decision.snapshot_id),
                "snapshot_active": bool(current_decision.snapshot_active),
                "snapshot_created_at": str(current_decision.created_at),
                "snapshot_expires_at": str(current_decision.expires_at),
                "chart_paths": live_chart_payload.get("paths", []),
                "chart_mean_path": live_chart_payload.get("mean_path", []),
                "chart_confidence_band_10": live_chart_payload.get("confidence_band_10", []),
                "chart_confidence_band_90": live_chart_payload.get("confidence_band_90", []),
                "atr": float(live_chart_payload.get("atr", 0.0)),
                "path_min": float(live_chart_payload.get("path_min", current_price)),
                "path_max": float(live_chart_payload.get("path_max", current_price)),
                "scale_factor": float(live_chart_payload.get("scale_factor", 0.0)),
                "base_time": int(now.timestamp()),
                "timeframe_sec": int(self.timeframe_sec),
                "horizon_steps": int(self.horizon_steps),
                "live_signal": self._decision_payload(live_decision),
                "snapshot_signal": self._decision_payload(snapshot_decision),
                "current_signal_source": "snapshot" if snapshot_decision is not None else "live",
                "live_paths": live_paths.tolist(),
                "snapshot_paths": snapshot_paths.tolist() if snapshot_paths is not None else None,
            },
        )
        self._inference_logger.info(
            json.dumps(
                {
                    "timestamp": now.isoformat(),
                    "event": "INFERENCE_DECISION",
                    "ev": float(live_decision.ev),
                    "std": float(live_decision.std),
                    "ev_threshold": float(live_decision.ev_threshold),
                    "confidence": float(live_decision.confidence),
                    "final_decision": str(current_decision.decision),
                    "signal_source": "snapshot" if snapshot_decision is not None else "live",
                    "hold_reason": str(live_decision.hold_reason),
                    "current_price": float(current_price),
                    "path_max": float(live_chart_payload.get("path_max", current_price)),
                    "path_min": float(live_chart_payload.get("path_min", current_price)),
                    "atr": float(live_chart_payload.get("atr", 0.0)),
                    "latency_ms": float(latency_ms),
                }
            )
        )
        return event

    async def _run_loop(self) -> None:
        while self._running:
            try:
                event = await self._infer_once()
                if event is None:
                    await asyncio.sleep(self.state_interval_seconds)
                    continue
                self._latest_event = event
                self._update_global_state(event)
                if self._queue.full():
                    _ = self._queue.get_nowait()
                await self._queue.put(event)
                if self._index < len(self._features) - 1:
                    self._index += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logging.getLogger("nexus.errors").exception("Inference loop failure: %s", exc)
            await asyncio.sleep(self.state_interval_seconds)

    async def start(self) -> None:
        """Start background inference loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name="nexus_inference_runner")

    async def stop(self) -> None:
        """Stop background inference loop."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
