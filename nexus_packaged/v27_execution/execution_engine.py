"""Snapshot-based execution decision engine for V27."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import uuid

import numpy as np
import pandas as pd


Decision = str


@dataclass
class ExecutionDecision:
    """Decision and execution-level metrics."""

    decision: Decision
    confidence: float
    ev: float
    std: float
    skew: float
    regime: str
    ev_threshold: float
    rr: int
    sl_distance: float
    tp_distance: float
    entry: float
    sl: float
    tp: float
    snapshot_id: str
    snapshot_active: bool
    created_at: str
    expires_at: str
    hold_reason: str

    def as_meta(self) -> dict:
        return {
            "ev": float(self.ev),
            "std": float(self.std),
            "skew": float(self.skew),
            "ev_threshold": float(self.ev_threshold),
            "rr": int(self.rr),
            "sl_distance": float(self.sl_distance),
            "tp_distance": float(self.tp_distance),
            "snapshot_id": str(self.snapshot_id),
            "snapshot_active": bool(self.snapshot_active),
            "snapshot_created_at": self.created_at,
            "snapshot_expires_at": self.expires_at,
            "hold_reason": self.hold_reason,
        }


class SnapshotExecutionEngine:
    """Generates gated execution decisions and freezes them via snapshots."""

    def __init__(
        self,
        *,
        ev_k: float = 0.02,
        conf_min: float = 0.10,
        min_tick: float = 0.01,
        snapshot_bars: int = 10,
        bar_seconds: int = 60,
    ) -> None:
        self.ev_k = float(np.clip(ev_k, 0.005, 0.05))
        self.conf_min = float(np.clip(conf_min, 0.01, 0.5))
        self.min_tick = float(max(1e-6, min_tick))
        self.snapshot_bars = int(max(1, snapshot_bars))
        self.bar_seconds = int(max(1, bar_seconds))
        self.active_snapshot: dict | None = None

    @staticmethod
    def _norm_confidence(ev: float, std: float) -> float:
        raw = abs(float(ev)) / (float(std) + 1e-6)
        return float(raw / (1.0 + raw))

    @staticmethod
    def _skew(values: np.ndarray) -> float:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size < 3:
            return 0.0
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        if std < 1e-12:
            return 0.0
        return float(np.mean(((arr - mean) / std) ** 3))

    @staticmethod
    def _atr(ohlc: pd.DataFrame, period: int = 14) -> float:
        if ohlc.empty:
            return 0.0
        frame = ohlc.tail(max(period + 2, 20)).copy()
        hl = frame["high"] - frame["low"]
        hc = (frame["high"] - frame["close"].shift(1)).abs()
        lc = (frame["low"] - frame["close"].shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        return float(0.0 if pd.isna(atr) else atr)

    @staticmethod
    def _rr_from_conf(confidence: float) -> int:
        if confidence < 0.10:
            return 0
        if confidence < 0.20:
            return 2
        if confidence < 0.35:
            return 3
        return 4

    def _from_snapshot(self, snapshot: dict) -> ExecutionDecision:
        return ExecutionDecision(
            decision=str(snapshot["decision"]),
            confidence=float(snapshot["confidence"]),
            ev=float(snapshot["ev"]),
            std=float(snapshot["std"]),
            skew=float(snapshot["skew"]),
            regime=str(snapshot["regime"]),
            ev_threshold=float(snapshot["ev_threshold"]),
            rr=int(snapshot["rr"]),
            sl_distance=float(snapshot["sl_distance"]),
            tp_distance=float(snapshot["tp_distance"]),
            entry=float(snapshot["entry"]),
            sl=float(snapshot["sl"]),
            tp=float(snapshot["tp"]),
            snapshot_id=str(snapshot["snapshot_id"]),
            snapshot_active=True,
            created_at=str(snapshot["created_at"]),
            expires_at=str(snapshot["expires_at"]),
            hold_reason="",
        )

    def _evaluate_core(
        self,
        *,
        paths: np.ndarray,
        entry_price: float,
        regime: str,
        ohlc: pd.DataFrame,
        bar_timestamp: datetime,
    ) -> ExecutionDecision:
        ts = pd.Timestamp(bar_timestamp).to_pydatetime()
        now_iso = ts.replace(tzinfo=timezone.utc).isoformat()

        arr = np.asarray(paths, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2 or entry_price <= 0:
            return ExecutionDecision(
                decision="HOLD",
                confidence=0.0,
                ev=0.0,
                std=0.0,
                skew=0.0,
                regime=str(regime),
                ev_threshold=0.0,
                rr=0,
                sl_distance=0.0,
                tp_distance=0.0,
                entry=float(entry_price),
                sl=float(entry_price),
                tp=float(entry_price),
                snapshot_id="",
                snapshot_active=False,
                created_at=now_iso,
                expires_at=now_iso,
                hold_reason="insufficient_paths",
            )

        returns = (arr[:, -1] - float(entry_price)) / float(entry_price)
        ev = float(np.mean(returns))
        std = float(np.std(returns))
        skew = self._skew(returns)
        confidence = self._norm_confidence(ev, std)
        ev_threshold = float(std * self.ev_k)
        rr = self._rr_from_conf(confidence)

        hold_reason = ""
        if abs(ev) < ev_threshold:
            hold_reason = "ev_below_dynamic_threshold"
        elif confidence < self.conf_min:
            hold_reason = f"confidence_below_{self.conf_min:.2f}"
        elif str(regime).upper() == "RANGING" and confidence < 0.15:
            hold_reason = "ranging_and_confidence_below_0.15"

        decision: Decision
        if hold_reason:
            decision = "HOLD"
        else:
            decision = "BUY" if ev > 0 else "SELL"

        atr = self._atr(ohlc, period=14)
        sl_distance = min(max(atr * 0.25, 0.0), atr * 0.8) if atr > 0 else 0.0
        if sl_distance < self.min_tick * 5:
            decision = "HOLD"
            hold_reason = "sl_distance_too_small"
            rr = 0

        tp_distance = float(sl_distance * rr) if rr > 0 else 0.0
        if decision == "BUY":
            sl = float(entry_price - sl_distance)
            tp = float(entry_price + tp_distance)
        elif decision == "SELL":
            sl = float(entry_price + sl_distance)
            tp = float(entry_price - tp_distance)
        else:
            sl = float(entry_price)
            tp = float(entry_price)

        if decision == "HOLD":
            return ExecutionDecision(
                decision="HOLD",
                confidence=confidence,
                ev=ev,
                std=std,
                skew=skew,
                regime=str(regime),
                ev_threshold=ev_threshold,
                rr=rr,
                sl_distance=sl_distance,
                tp_distance=tp_distance,
                entry=float(entry_price),
                sl=sl,
                tp=tp,
                snapshot_id="",
                snapshot_active=False,
                created_at=now_iso,
                expires_at=now_iso,
                hold_reason=hold_reason,
            )

        return ExecutionDecision(
            decision=decision,
            confidence=confidence,
            ev=ev,
            std=std,
            skew=skew,
            regime=str(regime),
            ev_threshold=ev_threshold,
            rr=rr,
            sl_distance=sl_distance,
            tp_distance=tp_distance,
            entry=float(entry_price),
            sl=float(sl),
            tp=float(tp),
            snapshot_id="",
            snapshot_active=False,
            created_at=now_iso,
            expires_at=now_iso,
            hold_reason="",
        )

    def evaluate_live(
        self,
        *,
        paths: np.ndarray,
        entry_price: float,
        regime: str,
        ohlc: pd.DataFrame,
        bar_timestamp: datetime,
    ) -> ExecutionDecision:
        """Evaluate live (non-frozen) decision from current paths."""
        return self._evaluate_core(
            paths=paths,
            entry_price=entry_price,
            regime=regime,
            ohlc=ohlc,
            bar_timestamp=bar_timestamp,
        )

    def get_active_snapshot(self, *, bar_timestamp: datetime) -> ExecutionDecision | None:
        """Return current snapshot decision if still active."""
        if self.active_snapshot is None:
            return None
        ts = pd.Timestamp(bar_timestamp).to_pydatetime().replace(tzinfo=timezone.utc)
        exp = datetime.fromisoformat(str(self.active_snapshot["expires_at"]))
        if ts >= exp:
            self.active_snapshot = None
            return None
        return self._from_snapshot(self.active_snapshot)

    def get_active_snapshot_paths(self, *, bar_timestamp: datetime) -> np.ndarray | None:
        """Return frozen snapshot path matrix when snapshot is active."""
        snap = self.get_active_snapshot(bar_timestamp=bar_timestamp)
        if snap is None or self.active_snapshot is None:
            return None
        matrix = self.active_snapshot.get("paths_matrix")
        if matrix is None:
            return None
        arr = np.asarray(matrix, dtype=np.float32)
        return arr if arr.ndim == 2 and arr.size else None

    def create_snapshot(
        self,
        *,
        live_decision: ExecutionDecision,
        paths: np.ndarray,
        bar_timestamp: datetime,
    ) -> ExecutionDecision | None:
        """Create a frozen snapshot from a live decision (used by auto-trader)."""
        existing = self.get_active_snapshot(bar_timestamp=bar_timestamp)
        if existing is not None:
            return existing
        if live_decision.decision not in {"BUY", "SELL"}:
            return None

        ts = pd.Timestamp(bar_timestamp).to_pydatetime().replace(tzinfo=timezone.utc)
        created_at = ts.replace(tzinfo=timezone.utc)
        expires_at = created_at + timedelta(seconds=self.bar_seconds * self.snapshot_bars)
        snapshot_id = str(uuid.uuid4())
        self.active_snapshot = {
            "snapshot_id": snapshot_id,
            "decision": str(live_decision.decision),
            "entry": float(live_decision.entry),
            "tp": float(live_decision.tp),
            "sl": float(live_decision.sl),
            "confidence": float(live_decision.confidence),
            "ev": float(live_decision.ev),
            "std": float(live_decision.std),
            "skew": float(live_decision.skew),
            "regime": str(live_decision.regime),
            "ev_threshold": float(live_decision.ev_threshold),
            "rr": int(live_decision.rr),
            "sl_distance": float(live_decision.sl_distance),
            "tp_distance": float(live_decision.tp_distance),
            "created_at": created_at.isoformat(),
            "expires_at": expires_at.isoformat(),
            "paths_matrix": np.asarray(paths, dtype=np.float32).copy(),
        }
        return self._from_snapshot(self.active_snapshot)

    def reset_state(self) -> None:
        """Clear all snapshot state."""
        self.active_snapshot = None

    def evaluate(
        self,
        *,
        paths: np.ndarray,
        entry_price: float,
        regime: str,
        ohlc: pd.DataFrame,
        bar_timestamp: datetime,
    ) -> ExecutionDecision:
        """Backward-compatible: current signal = snapshot if active else live."""
        snap = self.get_active_snapshot(bar_timestamp=bar_timestamp)
        if snap is not None:
            return snap
        return self.evaluate_live(
            paths=paths,
            entry_price=entry_price,
            regime=regime,
            ohlc=ohlc,
            bar_timestamp=bar_timestamp,
        )
