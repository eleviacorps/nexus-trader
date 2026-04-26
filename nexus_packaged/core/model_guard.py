"""Global inference guard state."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock


@dataclass
class InferenceGuardState:
    """Tracks whether inference is globally enabled."""

    enabled: bool = True
    reason: str = ""


_STATE = InferenceGuardState()
_LOCK = Lock()


def set_inference_enabled(enabled: bool, reason: str = "") -> None:
    """Update global inference guard state."""
    with _LOCK:
        _STATE.enabled = bool(enabled)
        _STATE.reason = reason


def get_inference_guard() -> InferenceGuardState:
    """Get a snapshot of current guard state."""
    with _LOCK:
        return InferenceGuardState(enabled=_STATE.enabled, reason=_STATE.reason)

