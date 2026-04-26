"""Core runtime components for Nexus packaged system."""

from __future__ import annotations

from nexus_packaged.core.backtest_engine import BacktestConfig, BacktestEngine, BacktestResult, TradeRecord
from nexus_packaged.core.diffusion_loader import BaseModelLoader, DiffusionModelLoader
from nexus_packaged.core.inference_runner import InferenceEvent, InferenceRunner
from nexus_packaged.core.model_guard import get_inference_guard, set_inference_enabled
from nexus_packaged.core.regime_detector import SignalSnapshot, build_signal_snapshot

__all__ = [
    "BaseModelLoader",
    "DiffusionModelLoader",
    "InferenceRunner",
    "InferenceEvent",
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    "TradeRecord",
    "SignalSnapshot",
    "build_signal_snapshot",
    "get_inference_guard",
    "set_inference_enabled",
]

