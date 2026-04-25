"""Local FastAPI server for Nexus packaged runtime."""

from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from nexus_packaged.core.backtest_engine import BacktestConfig, BacktestEngine
from nexus_packaged.core.model_guard import get_inference_guard
from nexus_packaged.trading.auto_trader import AutoTradeConfig
from nexus_packaged.trading.manual_trader import ManualOrderRequest
from nexus_packaged.v27_execution.web_bridge import register_execution_routes


@dataclass
class AppState:
    """Runtime object references used by API handlers."""

    settings: dict[str, Any]
    inference_runner: Any
    news_aggregator: Any
    trade_manager: Any
    auto_trader: Any
    manual_trader: Any
    mt5_connector: Any
    integrity_ok: bool
    features_path: str
    ohlcv_path: str
    backtest_jobs: dict[str, Any]
    backtest_executor: ProcessPoolExecutor
    settings_path: str


class ToggleRequest(BaseModel):
    enabled: bool


class CloseTradeRequest(BaseModel):
    trade_id: str


class MT5AccountConfigRequest(BaseModel):
    login: int | None = None
    password: str | None = None
    server: str | None = None
    execution_enabled: bool | None = None
    reconnect_attempts: int | None = None
    reconnect_delay_seconds: int | None = None
    reconnect_now: bool = True
    persist_to_settings: bool = False


def _persist_mt5_settings(settings_path: str, mt5_payload: dict[str, Any]) -> None:
    """Persist MT5 section updates into settings.json."""
    path = Path(settings_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("mt5", {}).update(mt5_payload)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _serialize_backtest_result(result: Any) -> dict[str, Any]:
    payload = asdict(result)
    payload["equity_curve"] = result.equity_curve.tolist()
    payload["drawdown_curve"] = result.drawdown_curve.tolist()
    payload["trade_log"] = [trade.to_dict() for trade in result.trade_log]
    payload["config"] = asdict(result.config)
    return payload


def _run_backtest_process(config_payload: dict[str, Any], features_path: str, ohlcv_path: str) -> dict[str, Any]:
    """ProcessPool worker."""
    features = np.load(features_path)
    ohlcv = pd.read_parquet(ohlcv_path)
    cfg = BacktestConfig(**config_payload)
    engine = BacktestEngine(cfg, features, ohlcv)
    result = engine.run()
    return _serialize_backtest_result(result)


def create_app(app_state: AppState) -> FastAPI:
    """Create and configure FastAPI app."""
    app = FastAPI(title="Nexus Trader API", version=str(app_state.settings.get("version", "v27.1")))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger = logging.getLogger("nexus.system")
    error_logger = logging.getLogger("nexus.errors")

    async def _guarded(handler):
        try:
            return await handler()
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            error_logger.exception("API handler failure: %s", exc)
            raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": str(app_state.settings.get("version", "v27.1"))}

    @app.get("/state")
    async def state() -> dict[str, Any]:
        async def _impl():
            now = datetime.now(timezone.utc)
            summary = app_state.trade_manager.get_session_summary()
            guard = get_inference_guard()
            event = app_state.inference_runner.latest_event
            state_payload = (
                app_state.inference_runner.get_global_state()
                if hasattr(app_state.inference_runner, "get_global_state")
                else {}
            )
            payload = dict(state_payload)
            # Always get fresh live price for API responses
            live_price = float(app_state.inference_runner.current_price())
            if live_price > 0:
                payload["price"] = live_price
            price = float(payload.get("price", live_price))
            payload["timestamp"] = str(payload.get("timestamp", now.isoformat()))
            payload["price"] = price
            payload["inference_latency_ms"] = float(event.latency_ms if event else 0.0)
            payload["mt5_connected"] = bool(app_state.mt5_connector.is_connected)
            payload["integrity_ok"] = bool(app_state.integrity_ok and guard.enabled)
            payload["last_price"] = price
            payload["auto_trade_enabled"] = bool(app_state.auto_trader.config.enabled)
            payload["auto_trade_mode"] = str(app_state.auto_trader.config.mode)
            payload["open_trades"] = int(summary.open_trades)
            payload["daily_pnl_usd"] = float(summary.daily_pnl_usd)
            payload["session_drawdown_pct"] = float(summary.session_drawdown_pct)
            return payload

        return await _guarded(_impl)

    @app.post("/reload")
    async def reload_runtime() -> dict[str, Any]:
        async def _impl():
            if hasattr(app_state.inference_runner, "reload_state"):
                app_state.inference_runner.reload_state()
            if hasattr(app_state.auto_trader, "reset_state"):
                app_state.auto_trader.reset_state()
            return {"status": "reloaded", "timestamp": datetime.now(timezone.utc).isoformat()}

        return await _guarded(_impl)

    @app.get("/prediction")
    async def prediction() -> dict[str, Any]:
        async def _impl():
            guard = get_inference_guard()
            if not guard.enabled:
                return {"error": "integrity_check_failed"}
            event = app_state.inference_runner.latest_event
            if event is None:
                raise HTTPException(status_code=404, detail={"error": "no_prediction_available"})
            return event.to_dict()

        return await _guarded(_impl)

    @app.get("/news")
    async def news() -> list[dict[str, Any]]:
        async def _impl():
            items = app_state.news_aggregator.get_cached()
            if not items:
                items = await app_state.news_aggregator.fetch_all()
            return [item.to_dict() for item in items]

        return await _guarded(_impl)

    @app.get("/mt5/account")
    async def mt5_account() -> dict[str, Any]:
        async def _impl():
            cfg = app_state.mt5_connector.get_runtime_config()
            info = await app_state.mt5_connector.get_account_info()
            return {"config": cfg, "account_info": info}

        return await _guarded(_impl)

    @app.post("/mt5/account/config")
    async def mt5_account_config(payload: MT5AccountConfigRequest) -> dict[str, Any]:
        async def _impl():
            update_payload = payload.model_dump(exclude_none=True)
            reconnect_now = bool(update_payload.pop("reconnect_now", True))
            persist_to_settings = bool(update_payload.pop("persist_to_settings", False))

            app_state.mt5_connector.update_runtime_config(update_payload)
            if persist_to_settings:
                _persist_mt5_settings(app_state.settings_path, update_payload)
            connected = bool(app_state.mt5_connector.is_connected)
            if reconnect_now:
                connected = bool(await app_state.mt5_connector.reconnect())
            return {
                "status": "updated",
                "connected": connected,
                "config": app_state.mt5_connector.get_runtime_config(),
            }

        return await _guarded(_impl)

    @app.post("/mt5/account/connect")
    async def mt5_account_connect() -> dict[str, Any]:
        async def _impl():
            connected = bool(await app_state.mt5_connector.reconnect())
            return {"connected": connected, "config": app_state.mt5_connector.get_runtime_config()}

        return await _guarded(_impl)

    @app.post("/mt5/account/disconnect")
    async def mt5_account_disconnect() -> dict[str, Any]:
        async def _impl():
            await app_state.mt5_connector.disconnect()
            return {"connected": False, "config": app_state.mt5_connector.get_runtime_config()}

        return await _guarded(_impl)

    @app.get("/trades/open")
    async def trades_open() -> list[dict[str, Any]]:
        async def _impl():
            return [trade.to_dict() for trade in app_state.trade_manager.get_open_trades()]

        return await _guarded(_impl)

    @app.get("/trades/history")
    async def trades_history(limit: int = Query(default=50, ge=1, le=500), source: str | None = Query(default=None)) -> list[dict[str, Any]]:
        async def _impl():
            return [trade.to_dict() for trade in app_state.trade_manager.get_trade_history(limit=limit, source=source)]

        return await _guarded(_impl)

    @app.post("/auto_trade/toggle")
    async def auto_trade_toggle(body: ToggleRequest) -> dict[str, Any]:
        async def _impl():
            app_state.auto_trader.config.enabled = bool(body.enabled)
            return {
                "auto_trade_enabled": bool(app_state.auto_trader.config.enabled),
                "mode": app_state.auto_trader.config.mode,
            }

        return await _guarded(_impl)

    @app.post("/auto_trade/config")
    async def auto_trade_config(payload: dict[str, Any]) -> dict[str, Any]:
        async def _impl():
            cfg = AutoTradeConfig(**payload)
            app_state.auto_trader.update_config(cfg)
            return {"status": "updated", "auto_trade_enabled": bool(app_state.auto_trader.config.enabled)}

        return await _guarded(_impl)

    @app.post("/trade/manual")
    async def trade_manual(payload: dict[str, Any]) -> dict[str, Any]:
        async def _impl():
            try:
                req = ManualOrderRequest(**payload)
                trade = app_state.manual_trader.place_trade(req)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc
            return trade.to_dict()

        return await _guarded(_impl)

    @app.post("/trade/close")
    async def trade_close(payload: CloseTradeRequest) -> dict[str, Any]:
        async def _impl():
            ok = app_state.manual_trader.close_trade(payload.trade_id)
            if not ok:
                raise HTTPException(status_code=404, detail={"error": "trade_not_found"})
            return {"status": "closed", "trade_id": payload.trade_id}

        return await _guarded(_impl)

    @app.post("/backtest/run")
    async def backtest_run(payload: dict[str, Any]) -> dict[str, Any]:
        async def _impl():
            job_id = str(__import__("uuid").uuid4())
            app_state.backtest_jobs[job_id] = {"status": "pending", "result": None, "error": None}
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(
                app_state.backtest_executor,
                _run_backtest_process,
                payload,
                app_state.features_path,
                app_state.ohlcv_path,
            )

            async def _collect() -> None:
                try:
                    result = await future
                    app_state.backtest_jobs[job_id] = {"status": "completed", "result": result, "error": None}
                except Exception as exc:  # noqa: BLE001
                    error_logger.exception("Backtest job failed: %s", exc)
                    app_state.backtest_jobs[job_id] = {"status": "failed", "result": None, "error": str(exc)}

            asyncio.create_task(_collect(), name=f"backtest_collect_{job_id}")
            return {"job_id": job_id, "status": "started"}

        return await _guarded(_impl)

    @app.get("/backtest/results/{job_id}")
    async def backtest_results(job_id: str) -> dict[str, Any]:
        async def _impl():
            if job_id not in app_state.backtest_jobs:
                raise HTTPException(status_code=404, detail={"error": "job_not_found"})
            payload = app_state.backtest_jobs[job_id]
            if payload["status"] == "completed":
                return payload["result"]
            if payload["status"] == "failed":
                return {"status": "failed", "error": payload["error"]}
            return {"status": "pending"}

        return await _guarded(_impl)

    @app.on_event("shutdown")
    async def _shutdown_event() -> None:
        logger.info("API shutting down")

    register_execution_routes(app, app_state)

    return app
