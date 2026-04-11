from __future__ import annotations

import asyncio
import copy
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.service.app import (
    FRONTEND_DIST_PATH,
    PaperCloseRequest,
    PaperModifyRequest,
    PaperOpenRequest,
    PaperResetRequest,
    _decorate_kimi_judge,
    build_kimi_context_payload,
    fallback_kimi_judge,
    read_system_telemetry,
)
from src.service.mt5_bridge import MT5Bridge
from src.service.live_data import build_fast_dashboard_payload, build_realtime_chart_payload, fetch_live_quote
from src.service.llm_sidecar import is_nvidia_nim_provider, read_packet_log, request_kimi_judge
from src.ui.web import render_web_app_html
from src.v16.paper import PaperTradingEngine
from src.v18.websocket_feed import LiveFeedManager
from src.v21.runtime import build_v21_local_judge, build_v21_runtime_state


def _judge_call(judge: Mapping[str, Any]) -> str:
    content = dict(judge.get("content", {}) if isinstance(judge.get("content"), Mapping) else {})
    return str(content.get("final_call", content.get("stance", "SKIP"))).upper()


def _judge_comparison_v21(kimi_judge: Mapping[str, Any], local_judge: Mapping[str, Any], runtime: Mapping[str, Any]) -> dict[str, Any]:
    kimi_call = _judge_call(kimi_judge)
    local_call = _judge_call(local_judge)
    agree = kimi_call == local_call
    return {
        "agreement": bool(agree),
        "agreement_label": "aligned" if agree else "split",
        "kimi_call": kimi_call,
        "local_call": local_call,
        "summary": f"Kimi and Local V21 both read {kimi_call}." if agree else f"Kimi is {kimi_call} while Local V21 is {local_call}.",
        "reasoning": "Use this split view to compare Kimi against the locally trained V21 stack on the same 15-minute bar.",
        "preferred_source": "aligned" if agree else "manual_compare",
        "v21_should_execute": bool(runtime.get("should_execute", False)),
        "v21_execution_reason": str(runtime.get("execution_reason", "")),
        "v20_should_execute": bool(runtime.get("should_execute", False)),
        "v20_execution_reason": str(runtime.get("execution_reason", "")),
    }


def _build_v21_forecast(payload: Mapping[str, Any], runtime: Mapping[str, Any]) -> dict[str, Any]:
    candles = list((((payload.get("market") or {}).get("candles")) or []))
    if not candles:
        return {"points": []}
    base_timestamp = candles[-1].get("timestamp")
    base = Path()
    del base
    import pandas as pd

    try:
        base_dt = pd.Timestamp(base_timestamp)
        if base_dt.tzinfo is None:
            base_dt = base_dt.tz_localize("UTC")
        else:
            base_dt = base_dt.tz_convert("UTC")
    except Exception:
        base_dt = pd.Timestamp.utcnow().tz_localize("UTC")
    consensus = list(runtime.get("consensus_path", []))
    minority = list(runtime.get("minority_path", []))
    upper = list(runtime.get("cone_upper", []))
    lower = list(runtime.get("cone_lower", []))
    points: list[dict[str, Any]] = []
    for index in range(1, min(len(consensus), len(minority), len(upper), len(lower))):
        points.append(
            {
                "minutes": index * 5,
                "timestamp": (base_dt + pd.Timedelta(minutes=index * 5)).isoformat(),
                "final_price": consensus[index],
                "minority_price": minority[index],
                "outer_upper": upper[index],
                "outer_lower": lower[index],
            }
        )
    return {"points": points}


class BrokerConnectRequest(BaseModel):
    login: int
    password: str
    server: str
    path: str = ""
    symbol_prefix: str = ""
    symbol_suffix: str = ""
    symbol_overrides: dict[str, str] = {}


class BrokerOrderRequest(BaseModel):
    symbol: str = "XAUUSD"
    direction: str
    volume: float
    stop_loss: float | None = None
    take_profit: float | None = None
    comment: str = "NexusTrader Manual"


class AutoTradeToggleRequest(BaseModel):
    enabled: bool
    symbol: str = "XAUUSD"
    lot_mode: str | None = None
    fixed_lot: float | None = None
    min_lot: float | None = None
    max_lot: float | None = None


def _quick_runtime_from_payload(payload: Mapping[str, Any], *, mode: str) -> dict[str, Any]:
    simulation = dict(payload.get("simulation", {}) if isinstance(payload.get("simulation"), Mapping) else {})
    technical = dict(payload.get("technical_analysis", {}) if isinstance(payload.get("technical_analysis"), Mapping) else {})
    market = dict(payload.get("market", {}) if isinstance(payload.get("market"), Mapping) else {})
    direction = str(simulation.get("direction", "HOLD")).upper()
    confidence = str(simulation.get("confidence_tier", "low")).lower()
    frequency_mode = str(mode).lower() == "frequency"
    should_execute = direction in {"BUY", "SELL"} if frequency_mode else direction in {"BUY", "SELL"} and confidence in {"moderate", "high", "very_high"}
    cone_width = float(simulation.get("cone_width_pips", 0.0) or 0.0)
    return {
        "available": True,
        "runtime_version": "v21_local_fast",
        "selected_branch_id": 1,
        "selected_branch_label": "fast_dashboard_proxy",
        "decision_direction": direction if should_execute else "HOLD",
        "raw_stance": direction,
        "cabr_score": float(simulation.get("cabr_score", 0.0) or 0.0),
        "cabr_raw_score": float(simulation.get("cabr_score", 0.0) or 0.0),
        "cpm_score": float(simulation.get("cpm_score", 0.0) or 0.0),
        "confidence_tier": confidence,
        "sqt_label": str(simulation.get("sqt_label", "NEUTRAL")),
        "cone_width_pips": cone_width,
        "lepl_action": direction,
        "lepl_probabilities": {"execute": 0.55 if should_execute else 0.35, "hold": 0.45 if should_execute else 0.65},
        "lepl_features": {
            "kelly_fraction": 0.02 if should_execute else 0.0,
            "suggested_lot": float(simulation.get("suggested_lot", 0.05) or 0.05),
            "conformal_confidence": float(simulation.get("overall_confidence", 0.0) or 0.0),
            "dangerous_branch_count": 0,
            "paper_equity": 1000.0,
        },
        "should_execute": should_execute,
        "execution_reason": "Fast cached desk runtime is active while the full local V21 stack warms in the background.",
        "branch_scores": [],
        "consensus_path": list(simulation.get("consensus_path", []) or []),
        "minority_path": list(simulation.get("minority_path", []) or []),
        "cone_upper": list(simulation.get("cone_outer_upper", []) or []),
        "cone_lower": list(simulation.get("cone_outer_lower", []) or []),
        "regime_probs": [],
        "v21_mode": "research" if frequency_mode else "production",
        "v21_dir_15m_prob": 0.5,
        "v21_bimamba_prob": 0.5,
        "v21_ensemble_prob": 0.5,
        "v21_disagree_prob": 0.0,
        "v21_meta_label_prob": float(simulation.get("overall_confidence", 0.0) or 0.0),
        "v21_dangerous_branch_count": 0,
        "v21_top_vsn_features": [],
        "v21_regime_label": str(simulation.get("detected_regime", technical.get("structure", "unknown"))),
    }


def create_app_v21() -> FastAPI:
    app = FastAPI(title="Nexus Trader V21 API", version="0.21.0")
    frontend_served = False
    if FRONTEND_DIST_PATH.exists():
        app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIST_PATH), html=True), name="nexus_ui_v21")
        frontend_served = True

    paper_trader = PaperTradingEngine(starting_balance=1000.0)
    feed_manager = LiveFeedManager()
    feed_manager.set_paper_engine(paper_trader)
    broker_bridge = MT5Bridge()
    kimi_cache: dict[str, dict[str, Any]] = {}
    runtime_cache: dict[str, tuple[float, dict[str, Any]]] = {}
    runtime_inflight: set[str] = set()

    def _cache_key(symbol: str, mode: str, provider: str, model: str | None) -> str:
        bucket = str(int(time.time() // 900))
        return "|".join([str(symbol).upper(), str(mode).lower(), str(provider).lower(), (model or "").strip(), bucket])

    def _runtime_key(symbol: str, mode: str) -> str:
        return f"{str(symbol).upper()}|{str(mode).lower()}"

    def _paper_state(symbol: str | None = None) -> dict[str, Any]:
        open_positions = paper_trader.state().get("open_positions", [])
        symbols = {str(position.get("symbol", "")).upper() for position in open_positions if position}
        if symbol:
            symbols.add(str(symbol).upper())
        current_prices = {}
        for symbol_name in sorted(symbols):
            try:
                current_prices[symbol_name] = float(fetch_live_quote(symbol_name) or 0.0)
            except Exception:
                continue
        return paper_trader.state(current_prices=current_prices)

    def _live_price(symbol: str) -> float:
        try:
            return float(fetch_live_quote(symbol) or 0.0)
        except Exception:
            return float(broker_bridge.quote(symbol) or 0.0)

    def _fallback_chart_payload(symbol: str, bars: int = 240) -> dict[str, Any]:
        candles = broker_bridge.recent_candles(symbol, count=max(60, min(int(bars), 720)))
        return {
            "symbol": str(symbol).upper(),
            "candles": candles,
            "source": "mt5_local" if candles else "empty_local",
        }

    def _fallback_dashboard_payload(symbol: str) -> dict[str, Any]:
        candles = broker_bridge.recent_candles(symbol, count=240)
        current_price = float(candles[-1]["close"]) if candles else float(broker_bridge.quote(symbol) or 0.0)
        direction = "HOLD"
        if len(candles) >= 2:
            if candles[-1]["close"] > candles[-2]["close"]:
                direction = "BUY"
            elif candles[-1]["close"] < candles[-2]["close"]:
                direction = "SELL"
        return {
            "symbol": str(symbol).upper(),
            "market": {
                "current_price": current_price,
                "candles": candles,
            },
            "realtime_chart": _fallback_chart_payload(symbol, bars=240),
            "technical_analysis": {
                "structure": "mt5_local_fallback",
                "location": "broker_price_only",
            },
            "simulation": {
                "direction": direction,
                "confidence_tier": "low",
                "tier_label": "MT5 LOCAL",
                "overall_confidence": 0.35,
                "cabr_score": 0.35,
                "cpm_score": 0.35,
                "cone_width_pips": 80.0,
                "consensus_path": [current_price] * 4 if current_price else [],
                "minority_path": [current_price] * 4 if current_price else [],
                "cone_outer_upper": [current_price] * 4 if current_price else [],
                "cone_outer_lower": [current_price] * 4 if current_price else [],
                "sqt_label": "LOCAL",
                "suggested_lot": 0.05,
                "detected_regime": "mt5_local_fallback",
                "branch_count": 3,
            },
            "feeds": {"news": [], "public_discussions": []},
            "mfg": {"disagreement": 0.0, "consensus_drift": 0.0},
        }

    def _decorate_local_v21(payload: dict[str, Any], local_judge: dict[str, Any]) -> dict[str, Any]:
        decorated = _decorate_kimi_judge(payload, local_judge)
        decorated["judge_name"] = "local_v21"
        decorated["provider"] = "local_v21"
        decorated["model"] = "v21_xlstm_bimamba"
        return decorated

    def _autotrade_state(symbol: str = "XAUUSD") -> dict[str, Any]:
        broker = broker_bridge.status()
        return {
            "enabled": bool(broker.get("autotrade_enabled", False)),
            "broker_connected": bool(broker.get("connected", False)),
            "symbol": str(symbol).upper(),
            "config": dict(broker.get("autotrade_config", {}) or {}),
            "last_action": broker.get("last_action", "idle"),
            "last_order": broker.get("last_order"),
            "last_error": broker.get("last_error", ""),
        }

    def _telemetry_with_broker() -> dict[str, Any]:
        telemetry = read_system_telemetry()
        broker = broker_bridge.status()
        if broker.get("connected"):
            auto = "Auto ON" if broker.get("autotrade_enabled") else "Auto OFF"
            server = broker.get("server") or "MT5"
            telemetry["broker_connection"] = f"MT5 {server} • {auto}"
        elif broker.get("installed"):
            telemetry["broker_connection"] = "MT5 ready • disconnected"
        else:
            telemetry["broker_connection"] = "Paper broker • MT5 package missing"
        return telemetry

    def _resolve_runtime(payload: dict[str, Any], symbol: str, mode: str) -> dict[str, Any]:
        now = time.time()
        key = _runtime_key(symbol, mode)
        cached = runtime_cache.get(key)
        if cached is not None and now - cached[0] <= 30:
            return copy.deepcopy(cached[1])

        def _worker(snapshot: dict[str, Any], snapshot_key: str) -> None:
            try:
                runtime = build_v21_runtime_state(snapshot, mode=mode)
                if runtime.get("available", False):
                    runtime_cache[snapshot_key] = (time.time(), runtime)
            except Exception:
                pass
            finally:
                runtime_inflight.discard(snapshot_key)

        if key not in runtime_inflight:
            runtime_inflight.add(key)
            threading.Thread(target=_worker, args=(copy.deepcopy(payload), key), daemon=True).start()

        return _quick_runtime_from_payload(payload, mode=mode)

    def _dashboard_payload(symbol: str, mode: str, llm_provider: str, llm_model: str | None, *, with_kimi: bool = False, force_kimi: bool = False) -> dict[str, Any]:
        try:
            payload = build_fast_dashboard_payload(symbol)
        except Exception:
            payload = _fallback_dashboard_payload(symbol)
        payload["paper_trading"] = _paper_state(symbol)
        try:
            payload["realtime_chart"] = build_realtime_chart_payload(symbol)
        except Exception:
            payload["realtime_chart"] = _fallback_chart_payload(symbol)
        runtime = _resolve_runtime(payload, symbol, mode)
        local_judge = _decorate_local_v21(payload, build_v21_local_judge(payload, runtime))
        simulation = dict(payload.get("simulation", {}))
        simulation.update(
            {
                "direction": runtime.get("decision_direction", simulation.get("direction", "HOLD")),
                "confidence_tier": runtime.get("confidence_tier", simulation.get("confidence_tier", "very_low")),
                "cabr_score": runtime.get("cabr_score", simulation.get("cabr_score", 0.0)),
                "cpm_score": runtime.get("cpm_score", simulation.get("cpm_score", 0.0)),
                "cone_width_pips": runtime.get("cone_width_pips", simulation.get("cone_width_pips", 0.0)),
                "consensus_path": runtime.get("consensus_path", simulation.get("consensus_path", [])),
                "minority_path": runtime.get("minority_path", simulation.get("minority_path", [])),
                "cone_outer_upper": runtime.get("cone_upper", simulation.get("cone_outer_upper", [])),
                "cone_outer_lower": runtime.get("cone_lower", simulation.get("cone_outer_lower", [])),
                "suggested_lot": ((runtime.get("lepl_features") or {}).get("suggested_lot")),
                "should_execute": runtime.get("should_execute", False),
                "execution_reason": runtime.get("execution_reason", ""),
                "sqt_label": runtime.get("sqt_label", simulation.get("sqt_label", "NEUTRAL")),
                "hurst_overall": simulation.get("hurst_overall"),
                "hurst_asymmetry": simulation.get("hurst_asymmetry"),
                "detected_regime": runtime.get("v21_regime_label", simulation.get("detected_regime")),
                "branch_count": max(int(runtime.get("v21_dangerous_branch_count", 0)) + 2, 8),
            }
        )
        payload["simulation"] = simulation
        payload["local_judge"] = local_judge
        payload["v21_runtime"] = runtime
        payload["v20_runtime"] = runtime
        payload["v19_runtime"] = runtime
        payload["broker"] = broker_bridge.status()
        payload["auto_trade"] = _autotrade_state(symbol)
        payload["final_forecast"] = _build_v21_forecast(payload, runtime)
        payload["stack_mode"] = "v21_local_dual_judge"
        cache_key = _cache_key(symbol, mode, llm_provider, llm_model)
        if with_kimi:
            if force_kimi or cache_key not in kimi_cache:
                kimi_context = build_kimi_context_payload(payload)
                if is_nvidia_nim_provider(llm_provider):
                    kimi_judge = request_kimi_judge(symbol, kimi_context, provider=llm_provider, model=llm_model)
                    if not kimi_judge.get("available", False):
                        kimi_judge = {
                            "available": False,
                            "provider": llm_provider,
                            "model": llm_model or "",
                            "error": kimi_judge.get("error", "Kimi request failed."),
                            "content": fallback_kimi_judge(payload),
                        }
                else:
                    kimi_judge = {
                        "available": False,
                        "provider": llm_provider,
                        "model": llm_model or "",
                        "reason": "provider_not_nim",
                        "content": fallback_kimi_judge(payload),
                    }
                kimi_cache[cache_key] = _decorate_kimi_judge(payload, kimi_judge)
            payload["kimi_judge"] = copy.deepcopy(kimi_cache[cache_key])
        else:
            payload["kimi_judge"] = {
                "available": False,
                "provider": llm_provider,
                "model": llm_model or "",
                "reason": "awaiting_15m_kimi_refresh",
                "content": fallback_kimi_judge(payload),
            }
        payload["judge_comparison"] = _judge_comparison_v21(payload["kimi_judge"], local_judge, runtime)
        return payload

    @app.get("/health")
    def health():
        return {"status": "ok", "version": "v21", "host": "app_v21"}

    @app.get("/api/system/telemetry")
    def system_telemetry():
        return _telemetry_with_broker()

    @app.get("/api/chart/realtime")
    def realtime_chart(symbol: str = "XAUUSD", bars: int = 240):
        try:
            return build_realtime_chart_payload(symbol, bars=max(60, min(int(bars), 720)))
        except Exception:
            return _fallback_chart_payload(symbol, bars=max(60, min(int(bars), 720)))

    @app.get("/api/dashboard/live")
    def dashboard_live(symbol: str = "XAUUSD", llm_provider: str = "nvidia_nim", llm_model: str | None = None, mode: str = "frequency"):
        return _dashboard_payload(symbol, mode, llm_provider, llm_model, with_kimi=False, force_kimi=False)

    @app.get("/api/llm/kimi-live")
    def kimi_live(symbol: str = "XAUUSD", llm_provider: str = "nvidia_nim", llm_model: str | None = None, mode: str = "frequency", force: bool = False):
        payload = _dashboard_payload(symbol, mode, llm_provider, llm_model, with_kimi=True, force_kimi=bool(force))
        return {
            "symbol": symbol,
            "mode": mode,
            "kimi_judge": payload["kimi_judge"],
            "local_judge": payload["local_judge"],
            "judge_comparison": payload["judge_comparison"],
            "v21_runtime": payload.get("v21_runtime", {}),
            "v20_runtime": payload.get("v21_runtime", {}),
            "v19_runtime": payload.get("v21_runtime", {}),
        }

    @app.get("/api/llm/judges-live")
    def judges_live(symbol: str = "XAUUSD", llm_provider: str = "nvidia_nim", llm_model: str | None = None, mode: str = "frequency", force: bool = False):
        payload = _dashboard_payload(symbol, mode, llm_provider, llm_model, with_kimi=True, force_kimi=bool(force))
        return {
            "symbol": symbol,
            "mode": mode,
            "kimi_judge": payload["kimi_judge"],
            "local_judge": payload["local_judge"],
            "judge_comparison": payload["judge_comparison"],
            "v21_runtime": payload.get("v21_runtime", {}),
            "v20_runtime": payload.get("v21_runtime", {}),
            "v19_runtime": payload.get("v21_runtime", {}),
        }

    @app.get("/api/llm/kimi-log")
    def kimi_log(limit: int = 12):
        return {"entries": read_packet_log(limit=limit)}

    @app.get("/api/paper/state")
    def paper_state(symbol: str = "XAUUSD"):
        return _paper_state(symbol)

    @app.get("/api/broker/status")
    def broker_status():
        return broker_bridge.status()

    @app.post("/api/broker/connect")
    def broker_connect(request: BrokerConnectRequest):
        status = broker_bridge.connect(
            login=request.login,
            password=request.password,
            server=request.server,
            path=request.path,
            symbol_prefix=request.symbol_prefix,
            symbol_suffix=request.symbol_suffix,
            symbol_overrides=request.symbol_overrides,
        )
        if not status.get("ok", False):
            raise HTTPException(status_code=400, detail=str(status.get("detail", "Broker connect failed.")))
        return status

    @app.post("/api/broker/disconnect")
    def broker_disconnect():
        return broker_bridge.disconnect()

    @app.post("/api/broker/order")
    def broker_order(request: BrokerOrderRequest):
        if not broker_bridge.can_trade():
            raise HTTPException(status_code=400, detail="MT5 is not connected.")
        try:
            order = broker_bridge.place_market_order(
                symbol=request.symbol,
                direction=request.direction,
                volume=request.volume,
                stop_loss=request.stop_loss,
                take_profit=request.take_profit,
                comment=request.comment,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"ok": True, "order": order, "broker": broker_bridge.status()}

    @app.post("/api/autotrade/toggle")
    def autotrade_toggle(request: AutoTradeToggleRequest):
        if bool(request.enabled) and not broker_bridge.can_trade():
            raise HTTPException(status_code=400, detail="Connect MT5 before enabling auto trader.")
        return {
            "ok": True,
            "auto_trade": broker_bridge.set_autotrade(
                bool(request.enabled),
                lot_mode=request.lot_mode,
                fixed_lot=request.fixed_lot,
                min_lot=request.min_lot,
                max_lot=request.max_lot,
            ),
        }

    @app.post("/api/paper/open")
    def paper_open(request: PaperOpenRequest):
        try:
            position = paper_trader.open_position(
                symbol=request.symbol,
                direction=request.direction,
                entry_price=request.entry_price,
                confidence_tier=request.confidence_tier,
                sqt_label=request.sqt_label,
                mode=request.mode,
                leverage=request.leverage,
                stop_pips=request.stop_pips,
                take_profit_pips=request.take_profit_pips,
                stop_loss=request.stop_loss,
                take_profit=request.take_profit,
                manual_lot=request.manual_lot,
                note=request.note,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"opened": position, "paper_trading": _paper_state(request.symbol)}

    @app.post("/api/paper/modify")
    def paper_modify(request: PaperModifyRequest):
        try:
            updated = paper_trader.modify_position(request.trade_id, stop_loss=request.stop_loss, take_profit=request.take_profit)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"modified": True, "position": updated, "paper_trading": _paper_state(str(updated.get('symbol', 'XAUUSD')))}

    @app.post("/api/paper/close")
    def paper_close(request: PaperCloseRequest):
        try:
            closed = paper_trader.close_position(request.trade_id, exit_price=request.exit_price)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"closed": closed, "paper_trading": _paper_state(str(closed.get('symbol', 'XAUUSD')))}

    @app.post("/api/paper/reset")
    def paper_reset(request: PaperResetRequest):
        paper_trader.reset(starting_balance=request.starting_balance)
        return {"paper_trading": _paper_state()}

    @app.websocket("/ws/live")
    async def websocket_live(ws: WebSocket, symbol: str = "XAUUSD"):
        await feed_manager.connect(ws, symbol=symbol)
        try:
            while True:
                message = (await ws.receive_text()).strip()
                if message and message.lower() != "ping":
                    feed_manager.set_symbol(ws, message)
        except WebSocketDisconnect:
            feed_manager.disconnect(ws)
        except Exception:
            feed_manager.disconnect(ws)

    @app.on_event("startup")
    async def startup_tasks():
        if getattr(app.state, "v21_feed_task", None) is None:
            app.state.v21_feed_task = asyncio.create_task(feed_manager.heartbeat_loop(price_fn=_live_price, sqt_fn=lambda: {"label": "V21"}))

        async def _autotrade_loop() -> None:
            while True:
                try:
                    broker = broker_bridge.status()
                    if broker.get("autotrade_enabled") and broker.get("connected"):
                        symbol = "XAUUSD"
                        try:
                            payload = build_fast_dashboard_payload(symbol)
                        except Exception:
                            payload = _fallback_dashboard_payload(symbol)
                        payload["paper_trading"] = _paper_state(symbol)
                        try:
                            payload["realtime_chart"] = build_realtime_chart_payload(symbol)
                        except Exception:
                            payload["realtime_chart"] = _fallback_chart_payload(symbol)
                        try:
                            runtime = build_v21_runtime_state(payload, mode="frequency")
                            if runtime.get("available", False):
                                runtime_cache[_runtime_key(symbol, "frequency")] = (time.time(), runtime)
                            else:
                                runtime = _quick_runtime_from_payload(payload, mode="frequency")
                        except Exception:
                            runtime = _quick_runtime_from_payload(payload, mode="frequency")
                        judge = build_v21_local_judge(payload, runtime)
                        content = dict(judge.get("content", {}))
                        stance = str(runtime.get("decision_direction", content.get("final_call", content.get("stance", "HOLD")))).upper()
                        bucket = int(time.time() // 900)
                        last_order = broker.get("last_order") or {}
                        last_bucket = int(((broker_bridge._read_state().get("last_bucket_by_symbol", {}) or {}).get(symbol, -1)))
                        if (
                            runtime.get("should_execute", False)
                            and stance in {"BUY", "SELL"}
                            and not broker_bridge.has_open_position(symbol)
                            and bucket != last_bucket
                        ):
                            suggested_volume = float((runtime.get("lepl_features") or {}).get("suggested_lot", 0.01) or 0.01)
                            volume, volume_source = broker_bridge.resolve_autotrade_volume(suggested_volume)
                            order = broker_bridge.place_market_order(
                                symbol=symbol,
                                direction=stance,
                                volume=max(round(volume, 2), 0.01),
                                stop_loss=content.get("stop_loss"),
                                take_profit=content.get("take_profit"),
                                comment="NexusTrader V21 Auto",
                            )
                            state = broker_bridge._read_state()
                            last_by_symbol = dict(state.get("last_bucket_by_symbol", {}) or {})
                            last_by_symbol[symbol] = bucket
                            broker_bridge._patch_state(
                                last_bucket_by_symbol=last_by_symbol,
                                last_order=order | {"volume_source": volume_source, "suggested_volume": suggested_volume},
                                last_action="autotrade_order_sent",
                                last_error="",
                            )
                        elif runtime.get("should_execute", False) and stance in {"BUY", "SELL"} and broker_bridge.has_open_position(symbol):
                            broker_bridge._patch_state(last_action="autotrade_skipped_existing_position")
                        elif not runtime.get("should_execute", False):
                            broker_bridge._patch_state(last_action="autotrade_waiting")
                        else:
                            broker_bridge._patch_state(last_action=f"autotrade_no_action_{stance.lower()}")
                    await asyncio.sleep(20)
                except Exception as exc:
                    broker_bridge._patch_state(last_error=str(exc), last_action="autotrade_error")
                    await asyncio.sleep(20)

        if getattr(app.state, "v21_autotrade_task", None) is None:
            app.state.v21_autotrade_task = asyncio.create_task(_autotrade_loop())

        # Warm the default desk payload once so the first browser load does not sit on the initializing banner.
        def _warm_default() -> None:
            try:
                try:
                    payload = build_fast_dashboard_payload("XAUUSD")
                except Exception:
                    payload = _fallback_dashboard_payload("XAUUSD")
                payload["paper_trading"] = _paper_state("XAUUSD")
                try:
                    payload["realtime_chart"] = build_realtime_chart_payload("XAUUSD")
                except Exception:
                    payload["realtime_chart"] = _fallback_chart_payload("XAUUSD")
                runtime_cache[_runtime_key("XAUUSD", "frequency")] = (time.time(), build_v21_runtime_state(payload, mode="frequency"))
            except Exception:
                pass

        threading.Thread(target=_warm_default, daemon=True).start()

    @app.on_event("shutdown")
    async def shutdown_tasks():
        task = getattr(app.state, "v21_feed_task", None)
        if task is not None:
            task.cancel()
            app.state.v21_feed_task = None
        autotrade_task = getattr(app.state, "v21_autotrade_task", None)
        if autotrade_task is not None:
            autotrade_task.cancel()
            app.state.v21_autotrade_task = None

    if not frontend_served:
        @app.get("/ui", response_class=HTMLResponse)
        def ui():
            return render_web_app_html()

    @app.get("/ui-legacy", response_class=HTMLResponse)
    def ui_legacy():
        return render_web_app_html()

    return app


app = create_app_v21()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8021)
