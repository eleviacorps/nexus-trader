"""Nexus packaged entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import asdict
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import uvicorn

from nexus_packaged.api.server import AppState, create_app
from nexus_packaged.core.backtest_engine import BacktestConfig, BacktestEngine
from nexus_packaged.core.diffusion_loader import DiffusionModelLoader
from nexus_packaged.core.feature_pipeline import run_pipeline
from nexus_packaged.core.inference_runner import InferenceRunner
from nexus_packaged.core.model_guard import get_inference_guard, set_inference_enabled
from nexus_packaged.core.price_provider import LivePriceProvider
from nexus_packaged.mt5.connector import MT5Connector
from nexus_packaged.news.aggregator import NewsAggregator
from nexus_packaged.protection.encryptor import derive_key_from_env, encrypt_model_weights
from nexus_packaged.protection.integrity import current_runtime_path, verify_integrity
from nexus_packaged.trading.auto_trader import AutoTradeConfig, AutoTrader
from nexus_packaged.trading.manual_trader import ManualTrader
from nexus_packaged.trading.trade_manager import TradeManager

INFERENCE_ENABLED = True


def _load_settings() -> dict[str, Any]:
    return json.loads(Path("nexus_packaged/config/settings.json").read_text(encoding="utf-8"))


def _load_local_env_file(path: Path) -> dict[str, str]:
    """Load key-value pairs from a local .env-style file.

    The file is optional and designed for local machine runtime convenience.
    Existing process environment values take precedence.
    """
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env_key = key.strip()
        env_value = value.strip().strip("'").strip('"')
        if not env_key:
            continue
        if os.environ.get(env_key) is None:
            os.environ[env_key] = env_value
            loaded[env_key] = env_value
    return loaded


def _configure_logging(settings: dict[str, Any]) -> None:
    """Configure all required rotating loggers before runtime init."""
    log_cfg = dict(settings.get("logging", {}))
    level_name = str(log_cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    max_bytes = int(float(log_cfg.get("max_file_size_mb", 50)) * 1024 * 1024)
    backup_count = int(log_cfg.get("backup_count", 3))

    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(fmt)
    root.addHandler(stream_handler)

    logger_map = {
        "nexus.system": str(log_cfg.get("system_log", "nexus_packaged/logs/system.log")),
        "nexus.inference": str(log_cfg.get("inference_log", "nexus_packaged/logs/inference.log")),
        "nexus.trades": str(log_cfg.get("trades_log", "nexus_packaged/logs/trades.log")),
        "nexus.errors": str(log_cfg.get("error_log", "nexus_packaged/logs/errors.log")),
    }
    for logger_name, path_text in logger_map.items():
        path = Path(path_text)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.touch()
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        logger.handlers.clear()
        rotating = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
        rotating.setLevel(level)
        if logger_name == "nexus.trades":
            # Trades log must be JSONL; keep raw message body in file output.
            rotating.setFormatter(logging.Formatter("%(message)s"))
        else:
            rotating.setFormatter(fmt)
        logger.addHandler(rotating)
        logger.addHandler(stream_handler)
        logger.propagate = False


class AsyncServiceHost:
    """Owns a dedicated asyncio loop for MT5/news/inference/autotrade."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True, name="nexus_async_services")
        self.started = False

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def start(self) -> None:
        if self.started:
            return
        self.thread.start()
        self.started = True

    def submit(self, coro):
        if not self.started:
            self.start()
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def stop(self) -> None:
        if not self.started:
            return
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5.0)
        self.started = False


class APIServerThread:
    """Daemon thread wrapping uvicorn server."""

    def __init__(self, app, host: str, port: int):
        self.config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
        )
        self.server = uvicorn.Server(self.config)
        self.thread = threading.Thread(target=self.server.run, daemon=True, name="nexus_api")

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5.0)

    def is_running(self) -> bool:
        return self.thread.is_alive()


def _create_backtest_executor(system_logger: logging.Logger):
    """Create a process pool for backtests with sandbox-safe fallback."""
    try:
        return ProcessPoolExecutor(max_workers=1)
    except (PermissionError, OSError) as exc:
        system_logger.warning(
            "ProcessPoolExecutor unavailable in this runtime (%s). Falling back to ThreadPoolExecutor.",
            exc,
        )
        return ThreadPoolExecutor(max_workers=1, thread_name_prefix="nexus_backtest")


async def _periodic_news_refresh(news: NewsAggregator) -> None:
    ttl = max(30, int(news.ttl))
    while True:
        try:
            await news.fetch_all()
        except Exception:  # noqa: BLE001
            logging.getLogger("nexus.errors").exception("News refresh failed")
        await asyncio.sleep(ttl)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nexus Trader packaged launcher")
    parser.add_argument("--no-mt5", action="store_true", help="Skip MT5 connection.")
    parser.add_argument("--no-webview", action="store_true", help="Force ASCII chart fallback.")
    parser.add_argument("--backtest-only", action="store_true", help="Run headless backtest and exit.")
    parser.add_argument("--rebuild-data", action="store_true", help="Force feature pipeline rebuild and exit.")
    parser.add_argument("--encrypt-model", type=str, default="", help="Encrypt model at PATH and exit.")
    parser.add_argument("--paper", action="store_true", help="Force paper mode.")
    return parser


def _run_backtest_only(settings: dict[str, Any], features_path: Path, ohlcv_path: Path) -> None:
    features = np.load(features_path)
    ohlcv = pd.read_parquet(ohlcv_path)
    cfg = BacktestConfig(
        start_date=str(settings["data"]["start_date"]),
        end_date=str(settings["data"]["end_date"]),
        timeframe_minutes=int(settings["backtest"]["default_timeframe_minutes"]),
        mode=str(settings["backtest"]["default_mode"]),
        confidence_threshold=float(settings["backtest"]["default_confidence_threshold"]),
        min_trades_for_valid_result=int(settings["backtest"]["default_min_trades"]),
        risk_reward=float(settings["backtest"]["default_risk_reward"]),
        trade_expiry_bars=int(settings["backtest"]["default_trade_expiry_bars"]),
        lot_mode=str(settings["backtest"]["default_lot_mode"]),
        fixed_lot_size=float(settings["backtest"]["default_fixed_lot_size"]),
        lot_min=float(settings["backtest"]["default_lot_min"]),
        lot_max=float(settings["backtest"]["default_lot_max"]),
        lot_range_mode=str(settings["backtest"]["default_lot_range_mode"]),
        risk_pct_per_trade=float(settings["backtest"]["default_risk_pct_per_trade"]),
        kelly_fraction=float(settings["backtest"]["default_kelly_fraction"]),
        leverage=int(settings["backtest"]["default_leverage"]),
        initial_equity=float(settings["backtest"]["default_initial_equity"]),
        commission_per_lot_round_trip=float(settings["backtest"]["default_commission_per_lot_round_trip"]),
        spread_pips=float(settings["backtest"]["default_spread_pips"]),
        sl_atr_multiplier=float(settings["backtest"]["default_sl_atr_multiplier"]),
        allowed_directions=list(settings["backtest"]["default_allowed_directions"]),
        max_open_trades=int(settings["backtest"]["default_max_open_trades"]),
        contract_size=float(settings["backtest"]["contract_size"]),
        pip_value_per_lot=float(settings["backtest"]["pip_value_per_lot"]),
    )
    result = BacktestEngine(cfg, features, ohlcv).run()
    print(json.dumps({**asdict(result), "equity_curve": result.equity_curve.tolist(), "drawdown_curve": result.drawdown_curve.tolist(), "trade_log": [t.to_dict() for t in result.trade_log], "config": asdict(cfg)}, indent=2, default=str))


def main() -> None:
    """CLI entrypoint."""
    global INFERENCE_ENABLED
    args = _build_parser().parse_args()
    loaded_env = _load_local_env_file(Path("nexus_packaged/.env.local"))
    settings = _load_settings()
    _configure_logging(settings)
    system_logger = logging.getLogger("nexus.system")
    error_logger = logging.getLogger("nexus.errors")
    if loaded_env:
        system_logger.info("Loaded %d environment variables from nexus_packaged/.env.local", len(loaded_env))

    # 2) Load settings done above.
    # 3) Integrity check.
    runtime_path = current_runtime_path()
    expected_hash_path = "dist/nexus_trader.exe.sha256"
    integrity_ok = verify_integrity(runtime_path, expected_hash_path)
    if not integrity_ok:
        set_inference_enabled(False, "integrity_check_failed")
        INFERENCE_ENABLED = False
        error_logger.error("Integrity check failed for runtime: %s", runtime_path)

    if args.encrypt_model:
        try:
            key = derive_key_from_env(
                env_var=str(settings["model"]["key_env_var"]),
                salt=str(settings["model"]["key_salt"]),
            )
            output_path = str(settings["model"]["encrypted_weights_path"])
            encrypt_model_weights(args.encrypt_model, output_path, key)
            system_logger.info("Encrypted model written to %s", output_path)
            return
        except Exception as exc:  # noqa: BLE001
            error_logger.exception("Model encryption failed: %s", exc)
            raise

    # 5) Feature pipeline check.
    try:
        ohlcv_path, features_path = run_pipeline(force_rebuild=bool(args.rebuild_data))
    except Exception as exc:  # noqa: BLE001
        error_logger.exception("Feature pipeline check failed: %s", exc)
        print("Feature artifacts are missing or invalid.")
        print("Run: python -m nexus_packaged.core.feature_pipeline --force-rebuild")
        if args.rebuild_data:
            return
        ohlcv_path = Path(settings["data"]["ohlcv_path"])
        features_path = Path(settings["data"]["features_path"])
    if args.rebuild_data:
        return

    if not ohlcv_path.exists() or not features_path.exists():
        raise RuntimeError("Required feature artifacts are missing.")
    ohlcv = pd.read_parquet(ohlcv_path)
    features = np.load(features_path)

    if args.backtest_only:
        _run_backtest_only(settings, features_path, ohlcv_path)
        return

    # 4) Model initialization (after integrity according to startup order).
    model_loader = None
    try:
        key = derive_key_from_env(
            env_var=str(settings["model"]["key_env_var"]),
            salt=str(settings["model"]["key_salt"]),
        )
        model_loader = DiffusionModelLoader(str(settings["model"]["encrypted_weights_path"]), key, settings=settings)
        model_loader.load()
        model_loader.warm_up()
    except Exception as exc:  # noqa: BLE001
        error_logger.exception("Model load failed: %s", exc)
        set_inference_enabled(False, f"model_load_failed:{exc}")
        INFERENCE_ENABLED = False
        # Keep system alive in degraded mode.
        if model_loader is None:
            class _NullLoader:
                def load(self) -> None:
                    return

                def warm_up(self) -> None:
                    return

                def predict(self, context_window):
                    cfg = settings["model"]
                    return np.zeros((int(cfg["num_paths"]), int(cfg["horizon"])), dtype=np.float32)

                @property
                def is_loaded(self) -> bool:
                    return False

            model_loader = _NullLoader()

    # 6) TradeManager + AutoTrader (forced OFF at startup).
    trade_manager = TradeManager(
        initial_equity=float(settings["backtest"]["default_initial_equity"]),
        pip_value_per_lot=float(settings["broker"]["pip_value_per_lot"]),
        leverage=int(settings["backtest"]["default_leverage"]),
        contract_size=float(settings["broker"]["contract_size"]),
    )
    mt5_connector = MT5Connector(settings)
    system_logger.info("ENV: TWELVEDATA_API_KEY present: %s", "TWELVEDATA_API_KEY" in os.environ)
    price_provider = LivePriceProvider(
        mt5_get_price=mt5_connector.get_live_price,
        ohlcv=ohlcv,
        symbol=str(settings.get("data", {}).get("symbol", "XAUUSD")),
    )
    inference_runner = InferenceRunner(
        model_loader=model_loader,
        features=features,
        ohlcv=ohlcv,
        settings=settings,
        live_price_provider=price_provider.get_price,
    )
    trade_manager.set_price_provider(inference_runner.current_price)
    auto_cfg = AutoTradeConfig(**dict(settings.get("auto_trade", {})))
    auto_cfg.enabled = False
    if args.paper:
        auto_cfg.paper_mode = True
    auto_trader = AutoTrader(auto_cfg, mt5_connector, inference_runner, trade_manager=trade_manager, settings=settings)

    # 7) Manual trader.
    manual_trader = ManualTrader(mt5_connector, trade_manager, settings)

    # 8,9,11) Start async services.
    news_aggregator = NewsAggregator(settings)
    host = AsyncServiceHost()
    host.start()
    news_refresh_task: asyncio.Task | None = None

    async def _bootstrap_async() -> None:
        nonlocal news_refresh_task
        if not args.no_mt5:
            mt5_timeout = int(settings.get("mt5", {}).get("connect_timeout_seconds", 20))
            try:
                connected = await asyncio.wait_for(mt5_connector.connect(), timeout=max(5, mt5_timeout))
                if not connected:
                    system_logger.warning("MT5 not connected during bootstrap; continuing in degraded mode.")
            except asyncio.TimeoutError:
                system_logger.warning("MT5 connect timed out; continuing in degraded mode.")
            except Exception:  # noqa: BLE001
                error_logger.exception("MT5 bootstrap failed; continuing in degraded mode.")
        await inference_runner.start()
        await auto_trader.start()
        news_refresh_task = asyncio.create_task(_periodic_news_refresh(news_aggregator), name="nexus_news_refresh")

    async def _shutdown_async() -> None:
        nonlocal news_refresh_task
        if news_refresh_task is not None and not news_refresh_task.done():
            news_refresh_task.cancel()
            with suppress(asyncio.CancelledError):
                await news_refresh_task
        news_refresh_task = None

    host.submit(_bootstrap_async()).result(timeout=30)

    # 10) Start API server in daemon thread.
    backtest_jobs: dict[str, Any] = {}
    app_state = AppState(
        settings=settings,
        inference_runner=inference_runner,
        news_aggregator=news_aggregator,
        trade_manager=trade_manager,
        auto_trader=auto_trader,
        manual_trader=manual_trader,
        mt5_connector=mt5_connector,
        integrity_ok=bool(get_inference_guard().enabled),
        features_path=str(features_path),
        ohlcv_path=str(ohlcv_path),
        backtest_jobs=backtest_jobs,
        backtest_executor=_create_backtest_executor(system_logger),
        settings_path=str(Path("nexus_packaged/config/settings.json")),
    )
    app = create_app(app_state)
    api_host = str(settings["api"]["host"])
    api_port = int(settings["api"]["port"])
    api_thread = APIServerThread(app, host=api_host, port=api_port)
    api_thread.start()

    # 12) Launch TUI (or degraded headless fallback if textual is unavailable).
    app_ui = None
    runtime = None
    try:
        from nexus_packaged.tui.app import NexusTraderApp, RuntimeContext
    except ModuleNotFoundError as exc:
        system_logger.warning(
            "TUI dependencies unavailable (%s). Running in degraded headless mode; API and workers stay online.",
            exc,
        )
    else:
        runtime = RuntimeContext(
            settings=settings,
            inference_runner=inference_runner,
            news_aggregator=news_aggregator,
            auto_trader=auto_trader,
            manual_trader=manual_trader,
            trade_manager=trade_manager,
            mt5_connector=mt5_connector,
            api_running_flag=api_thread.is_running,
            integrity_ok_flag=lambda: bool(get_inference_guard().enabled),
            ohlcv=ohlcv,
            settings_path=str(Path("nexus_packaged/config/settings.json")),
        )
        app_ui = NexusTraderApp(runtime, no_webview=bool(args.no_webview))
    try:
        if app_ui is not None:
            app_ui.run()
        else:
            # Keep service process alive when running without textual.
            while True:
                time.sleep(1.0)
    finally:
        # 13) Graceful shutdown.
        try:
            host.submit(_shutdown_async()).result(timeout=10)
        except Exception:  # noqa: BLE001
            pass
        try:
            host.submit(auto_trader.stop()).result(timeout=10)
        except Exception:  # noqa: BLE001
            pass
        try:
            host.submit(inference_runner.stop()).result(timeout=10)
        except Exception:  # noqa: BLE001
            pass
        try:
            if bool(settings.get("auto_trade", {}).get("close_all_on_exit", False)):
                trade_manager.close_all(reason="MANUAL")
        except Exception:  # noqa: BLE001
            pass
        try:
            host.submit(mt5_connector.disconnect()).result(timeout=10)
        except Exception:  # noqa: BLE001
            pass
        api_thread.stop()
        app_state.backtest_executor.shutdown(wait=False, cancel_futures=True)
        host.stop()
        logging.shutdown()


if __name__ == "__main__":
    main()
