"""Launcher for Nexus V28 clean execution stack."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import uvicorn

from nexus_packaged.core.diffusion_loader import DiffusionModelLoader
from nexus_packaged.mt5.connector import MT5Connector
from nexus_packaged.protection.encryptor import derive_key_from_env
from nexus_packaged.v28_clean.api import create_v28_app
from nexus_packaged.v28_clean.engine import V28CleanEngine


def _load_settings(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Nexus V28 clean execution stack.")
    parser.add_argument("--settings", default="nexus_packaged/config/settings.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--no-mt5", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger("nexus.system")

    settings = _load_settings(Path(args.settings))
    features_path = Path(settings["data"]["features_path"])
    ohlcv_path = Path(settings["data"]["ohlcv_path"])
    features = np.load(features_path)
    ohlcv = pd.read_parquet(ohlcv_path)

    key = derive_key_from_env(
        env_var=str(settings["model"]["key_env_var"]),
        salt=str(settings["model"]["key_salt"]),
    )
    model_loader = DiffusionModelLoader(str(settings["model"]["encrypted_weights_path"]), key, settings=settings)
    model_loader.load()
    model_loader.warm_up()

    mt5 = MT5Connector(settings)
    if not args.no_mt5:
        connected = asyncio.run(mt5.connect())
        logger.info("MT5 connected: %s", connected)
    else:
        logger.info("Starting without MT5 connection (--no-mt5).")

    engine = V28CleanEngine(
        settings=settings,
        model_loader=model_loader,
        mt5_connector=mt5,
        features=features,
        ohlcv=ohlcv,
    )
    engine.start()
    app = create_v28_app(engine)
    config = uvicorn.Config(app, host=args.host, port=int(args.port), log_level="info")
    server = uvicorn.Server(config)
    try:
        server.run()
    finally:
        engine.stop()
        asyncio.run(mt5.disconnect())


if __name__ == "__main__":
    main()
