"""Live price provider with MT5 + API fallback."""

from __future__ import annotations

import logging
import os
import time
from typing import Callable

import numpy as np
import pandas as pd
import requests


class LivePriceProvider:
    """Multi-source price provider with fallback chain."""

    def __init__(
        self,
        mt5_get_price: Callable[[], float] | None = None,
        ohlcv: pd.DataFrame | None = None,
        symbol: str = "XAUUSD",
    ):
        self._mt5_get_price = mt5_get_price
        self._ohlcv = ohlcv
        self._symbol = symbol
        self._logger = logging.getLogger("nexus.system")
        self._api_key = os.environ.get("TWELVEDATA_API_KEY", "").strip()
        self._last_api_price: float | None = None
        self._last_api_ts: float = 0.0
        self._cache_ttl_seconds = 3.0  # Cache API price for 3 seconds
        self._cached_price: float = 0.0
        self._cached_ts: float = 0.0

    def _get_twelve_data_price(self) -> float:
        """Get price from TwelveData API."""
        if not self._api_key:
            self._logger.info("TWELVEDATA: No API key configured")
            return 0.0
        try:
            symbol_map = {
                "XAUUSD": "XAU/USD",
                "XAUUSD.x": "XAU/USD",
                "GOLD": "XAU/USD",
            }
            tv_symbol = symbol_map.get(self._symbol, self._symbol)
            url = "https://api.twelvedata.com/price"
            params = {"symbol": tv_symbol, "apikey": self._api_key}
            self._logger.info("TWELVEDATA: Requesting %s", tv_symbol)
            resp = requests.get(url, params=params, timeout=2)
            self._logger.info("TWELVEDATA: Response status=%d", resp.status_code)
            if resp.status_code == 200:
                data = resp.json()
                self._logger.info("TWELVEDATA: Response data=%s", data)
                if "price" in data:
                    price = float(data["price"])
                    if np.isfinite(price) and price > 0:
                        self._last_api_price = price
                        return price
        except Exception as e:  # noqa: BLE001
            self._logger.info("TWELVEDATA API error: %s", e)
        return 0.0

    def _get_ohlc_price(self) -> float:
        """Get price from OHLC fallback."""
        if self._ohlcv is not None and not self._ohlcv.empty:
            return float(self._ohlcv.iloc[-1]["close"])
        return 0.0

    def get_price(self) -> float:
        """Get live price with fallback chain: MT5 -> TwelveData -> OHLC."""
        now = time.time()

        # Check cache first
        if self._cached_price > 0 and (now - self._cached_ts) < self._cache_ttl_seconds:
            self._logger.debug("PRICE_PROVIDER: returning cached price %.5f", self._cached_price)
            return self._cached_price

        self._logger.debug("PRICE_PROVIDER: cache miss, fetching new price")

        # 1. Try MT5
        if self._mt5_get_price is not None:
            try:
                price = float(self._mt5_get_price())
                if np.isfinite(price) and price > 0:
                    self._cached_price = price
                    self._cached_ts = now
                    self._logger.debug("Price source: MT5 -> %.5f", price)
                    return price
            except Exception as e:  # noqa: BLE001
                self._logger.debug("PRICE_PROVIDER: MT5 failed: %s", e)
                pass

        # 2. Try TwelveData API
        self._logger.debug("PRICE_PROVIDER: trying TwelveData, API key length=%d", len(self._api_key))
        price = self._get_twelve_data_price()
        if price > 0:
            self._cached_price = price
            self._cached_ts = now
            self._logger.info("Price source: TwelveData -> %.5f", price)
            return price

        # 3. Fallback to OHLC (no caching for fallback)
        ohlc_price = self._get_ohlc_price()
        if ohlc_price > 0:
            self._logger.debug("Price source: OHLCFallback -> %.5f", ohlc_price)
        return ohlc_price

    @property
    def source(self) -> str:
        """Return current price source for debugging."""
        if self._mt5_get_price is not None:
            try:
                price = float(self._mt5_get_price())
                if np.isfinite(price) and price > 0:
                    return "MT5"
            except Exception:  # noqa: BLE001
                pass

        if self._api_key and self._last_api_price:
            return "TwelveData"

        return "OHLCFallback"

    def update_ohlcv(self, ohlcv: pd.DataFrame) -> None:
        """Update OHLCV reference for fallback."""
        self._ohlcv = ohlcv