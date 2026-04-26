from __future__ import annotations

import json
import hashlib
import math
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from config.project_config import (
    CROWD_EVENTS_PATH,
    FEATURE_DIM_CROWD,
    FEATURE_DIM_NEWS,
    LIVE_SIMULATION_HISTORY_PATH,
    NEWS_EVENTS_PATH,
    PRICE_FEATURE_COLUMNS,
    SEQUENCE_LEN,
    V15_ECI_CALENDAR_PATH,
    V9_MEMORY_BANK_ENCODER_PATH,
    V9_MEMORY_BANK_INDEX_PATH,
    V9_PERSONA_CALIBRATION_HISTORY_PATH,
)
from src.service.llm_sidecar import is_nvidia_nim_provider, request_market_context, request_swarm_judgment
from src.service.specialist_bots import run_specialist_bots
from src.mcts.analog import get_historical_analog_scorer
from src.mcts.reverse_collapse import reverse_collapse
from src.mcts.tree import dominant_persona_name, expand_binary_tree, iter_leaves
from src.pipeline.perception import align_event_matrix, build_crowd_numeric_vectors, reduce_text_embeddings
from src.quant.hybrid import build_quant_features, merge_quant_features
from src.simulation.abm import persona_vote_breakdown, simulate_one_step
from src.simulation.personas import default_personas
from src.v6 import build_volatility_envelopes, detect_regime, get_historical_path_retriever, rank_branches_with_selector
from src.v9 import classify_contradiction, load_latest_persona_state, load_memory_bank, query_memory_bank
from src.v15.eci import EconomicCalendarIntegration
from src.v17.mmm import MultifractalMarketMemory
from src.v17.wltc import build_wltc_states
from src.v18.mfg_beliefs import MFGBeliefState

USER_AGENT = "Mozilla/5.0 (compatible; NexusTrader/0.3; +https://github.com/eleviacorps)"

SYMBOL_CONFIG: dict[str, dict[str, Any]] = {
    "XAUUSD": {
        "ticker": "GC=F",
        "label": "Gold",
        "news_query": "gold OR xauusd OR fed OR cpi OR dollar",
        "discussion_query": "gold OR xauusd OR fed OR cpi",
        "spot_api": "https://api.gold-api.com/price/XAU",
        "market_source": "spot_calibrated_gold_api_plus_gcf",
    },
    "EURUSD": {
        "ticker": "EURUSD=X",
        "label": "EUR/USD",
        "news_query": "eurusd OR euro dollar OR ecb OR fed",
        "discussion_query": "eurusd OR euro OR dxy",
        "market_source": "yahoo_chart",
    },
    "BTCUSD": {
        "ticker": "BTC-USD",
        "label": "Bitcoin",
        "news_query": "bitcoin OR btcusd OR crypto regulation OR etf",
        "discussion_query": "bitcoin OR btc OR crypto",
        "market_source": "yahoo_chart",
    },
}

MACRO_TICKERS: dict[str, str] = {
    "dollar_proxy": "UUP",
    "volatility": "^VIX",
    "rates_10y": "^TNX",
    "bonds": "TLT",
}

_CACHE: dict[str, tuple[float, Any]] = {}
SIMULATION_SCHEMA_VERSION = "live_v17_15m"


def _fetch_url(url: str, timeout: int = 20) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _fetch_json(url: str, timeout: int = 20) -> Any:
    return json.loads(_fetch_url(url, timeout=timeout).decode("utf-8"))


def _cached(key: str, ttl_seconds: int, factory):
    now = time.time()
    cached = _CACHE.get(key)
    if cached is not None and now - cached[0] <= ttl_seconds:
        return cached[1]
    value = factory()
    _CACHE[key] = (now, value)
    return value


def _economic_calendar() -> EconomicCalendarIntegration:
    return _cached(
        "v16_eci_integration",
        300,
        lambda: EconomicCalendarIntegration.from_csv(V15_ECI_CALENDAR_PATH),
    )


def _eci_note(context: Mapping[str, Any]) -> str:
    if bool(context.get("avoid_window")):
        mins = round(_safe_float(context.get("mins_to_next_high"), 0.0), 1)
        return f"Major event in {mins} minutes - wide cone, avoid execution."
    if bool(context.get("pre_release")):
        mins = round(_safe_float(context.get("mins_to_next_high"), 0.0), 1)
        return f"High-impact event due in {mins} minutes - simulator stays on, confidence reduced."
    if bool(context.get("reaction_window")):
        mins = round(_safe_float(context.get("mins_since_last_high"), 0.0), 1)
        return f"Post-event reaction window active - recent release was {mins} minutes ago."
    if bool(context.get("post_settling")):
        return "Post-release settling window - structure may stabilize."
    return "No high-impact event pressure near the current 15m horizon."


def _eci_display_context(current_time: Any) -> dict[str, Any]:
    base = _economic_calendar().get_context_at(current_time)
    modifier = 0.0
    if bool(base.get("avoid_window")):
        modifier = 0.65
    elif bool(base.get("pre_release")):
        modifier = 0.35
    elif bool(base.get("reaction_window")):
        modifier = 0.20
    return {
        **base,
        "note": _eci_note(base),
        "cone_width_modifier": round(modifier, 4),
    }


def _symbol_settings(symbol: str) -> dict[str, Any]:
    upper = symbol.strip().upper()
    return SYMBOL_CONFIG.get(upper, SYMBOL_CONFIG["XAUUSD"]) | {"symbol": upper}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return default
        return number
    except Exception:
        return default


def _cone_points(cone_payload: Any) -> list[dict[str, Any]]:
    if isinstance(cone_payload, Mapping):
        points = cone_payload.get("points", [])
        return list(points) if isinstance(points, list) else []
    if isinstance(cone_payload, list):
        return list(cone_payload)
    return []


def _build_wltc_context(recent_bars: list[dict[str, Any]]) -> dict[str, dict[str, float | int | str]]:
    try:
        states = build_wltc_states(recent_bars)
    except Exception:
        states = {}
    return {name: dict(state.summary()) for name, state in states.items()}


def _build_mmm_live_context(price_frame: pd.DataFrame) -> dict[str, Any]:
    if price_frame.empty:
        return {
            "hurst_overall": 0.5,
            "hurst_positive": 0.5,
            "hurst_negative": 0.5,
            "hurst_asymmetry": 0.0,
            "market_memory_regime": "random_walk",
        }
    mmm = MultifractalMarketMemory(window=252)
    features = price_frame.reset_index(drop=False).copy()
    features["return_1"] = pd.to_numeric(features.get("return_1"), errors="coerce").fillna(0.0)
    features["atr_pct"] = pd.to_numeric(features.get("atr_pct"), errors="coerce").fillna(0.0)
    enriched = mmm.rolling_features(features[["return_1", "atr_pct"]], return_col="return_1", vol_col="atr_pct")
    latest = enriched.iloc[-1].to_dict() if not enriched.empty else {}
    return {
        "hurst_overall": round(_safe_float(latest.get("hurst_overall"), 0.5), 4),
        "hurst_positive": round(_safe_float(latest.get("hurst_positive"), 0.5), 4),
        "hurst_negative": round(_safe_float(latest.get("hurst_negative"), 0.5), 4),
        "hurst_asymmetry": round(_safe_float(latest.get("hurst_asymmetry"), 0.0), 4),
        "market_memory_regime": str(latest.get("market_memory_regime", "random_walk")),
    }


def _build_mfg_context(recent_bars: list[dict[str, Any]]) -> dict[str, Any]:
    state = MFGBeliefState()
    state.update_from_bars(recent_bars)
    return state.summary()


def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _fetch_yahoo_chart(ticker: str, interval: str, range_: str) -> pd.DataFrame:
    encoded = urllib.parse.quote(ticker, safe="")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}?interval={interval}&range={range_}&includePrePost=false&events=div%2Csplits"
    payload = _fetch_json(url)
    result = payload["chart"]["result"][0]
    timestamps = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, unit="s", utc=True),
            "open": quote.get("open", []),
            "high": quote.get("high", []),
            "low": quote.get("low", []),
            "close": quote.get("close", []),
            "volume": quote.get("volume", []),
        }
    )
    frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close"]).copy()
    if frame.empty:
        raise ValueError(f"No chart data returned for {ticker}")
    frame["timestamp"] = frame["timestamp"].dt.tz_convert("UTC")
    frame = frame.set_index("timestamp").sort_index()
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
    return frame


def _market_cache_ttl(interval: str) -> int:
    normalized = str(interval or "5m").strip().lower()
    return {
        "1m": 5,
        "2m": 8,
        "5m": 15,
        "15m": 30,
        "30m": 60,
        "60m": 120,
        "1h": 120,
        "1d": 900,
    }.get(normalized, 30)


def fetch_recent_market_candles(
    symbol: str,
    interval: str = "5m",
    range_: str = "5d",
    ttl_seconds: int | None = None,
) -> pd.DataFrame:
    settings = _symbol_settings(symbol)
    cache_key = f"candles:{settings['ticker']}:{interval}:{range_}"
    ttl = int(ttl_seconds if ttl_seconds is not None else _market_cache_ttl(interval))
    frame = _cached(cache_key, ttl, lambda: _fetch_yahoo_chart(settings["ticker"], interval=interval, range_=range_))
    if settings["symbol"] != "XAUUSD":
        return frame

    def _spot_calibrate() -> pd.DataFrame:
        calibrated = frame.copy()
        try:
            payload = _fetch_json(str(settings.get("spot_api")), timeout=20)
            spot_price = _safe_float(payload.get("price"), default=float(calibrated["close"].iloc[-1]))
            futures_price = _safe_float(calibrated["close"].iloc[-1], default=spot_price)
            ratio = spot_price / futures_price if futures_price else 1.0
            for column in ["open", "high", "low", "close"]:
                calibrated[column] = (calibrated[column].astype(float) * ratio).round(5)
            calibrated.attrs["spot_price"] = round(spot_price, 5)
            calibrated.attrs["spot_ratio"] = ratio
            calibrated.attrs["market_source"] = str(settings.get("market_source", "spot_calibrated"))
        except Exception:
            calibrated.attrs["market_source"] = "futures_proxy_gc_f"
        return calibrated

    return _cached(f"{cache_key}:spot", max(2, min(ttl, 8)), _spot_calibrate)


def fetch_live_quote(symbol: str) -> float:
    settings = _symbol_settings(symbol)
    cache_key = f"live_quote:{settings['symbol']}"

    def _factory() -> float:
        if settings["symbol"] == "XAUUSD":
            try:
                payload = _fetch_json(str(settings.get("spot_api")), timeout=10)
                price = _safe_float(payload.get("price"), 0.0)
                if price > 0.0:
                    return price
            except Exception:
                pass
        frame = fetch_recent_market_candles(settings["symbol"], interval="1m", range_="1d", ttl_seconds=5)
        if frame.empty:
            return 0.0
        return _safe_float(frame["close"].iloc[-1], 0.0)

    return round(_safe_float(_cached(cache_key, 3, _factory), 0.0), 5)


def _score_news_text(text: str, market_label: str) -> float:
    lower = text.lower()
    bullish_terms = [
        "dovish",
        "rate cut",
        "cuts",
        "cooling inflation",
        "weak dollar",
        "dollar slips",
        "safe haven",
        "demand rises",
        "accumulation",
        "buying interest",
        market_label.lower(),
    ]
    bearish_terms = [
        "hawkish",
        "rate hike",
        "higher yields",
        "strong dollar",
        "dollar rises",
        "profit-taking",
        "sell-off",
        "liquidation",
        "risk-on",
        "breakdown",
    ]
    score = 0.0
    for term in bullish_terms:
        if term in lower:
            score += 1.0
    for term in bearish_terms:
        if term in lower:
            score -= 1.0
    return float(np.tanh(score / 3.0))


def _read_rss_items(url: str, limit: int) -> list[dict[str, Any]]:
    raw = _fetch_url(url)
    root = ET.fromstring(raw)
    items: list[dict[str, Any]] = []
    for item in root.findall(".//item")[:limit]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        published_raw = (item.findtext("pubDate") or "").strip()
        description = (item.findtext("description") or "").strip()
        source = (item.findtext("source") or "").strip()
        published_at: str | None = None
        if published_raw:
            try:
                parsed = pd.to_datetime(published_raw, utc=True)
                published_at = parsed.isoformat()
            except Exception:
                published_at = None
        if title:
            items.append(
                {
                    "title": title,
                    "link": link,
                    "published_at": published_at,
                    "description": description,
                    "source": source or "rss",
                }
            )
    return items


def fetch_live_news(symbol: str, limit: int = 8) -> list[dict[str, Any]]:
    settings = _symbol_settings(symbol)
    base_queries = [
        settings["news_query"],
        "geopolitical risk OR war OR sanctions OR central bank OR treasury yields",
        "finance markets OR stocks OR bonds OR inflation OR recession OR fed",
    ]

    def _factory():
        items: list[dict[str, Any]] = []
        for raw_query in base_queries:
            query = urllib.parse.quote(raw_query)
            url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
            try:
                items.extend(_read_rss_items(url, limit=limit))
            except Exception:
                continue
        output = []
        seen: set[tuple[str, str]] = set()
        for item in sorted(items, key=lambda entry: entry.get("published_at") or "", reverse=True):
            signature = (str(item.get("title", "")).strip().lower(), str(item.get("source", "")).strip().lower())
            if not signature[0] or signature in seen:
                continue
            seen.add(signature)
            sentiment = _score_news_text(item["title"], settings["label"])
            title_lower = str(item["title"]).lower()
            category = "general_finance"
            if any(term in title_lower for term in ["war", "missile", "sanction", "conflict", "geopolitical"]):
                category = "geopolitical"
            elif any(term in title_lower for term in ["fed", "rates", "inflation", "cpi", "treasury", "yield"]):
                category = "macro"
            elif any(term in title_lower for term in ["gold", "xau", "bullion"]):
                category = "gold"
            output.append(item | {"sentiment": round(sentiment, 4), "category": category})
            if len(output) >= limit:
                break
        if not output and NEWS_EVENTS_PATH.exists():
            try:
                frame = pd.read_parquet(NEWS_EVENTS_PATH).tail(limit)
                for _, row in frame.iterrows():
                    title = str(row.get("text", "")).strip()
                    if not title:
                        continue
                    output.append(
                        {
                            "title": title,
                            "link": str(row.get("url", "")),
                            "published_at": pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce").isoformat() if row.get("timestamp") is not None else None,
                            "description": "",
                            "source": str(row.get("source", "local_news_store")),
                            "sentiment": round(_score_news_text(title, settings["label"]), 4),
                        }
                    )
            except Exception:
                pass
        return output

    return _cached(f"news:{settings['symbol']}", 180, _factory)


def _discussion_classification(score: float) -> str:
    if score >= 0.55:
        return "greed"
    if score <= -0.55:
        return "fear"
    return "uncertain"


def fetch_public_discussions(symbol: str, limit: int = 8) -> list[dict[str, Any]]:
    settings = _symbol_settings(symbol)
    query = urllib.parse.quote(settings["discussion_query"])
    url = f"https://www.reddit.com/search.rss?q={query}&sort=new&t=day"

    def _factory():
        try:
            items = _read_rss_items(url, limit=limit)
        except Exception:
            items = []
        output = []
        for item in items:
            text = " ".join(part for part in [item["title"], item.get("description", "")] if part)
            sentiment = _score_news_text(text, settings["label"])
            output.append(
                {
                    "title": item["title"],
                    "link": item["link"],
                    "published_at": item["published_at"],
                    "source": "reddit_rss",
                    "sentiment": round(sentiment, 4),
                    "classification": _discussion_classification(sentiment),
                }
            )
        if output:
            greed_proxy = [item for item in output if item["classification"] == "greed"]
            fear_proxy = [item for item in output if item["classification"] == "fear"]
            if greed_proxy and fear_proxy:
                output.insert(
                    0,
                    {
                        "title": f"Crowd split: {len(greed_proxy)} greed vs {len(fear_proxy)} fear items",
                        "link": "",
                        "published_at": datetime.now(timezone.utc).isoformat(),
                        "source": "nexus_discussion_synth",
                        "sentiment": round(float(np.mean([item["sentiment"] for item in output])), 4),
                        "classification": "mixed",
                    },
                )
        if not output and CROWD_EVENTS_PATH.exists():
            try:
                frame = pd.read_parquet(CROWD_EVENTS_PATH).tail(limit)
                for _, row in frame.iterrows():
                    source = str(row.get("source", "local_crowd_store"))
                    classification = str(row.get("classification", "uncertain"))
                    value = _safe_float(row.get("value", 50.0), 50.0)
                    sentiment = float(np.tanh((value - 50.0) / 18.0))
                    output.append(
                        {
                            "title": f"{source} sentiment proxy",
                            "link": "",
                            "published_at": pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce").isoformat() if row.get("timestamp") is not None else None,
                            "source": source,
                            "sentiment": round(sentiment, 4),
                            "classification": classification,
                        }
                    )
            except Exception:
                pass
        return output

    return _cached(f"discussions:{settings['symbol']}", 180, _factory)


def _fallback_embed_texts(texts: Iterable[str], output_dim: int) -> np.ndarray:
    text_list = list(texts)
    rows = np.zeros((len(text_list), output_dim), dtype=np.float32)
    for row_index, text in enumerate(text_list):
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], byteorder="little", signed=False) % output_dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            rows[row_index, bucket] += sign
        norm = float(np.linalg.norm(rows[row_index]))
        if norm > 0:
            rows[row_index] /= norm
    return rows


def _embed_texts(texts: list[str], output_dim: int) -> np.ndarray:
    try:
        return reduce_text_embeddings(texts, output_dim)
    except Exception:
        return _fallback_embed_texts(texts, output_dim)


def fetch_fear_greed_snapshot() -> dict[str, Any]:
    def _factory():
        payload = _fetch_json("https://api.alternative.me/fng/?limit=5&format=json")
        rows = payload.get("data", [])
        if not rows:
            return {"value": 50.0, "classification": "neutral", "history": []}
        history = []
        for row in rows:
            value = _safe_float(row.get("value"), 50.0)
            timestamp = pd.to_datetime(int(row.get("timestamp", 0)), unit="s", utc=True).isoformat()
            history.append(
                {
                    "value": value,
                    "classification": str(row.get("value_classification", "neutral")),
                    "timestamp": timestamp,
                }
            )
        latest = history[0]
        return {"value": latest["value"], "classification": latest["classification"], "history": history}

    return _cached("fear_greed", 300, _factory)


def fetch_macro_context() -> dict[str, Any]:
    def _factory():
        series: dict[str, pd.DataFrame] = {}
        for label, ticker in MACRO_TICKERS.items():
            try:
                series[label] = _fetch_yahoo_chart(ticker, interval="1d", range_="3mo")
            except Exception:
                continue
        if not series:
            return {"macro_bias": 0.0, "macro_shock": 0.0, "driver": "macro_neutral", "components": {}}

        def latest_zscore(frame: pd.DataFrame) -> float:
            closes = frame["close"].astype(float)
            rolling_mean = closes.rolling(20, min_periods=5).mean()
            rolling_std = closes.rolling(20, min_periods=5).std(ddof=0).replace(0.0, np.nan)
            score = ((closes - rolling_mean) / rolling_std).fillna(0.0)
            return _safe_float(score.iloc[-1], 0.0)

        components = {name: latest_zscore(frame) for name, frame in series.items()}
        bias = np.tanh(
            (-0.60 * components.get("dollar_proxy", 0.0))
            + (0.35 * components.get("volatility", 0.0))
            + (-0.35 * components.get("rates_10y", 0.0))
            + (0.20 * components.get("bonds", 0.0))
        )
        shock = np.clip(
            abs(components.get("volatility", 0.0)) * 0.45
            + abs(components.get("dollar_proxy", 0.0)) * 0.35
            + abs(components.get("rates_10y", 0.0)) * 0.20,
            0.0,
            1.0,
        )
        driver_map = {
            "dollar_proxy": "dollar_regime",
            "volatility": "risk_regime",
            "rates_10y": "rates_regime",
            "bonds": "bond_regime",
        }
        dominant_component = max(components, key=lambda name: abs(components[name]))
        return {
            "macro_bias": round(float(bias), 4),
            "macro_shock": round(float(shock), 4),
            "driver": driver_map.get(dominant_component, "macro_neutral"),
            "components": {name: round(float(value), 4) for name, value in components.items()},
        }

    return _cached("macro_context", 240, _factory)


def engineer_price_features(candles: pd.DataFrame) -> pd.DataFrame:
    frame = candles.copy().sort_index()
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
    if frame["volume"].max() <= 0:
        frame["volume"] = 1.0

    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    open_ = frame["open"].astype(float)
    volume = frame["volume"].astype(float)

    returns = {period: close.pct_change(period).fillna(0.0) for period in (1, 3, 6, 12)}
    ema_9 = close.ewm(span=9, adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_sig
    rsi_14 = _compute_rsi(close, 14)
    rsi_7 = _compute_rsi(close, 7)
    lowest_14 = low.rolling(14, min_periods=2).min()
    highest_14 = high.rolling(14, min_periods=2).max()
    stoch_k = ((close - lowest_14) / (highest_14 - lowest_14).replace(0.0, np.nan) * 100.0).fillna(50.0)
    stoch_d = stoch_k.rolling(3, min_periods=1).mean().fillna(50.0)

    prev_close = close.shift(1)
    true_range = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_14 = true_range.rolling(14, min_periods=2).mean().bfill().fillna((high - low).mean())
    atr_pct = (atr_14 / close.replace(0.0, np.nan)).fillna(0.0)

    rolling_mean = close.rolling(20, min_periods=5).mean()
    rolling_std = close.rolling(20, min_periods=5).std(ddof=0).replace(0.0, np.nan)
    bb_upper = rolling_mean + 2 * rolling_std
    bb_lower = rolling_mean - 2 * rolling_std
    bb_width = ((bb_upper - bb_lower) / rolling_mean.replace(0.0, np.nan)).fillna(0.0)
    bb_pct = ((close - bb_lower) / (bb_upper - bb_lower).replace(0.0, np.nan)).clip(0.0, 1.0).fillna(0.5)

    candle_range = (high - low).replace(0.0, np.nan)
    body = (close - open_).abs()
    body_pct = (body / candle_range).fillna(0.0)
    upper_wick = ((high - pd.concat([open_, close], axis=1).max(axis=1)) / candle_range).fillna(0.0)
    lower_wick = ((pd.concat([open_, close], axis=1).min(axis=1) - low) / candle_range).fillna(0.0)
    displacement = (close - open_) / atr_14.replace(0.0, np.nan)

    rolling_high = high.rolling(20, min_periods=2).max()
    rolling_low = low.rolling(20, min_periods=2).min()
    dist_to_high = ((rolling_high - close) / atr_14.replace(0.0, np.nan)).clip(lower=0.0).fillna(0.0)
    dist_to_low = ((close - rolling_low) / atr_14.replace(0.0, np.nan)).clip(lower=0.0).fillna(0.0)
    hh = (high >= rolling_high.shift(1).fillna(rolling_high)).astype(float)
    ll = (low <= rolling_low.shift(1).fillna(rolling_low)).astype(float)
    volume_ratio = (volume / volume.rolling(20, min_periods=2).mean().replace(0.0, np.nan)).fillna(1.0)

    timestamps = frame.index.tz_convert("UTC") if frame.index.tz is not None else frame.index.tz_localize("UTC")
    hour = timestamps.hour.astype(float)
    dow = timestamps.dayofweek.astype(float)
    session_asian = ((hour >= 0) & (hour < 7)).astype(float)
    session_london = ((hour >= 7) & (hour < 13)).astype(float)
    session_ny = ((hour >= 13) & (hour < 21)).astype(float)
    session_overlap = ((hour >= 12) & (hour < 16)).astype(float)
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    dow_sin = np.sin(2 * np.pi * dow / 7.0)
    dow_cos = np.cos(2 * np.pi * dow / 7.0)

    enriched = frame.copy()
    enriched["return_1"] = returns[1]
    enriched["return_3"] = returns[3]
    enriched["return_6"] = returns[6]
    enriched["return_12"] = returns[12]
    enriched["rsi_14"] = rsi_14
    enriched["rsi_7"] = rsi_7
    enriched["macd_hist"] = macd_hist
    enriched["macd"] = macd
    enriched["macd_sig"] = macd_sig
    enriched["stoch_k"] = stoch_k
    enriched["stoch_d"] = stoch_d
    enriched["ema_9_ratio"] = ((close / ema_9.replace(0.0, np.nan)) - 1.0).fillna(0.0)
    enriched["ema_21_ratio"] = ((close / ema_21.replace(0.0, np.nan)) - 1.0).fillna(0.0)
    enriched["ema_50_ratio"] = ((close / ema_50.replace(0.0, np.nan)) - 1.0).fillna(0.0)
    enriched["ema_cross"] = np.sign(ema_9 - ema_21).astype(float)
    enriched["atr_pct"] = atr_pct
    enriched["atr_14"] = atr_14
    enriched["bb_width"] = bb_width
    enriched["bb_pct"] = bb_pct
    enriched["body_pct"] = body_pct
    enriched["upper_wick"] = upper_wick
    enriched["lower_wick"] = lower_wick
    enriched["is_bullish"] = (close >= open_).astype(float)
    enriched["displacement"] = displacement.fillna(0.0)
    enriched["dist_to_high"] = dist_to_high
    enriched["dist_to_low"] = dist_to_low
    enriched["hh"] = hh
    enriched["ll"] = ll
    enriched["volume_ratio"] = volume_ratio
    enriched["session_asian"] = session_asian
    enriched["session_london"] = session_london
    enriched["session_ny"] = session_ny
    enriched["session_overlap"] = session_overlap
    enriched["hour_sin"] = hour_sin
    enriched["hour_cos"] = hour_cos
    enriched["dow_sin"] = dow_sin
    enriched["dow_cos"] = dow_cos
    enriched["target_direction"] = np.where(close.shift(-1) > close, 1.0, 0.0)
    enriched = enriched.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    return enriched


def _session_name(timestamp: pd.Timestamp) -> str:
    hour = int(timestamp.tz_convert("UTC").hour if timestamp.tzinfo is not None else timestamp.hour)
    if 0 <= hour < 7:
        return "Asian"
    if 7 <= hour < 13:
        return "London"
    if 13 <= hour < 21:
        return "New York"
    return "After Hours"


def _swing_levels(price_frame: pd.DataFrame, lookback: int = 80) -> dict[str, list[dict[str, Any]]]:
    frame = price_frame.tail(lookback).copy()
    highs: list[dict[str, Any]] = []
    lows: list[dict[str, Any]] = []
    for index in range(2, len(frame) - 2):
        window_high = frame["high"].iloc[index - 2 : index + 3]
        window_low = frame["low"].iloc[index - 2 : index + 3]
        row = frame.iloc[index]
        ts = frame.index[index].isoformat()
        if float(row["high"]) >= float(window_high.max()):
            highs.append({"timestamp": ts, "price": round(float(row["high"]), 5)})
        if float(row["low"]) <= float(window_low.min()):
            lows.append({"timestamp": ts, "price": round(float(row["low"]), 5)})
    return {"resistance": highs[-4:], "support": lows[-4:]}


def _detect_order_blocks(price_frame: pd.DataFrame, atr: float, lookback: int = 80) -> list[dict[str, Any]]:
    frame = price_frame.tail(lookback).copy()
    if len(frame) < 4:
        return []
    zones: list[dict[str, Any]] = []
    threshold = max(atr * 0.35, 0.25)
    for index in range(1, len(frame) - 1):
        current = frame.iloc[index]
        nxt = frame.iloc[index + 1]
        prev = frame.iloc[index - 1]
        body = float(current["close"] - current["open"])
        next_body = float(nxt["close"] - nxt["open"])
        next_extension = abs(float(nxt["close"]) - float(current["close"]))

        if body < 0 and next_body > 0 and float(nxt["close"]) > float(current["high"]) and next_extension >= threshold:
            zones.append(
                {
                    "type": "bullish_order_block",
                    "timestamp": frame.index[index].isoformat(),
                    "low": round(float(current["low"]), 5),
                    "high": round(float(max(current["open"], current["close"])), 5),
                    "strength": round(min(1.0, next_extension / max(atr, 1e-6)), 4),
                    "note": "Last down candle before impulsive upside displacement",
                }
            )
        if body > 0 and next_body < 0 and float(nxt["close"]) < float(current["low"]) and next_extension >= threshold:
            zones.append(
                {
                    "type": "bearish_order_block",
                    "timestamp": frame.index[index].isoformat(),
                    "low": round(float(min(current["open"], current["close"])), 5),
                    "high": round(float(current["high"]), 5),
                    "strength": round(min(1.0, next_extension / max(atr, 1e-6)), 4),
                    "note": "Last up candle before impulsive downside displacement",
                }
            )
        if abs(float(prev["close"]) - float(current["close"])) < threshold * 0.15:
            continue
    return zones[-4:]


def _detect_fair_value_gaps(price_frame: pd.DataFrame, atr: float, lookback: int = 80) -> list[dict[str, Any]]:
    frame = price_frame.tail(lookback).copy()
    gaps: list[dict[str, Any]] = []
    min_gap = max(atr * 0.12, 0.08)
    for index in range(2, len(frame)):
        first = frame.iloc[index - 2]
        third = frame.iloc[index]
        if float(third["low"]) - float(first["high"]) >= min_gap:
            gaps.append(
                {
                    "type": "bullish_fvg",
                    "timestamp": frame.index[index - 1].isoformat(),
                    "low": round(float(first["high"]), 5),
                    "high": round(float(third["low"]), 5),
                    "size": round(float(third["low"] - first["high"]), 5),
                }
            )
        if float(first["low"]) - float(third["high"]) >= min_gap:
            gaps.append(
                {
                    "type": "bearish_fvg",
                    "timestamp": frame.index[index - 1].isoformat(),
                    "low": round(float(third["high"]), 5),
                    "high": round(float(first["low"]), 5),
                    "size": round(float(first["low"] - third["high"]), 5),
                }
            )
    return gaps[-4:]


def build_technical_analysis(price_frame: pd.DataFrame) -> dict[str, Any]:
    latest = price_frame.iloc[-1]
    current_price = float(latest["close"])
    atr = max(float(latest.get("atr_14", 0.0) or 0.0), current_price * 0.001)
    rsi = float(latest.get("rsi_14", 50.0) or 50.0)
    ema_cross = float(latest.get("ema_cross", 0.0) or 0.0)
    session = _session_name(price_frame.index[-1])
    structure = "bullish" if ema_cross > 0.1 and rsi >= 52 else "bearish" if ema_cross < -0.1 and rsi <= 48 else "balanced"
    levels = _swing_levels(price_frame)
    order_blocks = _detect_order_blocks(price_frame, atr=atr)
    fair_value_gaps = _detect_fair_value_gaps(price_frame, atr=atr)

    def _nearest(levels_list: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not levels_list:
            return None
        return min(levels_list, key=lambda item: abs(float(item["price"]) - current_price))

    nearest_support = _nearest(levels["support"])
    nearest_resistance = _nearest(levels["resistance"])
    dealing_range_high = float(price_frame["high"].tail(40).max())
    dealing_range_low = float(price_frame["low"].tail(40).min())
    equilibrium = (dealing_range_high + dealing_range_low) / 2.0
    location = "premium" if current_price > equilibrium else "discount"
    return {
        "session": session,
        "structure": structure,
        "rsi_14": round(rsi, 2),
        "atr_14": round(atr, 5),
        "equilibrium": round(equilibrium, 5),
        "dealing_range_high": round(dealing_range_high, 5),
        "dealing_range_low": round(dealing_range_low, 5),
        "location": location,
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "swing_levels": levels,
        "order_blocks": order_blocks,
        "fair_value_gaps": fair_value_gaps,
    }


def build_chart_snapshot(price_frame: pd.DataFrame, bars: int = 120) -> dict[str, Any]:
    frame = price_frame.tail(bars).copy()
    if frame.empty:
        return {"bars": 0}
    return {
        "bars": int(len(frame)),
        "close_series": [round(float(value), 5) for value in frame["close"].tolist()],
        "high_series": [round(float(value), 5) for value in frame["high"].tolist()],
        "low_series": [round(float(value), 5) for value in frame["low"].tolist()],
        "return_1_series": [round(float(value), 6) for value in frame["return_1"].tolist()],
        "rsi_14_series": [round(float(value), 4) for value in frame["rsi_14"].tolist()],
        "ema_cross_series": [round(float(value), 4) for value in frame["ema_cross"].tolist()],
    }


def _build_news_matrix(price_index: pd.Index, news_items: list[dict[str, Any]]) -> tuple[np.ndarray, float, float]:
    if not news_items:
        return np.zeros((len(price_index), FEATURE_DIM_NEWS), dtype=np.float32), 0.0, 0.0
    timestamps = pd.to_datetime([item["published_at"] for item in news_items], utc=True, errors="coerce")
    if getattr(timestamps, "tz", None) is not None:
        timestamps = timestamps.tz_convert("UTC")
    event_frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "text": [item["title"] for item in news_items],
            "source": [item.get("source", "news") for item in news_items],
        }
    ).dropna(subset=["timestamp"])
    if len(event_frame):
        event_frame["timestamp"] = event_frame["timestamp"].dt.tz_localize(None)
    vectors = _embed_texts(event_frame["text"].tolist(), FEATURE_DIM_NEWS)
    feature_columns = [f"news_{index:02d}" for index in range(FEATURE_DIM_NEWS)]
    if len(event_frame) == 0:
        return np.zeros((len(price_index), FEATURE_DIM_NEWS), dtype=np.float32), 0.0, 0.0
    embedding_frame = pd.concat([event_frame.reset_index(drop=True), pd.DataFrame(vectors, columns=feature_columns)], axis=1)
    aligned, _ = align_event_matrix(price_index, embedding_frame, feature_columns, tolerance_minutes=240)
    scores = np.asarray([_safe_float(item.get("sentiment"), 0.0) for item in news_items], dtype=np.float32)
    return aligned, float(scores.mean() if len(scores) else 0.0), float(np.clip(np.abs(scores).mean() if len(scores) else 0.0, 0.0, 1.0))


def _build_crowd_matrix(price_index: pd.Index, discussions: list[dict[str, Any]], fear_greed: dict[str, Any]) -> tuple[np.ndarray, float, float]:
    rows: list[dict[str, Any]] = []
    for item in discussions:
        score = _safe_float(item.get("sentiment"), 0.0)
        rows.append(
            {
                "timestamp": pd.to_datetime(item.get("published_at"), utc=True, errors="coerce"),
                "value": 50.0 + score * 35.0,
                "classification": item.get("classification", "uncertain"),
                "source": item.get("source", "discussion"),
            }
        )
    for item in fear_greed.get("history", []):
        rows.append(
            {
                "timestamp": pd.to_datetime(item.get("timestamp"), utc=True, errors="coerce"),
                "value": _safe_float(item.get("value"), 50.0),
                "classification": item.get("classification", "neutral"),
                "source": "alternative_me_fng",
            }
        )

    if not rows:
        return np.zeros((len(price_index), FEATURE_DIM_CROWD), dtype=np.float32), 0.0, 0.0
    event_frame = pd.DataFrame(rows).dropna(subset=["timestamp"]).sort_values("timestamp")
    if len(event_frame):
        event_frame["timestamp"] = pd.to_datetime(event_frame["timestamp"], errors="coerce", utc=True).dt.tz_localize(None)
    vectors = build_crowd_numeric_vectors(event_frame, output_dim=FEATURE_DIM_CROWD)
    feature_columns = [f"crowd_{index:02d}" for index in range(FEATURE_DIM_CROWD)]
    embedding_frame = pd.concat([event_frame.reset_index(drop=True), pd.DataFrame(vectors, columns=feature_columns)], axis=1)
    aligned, _ = align_event_matrix(price_index, embedding_frame, feature_columns, tolerance_minutes=1440)
    current_value = _safe_float(fear_greed.get("value"), 50.0)
    crowd_bias = float(np.tanh((50.0 - current_value) / 18.0))
    crowd_extreme = float(np.clip(abs(current_value - 50.0) / 50.0, 0.0, 1.0))
    return aligned, crowd_bias, crowd_extreme


def _build_persona_snapshot(decisions: dict[str, dict[str, float]]) -> dict[str, float]:
    return {name: round(float(values.get("impact", 0.0)), 4) for name, values in decisions.items()}


def _current_persona_weights(personas: Mapping[str, Any]) -> dict[str, float]:
    state = load_latest_persona_state(V9_PERSONA_CALIBRATION_HISTORY_PATH, personas)
    return {name: round(float(weight), 6) for name, weight in state.capital_weights.items()}


def _memory_bank_context(sequence: np.ndarray) -> dict[str, Any]:
    encoder, bank = load_memory_bank(V9_MEMORY_BANK_ENCODER_PATH, V9_MEMORY_BANK_INDEX_PATH)
    if encoder is None or bank is None or sequence.shape[0] < 60:
        return {"analog_confidence": 0.0, "bullish_probability": 0.5, "mean_distance": 0.0, "top_k_indices": []}
    window_size = int(np.asarray(bank.get("window_size", 60)).reshape(-1)[0]) if "window_size" in bank else 60
    window = np.asarray(sequence[-window_size:], dtype=np.float32)
    result = query_memory_bank(encoder, bank, window)
    return {
        "analog_confidence": round(float(result.analog_confidence), 6),
        "bullish_probability": round(float(result.bullish_probability), 6),
        "mean_distance": round(float(result.mean_distance), 6),
        "top_k_indices": result.top_k_indices,
    }


def _alignment_score(probability: float, branch_direction: float) -> float:
    prob = float(np.clip(probability, 0.0, 1.0))
    return float(prob if branch_direction >= 0.0 else (1.0 - prob))


def _weighted_persona_bias(persona_snapshot: Mapping[str, float], persona_weights: Mapping[str, float]) -> float:
    total_weight = 0.0
    weighted_bias = 0.0
    for name, impact in persona_snapshot.items():
        weight = float(persona_weights.get(name, 0.0))
        weighted_bias += float(impact) * weight
        total_weight += abs(weight)
    if total_weight <= 0.0:
        return 0.0
    return float(np.clip(weighted_bias / total_weight, -1.0, 1.0))


def _llm_content(llm_context: Mapping[str, Any] | None) -> dict[str, Any]:
    if not llm_context:
        return {}
    content = llm_context.get("content", {}) if isinstance(llm_context, Mapping) else {}
    return content if isinstance(content, dict) else {}


def _llm_numeric_prior(llm_content: Mapping[str, Any]) -> float:
    if not llm_content:
        return 0.5
    institutional = _safe_float(llm_content.get("institutional_bias"), 0.0)
    whale = _safe_float(llm_content.get("whale_bias"), 0.0)
    retail = _safe_float(llm_content.get("retail_bias"), 0.0)
    market_bias = (0.40 * institutional) + (0.35 * whale) + (0.25 * retail)
    return float(np.clip(0.5 + 0.5 * market_bias, 0.0, 1.0))


def _apply_llm_persona_tilts(personas: dict[str, Any], llm_content: Mapping[str, Any]) -> dict[str, float]:
    bias_map = {
        "institutional": _safe_float(llm_content.get("institutional_bias"), 0.0),
        "whale": _safe_float(llm_content.get("whale_bias"), 0.0),
        "retail": _safe_float(llm_content.get("retail_bias"), 0.0),
        "algo": 0.35 * _safe_float(llm_content.get("institutional_bias"), 0.0) + 0.15 * _safe_float(llm_content.get("whale_bias"), 0.0),
        "noise": 0.15 * _safe_float(llm_content.get("retail_bias"), 0.0),
    }
    raw_weights = {}
    for name, persona in personas.items():
        tilt = bias_map.get(name, 0.0)
        persona.capital_weight = max(0.01, float(persona.capital_weight) * (1.0 + 0.30 * tilt))
        raw_weights[name] = float(persona.capital_weight)
    total = sum(raw_weights.values()) or 1.0
    normalized = {name: weight / total for name, weight in raw_weights.items()}
    for name, persona in personas.items():
        persona.capital_weight = normalized[name]
    return {f"{name}_weight": round(value, 4) for name, value in normalized.items()}


def _build_branch_conversation(branches: list[dict[str, Any]], llm_content: Mapping[str, Any], simulation: Mapping[str, Any]) -> dict[str, Any]:
    if not branches:
        return {
            "summary": "No branches available.",
            "top_branch": "No branch summary available.",
            "minority_branch": "No minority scenario available.",
            "debate_lines": [],
        }
    ranked = sorted(
        branches,
        key=lambda branch: (
            _safe_float(branch.get("probability"), 0.0) * 0.65
            + _safe_float(branch.get("branch_fitness"), 0.0) * 0.25
            + _safe_float(branch.get("minority_guardrail"), 0.0) * 0.10
        ),
        reverse=True,
    )
    top_branch = ranked[0]
    bullish = [branch for branch in branches if branch.get("predicted_prices", [0])[-1] >= branch.get("predicted_prices", [0])[0]]
    bearish = [branch for branch in branches if branch.get("predicted_prices", [0])[-1] < branch.get("predicted_prices", [0])[0]]
    consensus_bias = str(simulation.get("scenario_bias", "neutral"))
    minority_pool = bearish if consensus_bias == "bullish" else bullish
    minority_ranked = sorted(
        minority_pool,
        key=lambda branch: (
            _safe_float(branch.get("minority_guardrail"), 0.0) * 0.55
            + _safe_float(branch.get("probability"), 0.0) * 0.30
            + _safe_float(branch.get("branch_fitness"), 0.0) * 0.15
        ),
        reverse=True,
    )
    minority_branch = minority_ranked[0] if minority_ranked else branches[min(len(branches) - 1, 1)]
    if minority_branch.get("path_id") == top_branch.get("path_id") and len(ranked) > 1:
        minority_branch = ranked[-1]
    narrative = str(llm_content.get("dominant_narrative", "mixed narrative")).strip() or "mixed narrative"
    explanation = str(llm_content.get("explanation", "Branch disagreement remains interpretable.")).strip()
    macro = str(llm_content.get("macro_thesis", "macro view unavailable")).strip()
    supporting = ranked[1:4]
    debate_lines = [
        f"Institutional: {macro}. Branch {top_branch.get('path_id')} is strongest because it fits the current regime best.",
        f"Retail: {narrative}. I chase branch {top_branch.get('path_id')} because the crowd reads it as continuation.",
        f"Whale: Branch {minority_branch.get('path_id')} is the trap scenario I preserve in case the crowd gets too one-sided.",
        f"Algo: Branch {top_branch.get('path_id')} has the cleanest structure, while branch {minority_branch.get('path_id')} is the main invalidation path.",
    ]
    return {
        "summary": explanation,
        "top_branch": f"Branch {top_branch.get('path_id')} | driver={top_branch.get('dominant_driver')} | persona={top_branch.get('dominant_persona')} | label={top_branch.get('branch_label')}",
        "minority_branch": f"Branch {minority_branch.get('path_id')} | driver={minority_branch.get('dominant_driver')} | persona={minority_branch.get('dominant_persona')} | label={minority_branch.get('branch_label')}",
        "supporting_branches": [
            f"Branch {branch.get('path_id')} | prob={_safe_float(branch.get('probability'), 0.0):.3f} | label={branch.get('branch_label')}"
            for branch in supporting
        ],
        "top_branch_id": top_branch.get("path_id"),
        "minority_branch_id": minority_branch.get("path_id"),
        "debate_lines": debate_lines,
    }


def _fallback_swarm_judge(bot_swarm: Mapping[str, Any], simulation: Mapping[str, Any], technical_analysis: Mapping[str, Any]) -> dict[str, Any]:
    bots = list(bot_swarm.get("bots", []) or [])
    aggregate = bot_swarm.get("aggregate", {}) if isinstance(bot_swarm, Mapping) else {}
    top_bot = max(bots, key=lambda item: _safe_float(item.get("confidence"), 0.0), default={})
    weakest_bot = min(bots, key=lambda item: _safe_float(item.get("confidence"), 0.0), default={})
    bias = str(aggregate.get("signal", simulation.get("scenario_bias", "neutral")))
    location = str(technical_analysis.get("location", "balanced")).lower()
    master_confidence = _safe_float(aggregate.get("confidence"), _safe_float(simulation.get("overall_confidence"), 0.0))
    return {
        "available": False,
        "content": {
            "master_bias": bias,
            "master_confidence": round(master_confidence, 6),
            "manual_stance": "buy" if bias == "bullish" and master_confidence >= 0.58 else "sell" if bias == "bearish" and master_confidence >= 0.58 else "hold",
            "manual_action_reason": f"Use the simulator bias as a manual stance only while structure remains in {location}.",
            "crowd_emotion": "compressed conviction" if master_confidence > 0.55 else "mixed emotion",
            "crowd_lean": f"crowd leans {bias}",
            "discussion_takeaway": "Public reaction is still mixed and should be treated as a sentiment helper, not a decisive signal.",
            "top_bot": str(top_bot.get("name", "unknown")),
            "weakest_bot": str(weakest_bot.get("name", "unknown")),
            "judge_summary": f"The bot swarm and simulator lean {bias}, with market structure currently in {location}.",
            "debate_lines": [
                f"Judge: strongest specialist right now is {top_bot.get('name', 'unknown')}.",
                f"Judge: weakest conviction comes from {weakest_bot.get('name', 'unknown')}.",
                f"Judge: simulator bias is {simulation.get('scenario_bias', 'neutral')}.",
                f"Judge: structural location is {location}.",
            ],
            "public_reaction_lines": [
                "Retail chat would likely chase the visible move if price keeps extending.",
                "Institutional-style readers would wait for structure confirmation around the active zone.",
                "Contrarian participants would watch the minority scenario for a trap or squeeze failure.",
            ],
            "minority_case": "Minority case survives through the weaker branches and skeptical bot layer.",
            "actionable_structure": f"Order-block and equilibrium context point to a {location} location test.",
        },
    }


def _build_branch_graph(branches: list[dict[str, Any]], highlighted_branches: Mapping[str, Any], anchor_price: float) -> dict[str, Any]:
    top_branch_id = highlighted_branches.get("top_branch_id")
    minority_branch_id = highlighted_branches.get("minority_branch_id")
    traces = []
    for branch in branches[:12]:
        branch_id = branch.get("path_id")
        prices = [anchor_price] + list(branch.get("predicted_prices", []))
        timestamps = [None] + list(branch.get("timestamps", []))
        traces.append(
            {
                "path_id": branch_id,
                "timestamps": timestamps,
                "prices": prices,
                "probability": branch.get("probability"),
                "branch_fitness": branch.get("branch_fitness"),
                "branch_label": branch.get("branch_label"),
                "highlight": "top" if branch_id == top_branch_id else "minority" if branch_id == minority_branch_id else "support",
            }
        )
    return {"traces": traces}


def _llm_price_tilt(current_price: float, atr: float, llm_content: Mapping[str, Any], minutes: int) -> float:
    llm_bias = (_llm_numeric_prior(llm_content) - 0.5) * 2.0
    horizon_scale = {5: 0.35, 10: 0.55, 15: 0.75, 20: 0.90, 25: 1.05, 30: 1.25}.get(minutes, 0.55)
    return float(current_price + (atr * horizon_scale * llm_bias))


def _build_final_forecast(
    payload: Mapping[str, Any],
    bot_swarm: Mapping[str, Any],
    llm_content: Mapping[str, Any],
    model_prediction: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    market = payload.get("market", {}) if isinstance(payload, Mapping) else {}
    current_price = _safe_float(market.get("current_price"), 0.0)
    atr = max(_safe_float(payload.get("current_row", {}).get("atr_14"), current_price * 0.0015), 0.25)
    cone = list(payload.get("cone", []) if isinstance(payload, Mapping) else [])
    bot_horizons = {int(item.get("minutes")): item for item in (bot_swarm.get("aggregate", {}).get("horizon_predictions", []) if isinstance(bot_swarm, Mapping) else [])}
    model_probability = _safe_float((model_prediction or {}).get("bullish_probability"), 0.5)
    model_bias = (model_probability - 0.5) * 2.0
    points = []
    for point in cone:
        minutes = int(_safe_float(point.get("horizon"), 1.0) * 5)
        branch_center = _safe_float(point.get("center_price"), current_price)
        bot_target = _safe_float(bot_horizons.get(minutes, {}).get("target_price"), branch_center)
        llm_target = _llm_price_tilt(current_price, atr, llm_content, minutes)
        model_target = current_price + (atr * {5: 0.22, 10: 0.36, 15: 0.52, 20: 0.68, 25: 0.84, 30: 1.00}.get(minutes, 0.36) * model_bias)
        final_price = (0.52 * branch_center) + (0.23 * bot_target) + (0.15 * llm_target) + (0.10 * model_target)
        points.append(
            {
                "minutes": minutes,
                "timestamp": point.get("timestamp"),
                "branch_center": round(branch_center, 5),
                "bot_target": round(bot_target, 5),
                "llm_target": round(llm_target, 5),
                "final_price": round(float(final_price), 5),
            }
        )
    horizon_table = [
        {
            "minutes": item["minutes"],
            "final_price": item["final_price"],
            "branch_center": item["branch_center"],
            "bot_target": item["bot_target"],
            "llm_target": item["llm_target"],
        }
        for item in points
        if item["minutes"] in {5, 10, 15, 30}
    ]
    return {"points": points, "horizon_table": horizon_table}


def _normalized_weights_from_leaves(leaves) -> np.ndarray:
    weights = np.asarray([max(float(leaf.probability_weight), 1e-9) for leaf in leaves], dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        return np.full(len(weights), 1.0 / max(1, len(weights)), dtype=np.float64)
    return weights / total


def _price_cone_from_leaves(leaves, last_timestamp: pd.Timestamp, last_price: float, step_minutes: int = 5) -> list[dict[str, Any]]:
    if not leaves:
        return []
    weights = _normalized_weights_from_leaves(leaves)
    max_horizon = max(len(leaf.path_prices or []) for leaf in leaves)
    if max_horizon <= 0:
        return []

    points: list[dict[str, Any]] = []
    for horizon_index in range(max_horizon):
        horizon_prices = []
        horizon_weights = []
        for weight, leaf in zip(weights, leaves):
            prices = leaf.path_prices or []
            if horizon_index < len(prices):
                horizon_prices.append(float(prices[horizon_index]))
                horizon_weights.append(float(weight))
        if not horizon_prices:
            continue
        price_array = np.asarray(horizon_prices, dtype=np.float64)
        weight_array = np.asarray(horizon_weights, dtype=np.float64)
        weight_array = weight_array / weight_array.sum()
        center_price = float(np.average(price_array, weights=weight_array))
        variance = float(np.average((price_array - center_price) ** 2, weights=weight_array))
        std_dev = variance ** 0.5
        lower_price = max(0.0, center_price - std_dev)
        upper_price = max(center_price, center_price + std_dev)
        center_probability = 0.5 if last_price == 0.0 else float(np.clip(0.5 + ((center_price - last_price) / max(last_price, 1e-6)) * 12.0, 0.0, 1.0))
        lower_probability = 0.5 if last_price == 0.0 else float(np.clip(0.5 + ((lower_price - last_price) / max(last_price, 1e-6)) * 12.0, 0.0, 1.0))
        upper_probability = 0.5 if last_price == 0.0 else float(np.clip(0.5 + ((upper_price - last_price) / max(last_price, 1e-6)) * 12.0, 0.0, 1.0))
        horizon = horizon_index + 1
        points.append(
            {
                "horizon": horizon,
                "timestamp": (last_timestamp + pd.Timedelta(minutes=step_minutes * horizon)).isoformat(),
                "center_probability": round(center_probability, 6),
                "lower_probability": round(lower_probability, 6),
                "upper_probability": round(upper_probability, 6),
                "center_price": round(center_price, 5),
                "lower_price": round(lower_price, 5),
                "upper_price": round(upper_price, 5),
            }
        )
    return points


def _candles_to_records(price_frame: pd.DataFrame, limit: int = 240) -> list[dict[str, Any]]:
    return [
        {
            "timestamp": index.isoformat(),
            "open": round(float(row["open"]), 5),
            "high": round(float(row["high"]), 5),
            "low": round(float(row["low"]), 5),
            "close": round(float(row["close"]), 5),
            "volume": round(float(row["volume"]), 2),
        }
        for index, row in price_frame.tail(limit).iterrows()
    ]


def build_realtime_chart_payload(symbol: str, bars: int = 240) -> dict[str, Any]:
    try:
        frame = fetch_recent_market_candles(symbol, interval="1m", range_="1d", ttl_seconds=5)
    except Exception:
        frame = pd.DataFrame()
    interval = "1m"
    if frame.empty:
        try:
            frame = fetch_recent_market_candles(symbol, interval="5m", range_="5d", ttl_seconds=15)
            interval = "5m"
        except Exception:
            frame = pd.DataFrame()
    if frame.empty:
        return {"symbol": str(symbol).upper(), "interval": interval, "candles": []}
    return {
        "symbol": str(symbol).upper(),
        "interval": interval,
        "source": str(frame.attrs.get("market_source", "unknown")),
        "candles": _candles_to_records(frame, limit=bars),
    }


def build_fast_dashboard_payload(symbol: str) -> dict[str, Any]:
    candles = fetch_recent_market_candles(symbol, interval="5m", range_="5d", ttl_seconds=15)
    if candles.empty:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "symbol": str(symbol).upper(),
            "market": {"candles": [], "current_price": 0.0, "source": "fast_dashboard_proxy"},
            "simulation": {},
            "cone": [],
            "branches": [],
            "highlighted_branches": {"top_branch_id": 1, "minority_branch_id": 2},
            "current_row": {},
            "technical_analysis": {},
            "feeds": {"news": [], "public_discussions": [], "fear_greed": {}, "macro": {}},
            "bot_swarm": {"aggregate": {}, "bots": [], "persona_reactions": {}},
            "llm_context": {"available": False, "provider": "fast_dashboard_proxy", "content": {}},
            "wltc": {},
            "mmm": {},
            "mfg": {},
        }

    frame = candles.copy().tail(480)
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"]).copy()
    frame["volume"] = frame["volume"].fillna(0.0)

    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    open_ = frame["open"]
    volume = frame["volume"]
    prev_close = close.shift(1).fillna(close)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_series = tr.ewm(alpha=1 / 14, adjust=False, min_periods=1).mean()
    rsi_series = _compute_rsi(close, 14)
    ema_fast = close.ewm(span=21, adjust=False).mean()
    ema_slow = close.ewm(span=55, adjust=False).mean()
    ema_cross_series = ((ema_fast - ema_slow) / close.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    equilibrium_series = ((high.rolling(20, min_periods=1).max()) + (low.rolling(20, min_periods=1).min())) / 2.0
    returns = close.pct_change().fillna(0.0)
    momentum_series = returns.rolling(6, min_periods=1).sum()
    volume_ma = volume.rolling(20, min_periods=1).mean().replace(0.0, np.nan)
    volume_ratio_series = (volume / volume_ma).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    current_price = _safe_float(close.iloc[-1], 0.0)
    last_time = frame.index[-1]
    atr = max(_safe_float(atr_series.iloc[-1], current_price * 0.0012), 0.25)
    rsi = _safe_float(rsi_series.iloc[-1], 50.0)
    ema_cross = _safe_float(ema_cross_series.iloc[-1], 0.0)
    equilibrium = _safe_float(equilibrium_series.iloc[-1], current_price)
    volume_ratio = _safe_float(volume_ratio_series.iloc[-1], 1.0)
    price_stretch = float(np.tanh((current_price - equilibrium) / max(atr * 2.0, 1e-9)))
    trend_strength = float(np.tanh((ema_fast.iloc[-1] - ema_slow.iloc[-1]) / max(atr * 2.0, 1e-9)))
    momentum = float(np.tanh(_safe_float(momentum_series.iloc[-1], 0.0) * 45.0))

    hurst_overall = float(np.clip(0.5 + (0.18 * trend_strength) + (0.07 * momentum), 0.28, 0.74))
    hurst_positive = float(np.clip(hurst_overall + (0.06 if trend_strength >= 0.0 else -0.03), 0.25, 0.82))
    hurst_negative = float(np.clip(hurst_overall + (-0.03 if trend_strength >= 0.0 else 0.06), 0.25, 0.82))
    hurst_asymmetry = float(np.clip(hurst_positive - hurst_negative, -0.25, 0.25))

    directional_bias = float(
        np.clip(
            (0.44 * trend_strength)
            + (0.24 * np.tanh((rsi - 50.0) / 9.0))
            + (0.18 * price_stretch)
            + (0.14 * momentum),
            -0.95,
            0.95,
        )
    )
    confidence = float(
        np.clip(
            0.30
            + (0.36 * abs(directional_bias))
            + (0.08 * abs(hurst_asymmetry))
            + (0.06 * min(volume_ratio, 2.0))
            - (0.08 * abs(price_stretch)),
            0.24,
            0.88,
        )
    )
    scenario_bias = "bullish" if directional_bias > 0.03 else "bearish" if directional_bias < -0.03 else "neutral"
    direction = "BUY" if directional_bias > 0.08 else "SELL" if directional_bias < -0.08 else "HOLD"
    disagreement = float(np.clip(0.10 + (0.32 * (1.0 - confidence)), 0.10, 0.46))

    recent_window = frame.tail(48)
    lower_candidates = recent_window[recent_window["low"] <= current_price]["low"]
    upper_candidates = recent_window[recent_window["high"] >= current_price]["high"]
    support_price = _safe_float(lower_candidates.tail(12).mean(), _safe_float(low.tail(12).min(), current_price - atr))
    resistance_price = _safe_float(upper_candidates.tail(12).mean(), _safe_float(high.tail(12).max(), current_price + atr))

    order_block_rows = recent_window.assign(
        body=(recent_window["close"] - recent_window["open"]).abs(),
        volume_ratio=(volume_ratio_series.tail(len(recent_window)).values),
    ).sort_values(["volume_ratio", "body"], ascending=[False, False]).head(4)
    order_blocks = [
        {
            "low": round(float(min(row["low"], row["open"], row["close"])), 5),
            "high": round(float(max(row["high"], row["open"], row["close"])), 5),
            "strength": round(float(np.clip(row.get("volume_ratio", 1.0) / 2.0, 0.2, 1.0)), 4),
        }
        for _, row in order_block_rows.iterrows()
    ]

    fair_value_gaps: list[dict[str, Any]] = []
    tail_rows = frame.tail(40).reset_index(drop=False)
    for index in range(2, len(tail_rows)):
        prev_prev = tail_rows.iloc[index - 2]
        current = tail_rows.iloc[index]
        if current["low"] > prev_prev["high"]:
            fair_value_gaps.append(
                {
                    "low": round(float(prev_prev["high"]), 5),
                    "high": round(float(current["low"]), 5),
                    "size": round(float(current["low"] - prev_prev["high"]), 5),
                }
            )
        elif current["high"] < prev_prev["low"]:
            fair_value_gaps.append(
                {
                    "low": round(float(current["high"]), 5),
                    "high": round(float(prev_prev["low"]), 5),
                    "size": round(float(prev_prev["low"] - current["high"]), 5),
                }
            )
        if len(fair_value_gaps) >= 4:
            break

    technical_analysis = {
        "structure": "bullish" if ema_fast.iloc[-1] >= ema_slow.iloc[-1] else "bearish",
        "location": "discount" if current_price <= equilibrium else "premium",
        "rsi_14": round(rsi, 4),
        "atr_14": round(atr, 5),
        "equilibrium": round(equilibrium, 5),
        "nearest_support": {"price": round(support_price, 5), "distance": round(current_price - support_price, 5)},
        "nearest_resistance": {"price": round(resistance_price, 5), "distance": round(resistance_price - current_price, 5)},
        "order_blocks": order_blocks,
        "fair_value_gaps": fair_value_gaps,
        "quant_regime_strength": round(confidence, 6),
        "quant_transition_risk": round(1.0 - confidence, 6),
        "quant_vol_realism": round(float(np.clip(1.0 - abs(price_stretch), 0.0, 1.0)), 6),
        "quant_fair_value_z": round(float((current_price - equilibrium) / max(atr, 1e-9)), 6),
    }

    try:
        news_items = fetch_live_news(symbol, limit=6)
    except Exception:
        news_items = []
    try:
        discussion_items = fetch_public_discussions(symbol, limit=6)
    except Exception:
        discussion_items = []
    try:
        fear_greed = fetch_fear_greed_snapshot()
    except Exception:
        fear_greed = {"value": 50.0, "classification": "neutral", "history": []}
    try:
        macro_context = fetch_macro_context()
    except Exception:
        macro_context = {"macro_bias": 0.0, "macro_shock": 0.0, "driver": "macro_neutral", "components": {}}

    news_scores = np.asarray([_safe_float(item.get("sentiment"), 0.0) for item in news_items], dtype=np.float64)
    discussion_scores = np.asarray([_safe_float(item.get("sentiment"), 0.0) for item in discussion_items], dtype=np.float64)
    news_bias = float(np.clip(news_scores.mean() if news_scores.size else 0.0, -1.0, 1.0))
    news_intensity = float(np.clip(np.abs(news_scores).mean() if news_scores.size else 0.0, 0.0, 1.0))
    fear_value = _safe_float(fear_greed.get("value"), 50.0)
    fear_bias = float(np.tanh((50.0 - fear_value) / 18.0))
    crowd_bias = float(
        np.clip(
            (0.65 * (discussion_scores.mean() if discussion_scores.size else 0.0)) + (0.35 * fear_bias),
            -1.0,
            1.0,
        )
    )
    crowd_extreme = float(
        np.clip(
            max(
                abs(discussion_scores).mean() if discussion_scores.size else 0.0,
                abs(fear_value - 50.0) / 50.0,
            ),
            0.0,
            1.0,
        )
    )
    macro_bias = float(np.clip(_safe_float(macro_context.get("macro_bias"), 0.0), -1.0, 1.0))
    macro_shock = float(np.clip(_safe_float(macro_context.get("macro_shock"), 0.0), 0.0, 1.0))

    retail_t = float(np.clip((0.55 * abs(momentum)) + (0.45 * abs((rsi - 50.0) / 50.0)), 0.0, 1.0))
    institutional_t = float(np.clip((0.62 * abs(trend_strength)) + (0.20 * min(volume_ratio / 2.0, 1.0)) + (0.18 * abs(price_stretch)), 0.0, 1.0))
    noise_t = float(np.clip(0.45 * disagreement + 0.20 * abs(price_stretch), 0.0, 1.0))
    wltc_context = {
        "retail": {"testosterone_index": round(retail_t, 4), "fundamental_tracking": round(float(np.clip(1.0 - retail_t, 0.05, 1.0)), 4)},
        "institutional": {"testosterone_index": round(institutional_t, 4), "fundamental_tracking": round(float(np.clip(1.0 - institutional_t * 0.65, 0.12, 1.0)), 4)},
        "noise": {"testosterone_index": round(noise_t, 4), "fundamental_tracking": round(float(np.clip(1.0 - noise_t, 0.05, 1.0)), 4)},
    }
    mmm_context = {
        "hurst_overall": round(hurst_overall, 4),
        "hurst_positive": round(hurst_positive, 4),
        "hurst_negative": round(hurst_negative, 4),
        "hurst_asymmetry": round(hurst_asymmetry, 4),
        "market_memory_regime": "trend_following" if abs(trend_strength) >= 0.12 else "balanced_range",
    }
    mfg_context = {
        "disagreement": round(disagreement, 8),
        "consensus_drift": round(float(directional_bias * atr / max(current_price, 1e-9)), 8),
    }

    recent_bars = [
        {
            "timestamp": index.isoformat(),
            "open": round(float(row["open"]), 5),
            "high": round(float(row["high"]), 5),
            "low": round(float(row["low"]), 5),
            "close": round(float(row["close"]), 5),
            "volume": round(float(row["volume"]), 2),
        }
        for index, row in frame.tail(72).iterrows()
    ]
    horizon_scales = [0.45, 0.85, 1.15]
    future_points: list[dict[str, Any]] = []
    consensus_prices = [round(current_price, 5)]
    minority_prices = [round(current_price, 5)]
    for index, scale in enumerate(horizon_scales, start=1):
        drift = directional_bias * atr * scale
        center_price = current_price + drift
        minority_bias = float(np.clip((0.55 * price_stretch) - (0.30 * directional_bias) + (0.15 * macro_bias) - (0.10 * news_bias), -0.60, 0.60))
        minority_center = current_price + (atr * scale * minority_bias)
        if abs(minority_center - center_price) < (0.08 * atr):
            minority_center = current_price + (atr * scale * (minority_bias - (0.18 * np.sign(directional_bias or 1.0))))
        band = atr * (0.78 + ((1.0 - confidence) * 1.10) + (0.15 * index))
        future_ts = (last_time + pd.Timedelta(minutes=5 * index)).isoformat()
        future_points.append(
            {
                "timestamp": future_ts,
                "horizon": index,
                "center_price": round(center_price, 5),
                "lower_price": round(center_price - band, 5),
                "upper_price": round(center_price + band, 5),
            }
        )
        consensus_prices.append(round(center_price, 5))
        minority_prices.append(round(minority_center, 5))

    top_branch = {
        "path_id": 1,
        "probability": round(float(0.5 + (0.5 * directional_bias)), 6),
        "selector_score": round(confidence, 6),
        "predicted_prices": [point["center_price"] for point in future_points],
        "timestamps": [point["timestamp"] for point in future_points],
        "dominant_persona": "institutional" if abs(trend_strength) >= 0.12 else "mixed",
        "dominant_driver": "fast_dashboard_proxy",
        "branch_label": "consensus_path",
    }
    minority_branch = {
        "path_id": 2,
        "probability": round(float(1.0 - top_branch["probability"]), 6),
        "selector_score": round(float(np.clip(1.0 - confidence, 0.05, 0.95)), 6),
        "predicted_prices": minority_prices[1:],
        "timestamps": [point["timestamp"] for point in future_points],
        "dominant_persona": "retail",
        "dominant_driver": "minority_case",
        "branch_label": "minority_path",
    }

    testosterone_index = {
        name: round(_safe_float(summary.get("testosterone_index"), 0.0), 4)
        for name, summary in wltc_context.items()
    }
    market_source = str(frame.attrs.get("market_source", candles.attrs.get("market_source", "fast_dashboard_proxy")))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": str(symbol).upper(),
        "market": {
            "candles": _candles_to_records(frame, limit=240),
            "current_price": round(current_price, 5),
            "source": market_source,
        },
        "simulation": {
            "mean_probability": round(float(0.5 + (0.5 * directional_bias)), 6),
            "consensus_score": round(confidence * 0.72, 6),
            "overall_confidence": round(confidence, 6),
            "dominant_driver": "fast_dashboard_proxy",
            "scenario_bias": scenario_bias,
            "analog_bias": round(price_stretch, 6),
            "analog_confidence": round(confidence * 0.68, 6),
            "analog_support": 72,
            "memory_bank_bias": round(directional_bias * 0.60, 6),
            "memory_bank_confidence": round(confidence * 0.54, 6),
            "branch_count": 2,
            "detected_regime": str(mmm_context.get("market_memory_regime", "balanced_range")),
            "regime_confidence": round(confidence * 0.78, 6),
            "retrieval_similarity": round(0.52 + (0.18 * abs(directional_bias)), 6),
            "selector_top_branch_id": 1,
            "selector_top_score": round(confidence, 6),
            "contradiction_type": "agreement_bull" if direction == "BUY" else "agreement_bear" if direction == "SELL" else "balanced",
            "contradiction_confidence": round(1.0 - disagreement, 6),
            "cone_treatment": "normal_cone",
            "hurst_overall": round(hurst_overall, 6),
            "hurst_positive": round(hurst_positive, 6),
            "hurst_negative": round(hurst_negative, 6),
            "hurst_asymmetry": round(hurst_asymmetry, 6),
            "market_memory_regime": str(mmm_context.get("market_memory_regime", "balanced_range")),
            "mfg_disagreement": round(_safe_float(mfg_context.get("disagreement"), 0.0), 8),
            "mfg_consensus_drift": round(_safe_float(mfg_context.get("consensus_drift"), 0.0), 8),
            "testosterone_index": testosterone_index,
            "macro_bias": round(macro_bias, 6),
            "news_bias": round(news_bias, 6),
            "crowd_bias": round(crowd_bias, 6),
            "crowd_extreme": round(crowd_extreme, 6),
        },
        "cone": future_points,
        "branches": [top_branch, minority_branch],
        "highlighted_branches": {"top_branch_id": 1, "minority_branch_id": 2},
        "current_row": {
            "close": round(current_price, 5),
            "atr_14": round(atr, 5),
            "rsi_14": round(rsi, 4),
            "ema_cross": round(ema_cross, 6),
            "volume_ratio": round(volume_ratio, 6),
            "hurst_overall": round(hurst_overall, 6),
            "hurst_positive": round(hurst_positive, 6),
            "hurst_negative": round(hurst_negative, 6),
            "hurst_asymmetry": round(hurst_asymmetry, 6),
            "wltc_testosterone_retail": round(retail_t, 6),
            "wltc_testosterone_noise": round(noise_t, 6),
            "wltc_fundamental_tracking_retail": round(float(wltc_context["retail"]["fundamental_tracking"]), 6),
            "wltc_fundamental_tracking_institutional": round(float(wltc_context["institutional"]["fundamental_tracking"]), 6),
            "macro_bias": round(macro_bias, 6),
            "macro_shock": round(macro_shock, 6),
            "news_bias": round(news_bias, 6),
            "news_intensity": round(news_intensity, 6),
            "crowd_bias": round(crowd_bias, 6),
            "crowd_extreme": round(crowd_extreme, 6),
        },
        "technical_analysis": technical_analysis,
        "feeds": {
            "news": news_items,
            "public_discussions": discussion_items,
            "fear_greed": fear_greed,
            "macro": macro_context,
        },
        "bot_swarm": {
            "aggregate": {
                "signal": "bullish" if direction == "BUY" else "bearish" if direction == "SELL" else "neutral",
                "bullish_probability": round(float(0.5 + (0.5 * directional_bias)), 6),
                "bearish_probability": round(float(0.5 - (0.5 * directional_bias)), 6),
                "confidence": round(confidence * 0.88, 6),
                "disagreement": round(disagreement, 6),
                "horizon_predictions": [
                    {
                        "minutes": index * 5,
                        "bullish_probability": round(float(0.5 + (0.5 * directional_bias)), 6),
                        "target_price": point["center_price"],
                    }
                    for index, point in enumerate(future_points, start=1)
                ],
            },
            "bots": [],
            "persona_reactions": {},
        },
        "llm_context": {"available": False, "provider": "fast_dashboard_proxy", "content": {}},
        "wltc": wltc_context,
        "mmm": mmm_context,
        "mfg": mfg_context,
        "recent_bars": recent_bars,
    }


def _read_simulation_history(limit: int = 200) -> list[dict[str, Any]]:
    if not LIVE_SIMULATION_HISTORY_PATH.exists():
        return []
    try:
        payload = json.loads(LIVE_SIMULATION_HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return payload[-limit:]


def _write_simulation_history(entries: list[dict[str, Any]]) -> None:
    LIVE_SIMULATION_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    LIVE_SIMULATION_HISTORY_PATH.write_text(json.dumps(entries[-200:], indent=2), encoding="utf-8")


def build_history_entry(payload: dict[str, Any], model_prediction: dict[str, Any] | None = None) -> dict[str, Any]:
    market = payload.get("market", {})
    simulation = payload.get("simulation", {})
    candles = list(market.get("candles", []))
    anchor_timestamp = candles[-1]["timestamp"] if candles else payload.get("generated_at")
    anchor_price = _safe_float(market.get("current_price"), 0.0)
    cone = _cone_points(payload.get("cone", []))
    entry = {
        "symbol": payload.get("symbol", "XAUUSD"),
        "generated_at": payload.get("generated_at"),
        "simulation_version": SIMULATION_SCHEMA_VERSION,
        "anchor_timestamp": anchor_timestamp,
        "anchor_price": anchor_price,
        "market_source": market.get("source", "unknown"),
        "scenario_bias": simulation.get("scenario_bias", "neutral"),
        "overall_confidence": _safe_float(simulation.get("overall_confidence"), 0.0),
        "consensus_score": _safe_float(simulation.get("consensus_score"), 0.0),
        "uncertainty_width": _safe_float(simulation.get("uncertainty_width"), 0.0),
        "dominant_driver": simulation.get("dominant_driver", "unknown"),
        "cone": cone,
        "center_path": [
            {"timestamp": point.get("timestamp"), "price": _safe_float(point.get("center_price"), anchor_price)}
            for point in cone
        ],
        "confidence_tier": simulation.get("confidence_tier", "uncertain"),
        "tier_color": simulation.get("tier_color", "#d50000"),
        "tier_label": simulation.get("tier_label", "UNCERTAIN - observe only"),
        "direction": simulation.get("direction", "NEUTRAL"),
        "cabr_score": _safe_float(simulation.get("cabr_score"), 0.5),
        "bst_score": _safe_float(simulation.get("bst_score"), 0.5),
        "cone_width_pips": _safe_float(simulation.get("cone_width_pips"), 0.0),
        "cpm_score": _safe_float(simulation.get("cpm_score"), 0.5),
        "crowd_persona": simulation.get("crowd_persona", "mixed"),
        "minority_path": list(simulation.get("minority_path", [])),
        "minority_direction": simulation.get("minority_direction", "NEUTRAL"),
        "mode": simulation.get("mode", "frequency"),
        "primary_horizon_minutes": int(_safe_float(simulation.get("primary_horizon_minutes"), 15)),
        "sqt_label": simulation.get("sqt_label", "NEUTRAL"),
        "sqt_accuracy": _safe_float(simulation.get("sqt_accuracy"), 0.5),
        "eci": dict(payload.get("eci", {})),
        "relativistic_cone": dict(payload.get("relativistic_cone", {})),
        "wltc": dict(payload.get("wltc", {})),
        "mmm": dict(payload.get("mmm", {})),
        "model_prediction": model_prediction or {},
    }
    return entry


def record_simulation_history(payload: dict[str, Any], model_prediction: dict[str, Any] | None = None) -> dict[str, Any]:
    entry = build_history_entry(payload, model_prediction=model_prediction)
    history = _read_simulation_history(limit=400)
    history.append(entry)
    history = history[-200:]
    _write_simulation_history(history)
    return entry


def _history_entries_for_symbol(symbol: str, limit: int = 20) -> list[dict[str, Any]]:
    upper = symbol.upper()
    return [entry for entry in _read_simulation_history(limit=200) if str(entry.get("symbol", "")).upper() == upper][-limit:]


def build_simulation_comparison(symbol: str, candles: pd.DataFrame, history_limit: int = 12) -> dict[str, Any]:
    current_source = str(candles.attrs.get("market_source", "unknown"))
    history = [
        entry
        for entry in _history_entries_for_symbol(symbol, limit=history_limit * 4)
        if str(entry.get("market_source", "unknown")) == current_source
        and str(entry.get("simulation_version", "")) == SIMULATION_SCHEMA_VERSION
    ][-history_limit:]
    candle_records = _candles_to_records(candles, limit=240)
    if not history:
        return {
            "live_market": candle_records,
            "active_prediction": None,
            "recent_simulations": [],
            "server_timestamp": datetime.now(timezone.utc).isoformat(),
            "market_source": current_source,
        }

    recent_summaries: list[dict[str, Any]] = []
    active_payload: dict[str, Any] | None = None
    for position, entry in enumerate(reversed(history), start=1):
        anchor_timestamp = pd.to_datetime(entry.get("anchor_timestamp"), utc=True, errors="coerce")
        if pd.isna(anchor_timestamp):
            continue
        actual = candles[candles.index > anchor_timestamp].copy()
        predicted_center = list(entry.get("center_path", []))
        cone = _cone_points(entry.get("cone", []))
        actual_future = []
        matched = 0
        hits = 0
        for prediction, (_, actual_row) in zip(predicted_center, actual.iterrows()):
            actual_close = _safe_float(actual_row.get("close"), 0.0)
            cone_point = cone[matched] if matched < len(cone) else {}
            lower = _safe_float(cone_point.get("lower_price"), actual_close)
            upper = _safe_float(cone_point.get("upper_price"), actual_close)
            inside = lower <= actual_close <= upper
            hits += 1 if inside else 0
            matched += 1
            actual_future.append(
                {
                    "timestamp": actual_row.name.isoformat(),
                    "close": round(actual_close, 5),
                    "inside_cone": inside,
                    "predicted_center": _safe_float(prediction.get("price"), actual_close),
                }
            )

        last_actual_close = _safe_float(actual_future[-1]["close"], entry.get("anchor_price", 0.0)) if actual_future else _safe_float(entry.get("anchor_price"), 0.0)
        realized_direction = "bullish" if last_actual_close >= _safe_float(entry.get("anchor_price"), 0.0) else "bearish"
        direction_match = realized_direction == str(entry.get("scenario_bias", "neutral"))
        minority_path = list(entry.get("minority_path", []))
        minority_terminal = _safe_float(minority_path[-1], entry.get("anchor_price", 0.0)) if minority_path else _safe_float(entry.get("anchor_price"), 0.0)
        center_terminal = _safe_float(entry.get("center_path", [{}])[-1].get("price"), entry.get("anchor_price", 0.0)) if entry.get("center_path") else _safe_float(entry.get("anchor_price"), 0.0)
        minority_was_closer = abs(last_actual_close - minority_terminal) < abs(last_actual_close - center_terminal) if matched else None
        summary = {
            "generated_at": entry.get("generated_at"),
            "anchor_timestamp": entry.get("anchor_timestamp"),
            "scenario_bias": entry.get("scenario_bias"),
            "realized_direction": realized_direction if matched else "pending",
            "direction_match": direction_match if matched else None,
            "matched_points": matched,
            "hit_rate": round(hits / matched, 4) if matched else None,
            "overall_confidence": _safe_float(entry.get("overall_confidence"), 0.0),
            "dominant_driver": entry.get("dominant_driver", "unknown"),
            "confidence_tier": entry.get("confidence_tier", "uncertain"),
            "tier_color": entry.get("tier_color", "#d50000"),
            "sqt_label": entry.get("sqt_label", "NEUTRAL"),
            "minority_was_closer": minority_was_closer,
        }
        recent_summaries.append(summary)

        if position == 1:
            active_payload = {
                "simulation": entry,
                "actual_future": actual_future,
                "matched_points": matched,
                "hit_rate": round(hits / matched, 4) if matched else None,
                "direction_match": direction_match if matched else None,
                "realized_direction": realized_direction if matched else "pending",
                "minority_was_closer": minority_was_closer,
            }

    return {
        "live_market": candle_records,
        "active_prediction": active_payload,
        "recent_simulations": recent_summaries,
        "server_timestamp": datetime.now(timezone.utc).isoformat(),
        "market_source": current_source,
    }


def _branch_payload(
    price_frame: pd.DataFrame,
    current_row: dict[str, Any],
    symbol: str,
    personas: dict[str, Any],
    llm_content: Mapping[str, Any],
    *,
    persona_weights: Mapping[str, float] | None = None,
    memory_bank_context: Mapping[str, Any] | None = None,
    contradiction_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    seed_state = simulate_one_step(current_row=current_row, personas=personas, seed=42)
    persona_snapshot = _build_persona_snapshot(persona_vote_breakdown(seed_state))
    weighted_persona_bias = _weighted_persona_bias(persona_snapshot, persona_weights or {})
    analog_scorer = get_historical_analog_scorer()
    analog_history = (
        price_frame[PRICE_FEATURE_COLUMNS]
        .tail(analog_scorer.window_size)
        .astype(float)
        .to_dict(orient="records")
    )
    analog_snapshot = analog_scorer.score_window(analog_history)
    current_row["analog_bias"] = float(analog_snapshot.directional_bias)
    current_row["analog_confidence"] = float(analog_snapshot.confidence)
    current_row["analog_support"] = int(analog_snapshot.support)
    root = expand_binary_tree(current_row, personas, max_depth=6, analog_scorer=analog_scorer, history_rows=analog_history)
    leaves = iter_leaves(root)
    collapse = reverse_collapse(leaves)

    last_timestamp = price_frame.index[-1]
    last_price = float(price_frame.iloc[-1]["close"])
    regime = detect_regime(current_row)
    envelopes = build_volatility_envelopes(last_price, current_row, regime, horizons=(5, 15, 30))
    try:
        retrieval = get_historical_path_retriever().retrieve(current_row)
    except Exception:
        retrieval = None
    atr = max(float(current_row.get("atr_14", 0.0) or 0.0), max(last_price * 0.0015, 0.5))
    ranked_leaves = sorted(
        leaves,
        key=lambda item: (float(item.probability_weight) * 0.62)
        + (float(item.branch_fitness) * 0.18)
        + (float(item.analog_confidence) * 0.12)
        + (float(item.minority_guardrail) * 0.08),
        reverse=True,
    )
    branches = []
    for index, leaf in enumerate(ranked_leaves, start=1):
        prices = leaf.path_prices or [last_price]
        timestamps = [(last_timestamp + pd.Timedelta(minutes=5 * step)).isoformat() for step in range(1, len(prices) + 1)]
        branches.append(
            {
                "path_id": index,
                "probability": round(float(leaf.probability_weight), 6),
                "branch_fitness": round(float(leaf.branch_fitness), 6),
                "minority_guardrail": round(float(leaf.minority_guardrail), 6),
                "analog_bias": round(float(leaf.analog_bias), 6),
                "analog_confidence": round(float(leaf.analog_confidence), 6),
                "analog_support": int(leaf.analog_support),
                "branch_label": str(leaf.branch_label),
                "predicted_prices": [round(float(price), 5) for price in prices],
                "timestamps": timestamps,
                "dominant_persona": dominant_persona_name(leaf.state) if leaf.state is not None else "unknown",
                "dominant_driver": leaf.dominant_driver,
            }
        )
    selector_result = rank_branches_with_selector(branches, current_row, regime, envelopes, retrieval)
    selector_score_map = {entry["path_id"]: entry["selector_score"] for entry in selector_result.rationale}
    selector_rationale_map = {entry["path_id"]: entry["top_drivers"] for entry in selector_result.rationale}
    contradiction_type = str((contradiction_context or {}).get("type", "none"))
    contradiction_conf = _safe_float((contradiction_context or {}).get("confidence"), 0.0)
    contradiction_long_prob = _safe_float((contradiction_context or {}).get("long_bias_probability"), 0.5)
    contradiction_scale_map = {
        "agreement_bull": 1.02,
        "agreement_bear": 1.02,
        "short_term_contrary": 0.96,
        "long_term_contrary": 0.95,
        "full_disagreement": 0.86,
    }
    for branch in branches:
        path_id = branch.get("path_id")
        branch["selector_score"] = round(float(selector_score_map.get(path_id, 0.0)), 6)
        branch["selector_rationale"] = selector_rationale_map.get(path_id, [])
        branch_direction = 1.0 if _safe_float(branch.get("predicted_prices", [last_price])[-1], last_price) >= last_price else -1.0
        memory_alignment = _alignment_score(_safe_float((memory_bank_context or {}).get("bullish_probability"), 0.5), branch_direction)
        persona_alignment = _alignment_score(0.5 + 0.5 * weighted_persona_bias, branch_direction)
        contradiction_alignment = _alignment_score(contradiction_long_prob, branch_direction)
        branch["memory_bank_alignment"] = round(float(memory_alignment), 6)
        branch["persona_alignment"] = round(float(persona_alignment), 6)
        branch["contradiction_alignment"] = round(float(contradiction_alignment), 6)
        branch["v9_live_score"] = round(
            float(
                (
                    0.42 * _safe_float(branch.get("selector_score"), 0.0)
                    + 0.23 * _safe_float(branch.get("probability"), 0.0)
                    + 0.15 * _safe_float(branch.get("branch_fitness"), 0.0)
                    + 0.06 * _safe_float(branch.get("minority_guardrail"), 0.0)
                    + 0.08 * memory_alignment
                    + 0.04 * persona_alignment
                    + 0.02 * contradiction_alignment
                )
                * contradiction_scale_map.get(contradiction_type, 1.0)
            ),
            6,
        )
    branches = sorted(
        branches,
        key=lambda branch: _safe_float(branch.get("v9_live_score"), 0.0),
        reverse=True,
    )

    cone_points = _price_cone_from_leaves(leaves, last_timestamp=last_timestamp, last_price=last_price, step_minutes=5)
    predicted_center = cone_points[-1]["center_price"] if cone_points else last_price
    mean_move = abs(predicted_center - last_price)
    directional_votes = []
    for leaf in leaves:
        if leaf.path_prices:
            directional_votes.append(1.0 if leaf.path_prices[-1] >= last_price else -1.0)
    directional_agreement = abs(float(np.mean(directional_votes))) if directional_votes else 0.0
    move_strength = min(1.0, mean_move / max(atr * 2.5, 1e-6))
    weighted_analog_confidence = float(
        np.average(
            np.asarray([float(leaf.analog_confidence) for leaf in leaves], dtype=np.float32),
            weights=np.asarray([max(float(leaf.probability_weight), 1e-6) for leaf in leaves], dtype=np.float32),
        )
    ) if leaves else 0.0
    overall_confidence = max(
        0.0,
        min(
            0.94,
            directional_agreement
            * (0.30 + 0.55 * move_strength + 0.15 * weighted_analog_confidence)
            * collapse.consensus_score,
        ),
    )
    overall_confidence = float(
        np.clip(
            overall_confidence
            * contradiction_scale_map.get(contradiction_type, 1.0)
            * (0.92 + 0.08 * _safe_float((memory_bank_context or {}).get("analog_confidence"), 0.0)),
            0.0,
            0.94,
        )
    )
    price_change = last_price - float(price_frame.iloc[-2]["close"]) if len(price_frame) > 1 else 0.0
    branch_conversation = _build_branch_conversation(branches, llm_content, {"scenario_bias": "bullish" if predicted_center >= last_price else "bearish"})
    return {
        "symbol": symbol,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "simulator_mode": "market_simulator",
        "market": {
            "candles": _candles_to_records(price_frame, limit=240),
            "current_price": round(last_price, 5),
            "price_change": round(float(price_change), 5),
            "source": str(price_frame.attrs.get("market_source", "unknown")),
        },
        "simulation": {
            "mean_probability": collapse.mean_probability,
            "uncertainty_width": collapse.uncertainty_width,
            "consensus_score": collapse.consensus_score,
            "overall_confidence": round(float(overall_confidence), 6),
            "dominant_driver": collapse.dominant_driver,
            "scenario_bias": "bullish" if predicted_center >= last_price else "bearish",
            "llm_numeric_prior": round(_llm_numeric_prior(llm_content), 6),
            "analog_bias": round(float(analog_snapshot.directional_bias), 6),
            "analog_confidence": round(float(analog_snapshot.confidence), 6),
            "analog_support": int(analog_snapshot.support),
            "memory_bank_bias": round(_safe_float(current_row.get("memory_bank_bias"), 0.0), 6),
            "memory_bank_confidence": round(_safe_float(current_row.get("memory_bank_confidence"), 0.0), 6),
            "branch_count": len(branches),
            "detected_regime": regime.dominant_regime,
            "regime_confidence": round(float(regime.dominant_confidence), 6),
            "retrieval_similarity": round(float(retrieval.similarity), 6) if retrieval is not None else 0.0,
            "selector_top_branch_id": selector_result.selected_branch_id,
            "selector_top_score": round(float(selector_result.selected_score), 6),
        },
        "personas": persona_snapshot,
        "branches": branches,
        "cone": cone_points,
        "branch_conversation": branch_conversation,
        "highlighted_branches": {
            "top_branch_id": selector_result.selected_branch_id or branch_conversation.get("top_branch_id"),
            "minority_branch_id": branch_conversation.get("minority_branch_id"),
        },
        "selector_rationale": selector_result.rationale,
    }


def build_live_sequence(
    symbol: str,
    sequence_len: int = SEQUENCE_LEN,
    llm_provider: str | None = None,
    llm_model: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    candles = fetch_recent_market_candles(symbol)
    price_frame = engineer_price_features(candles)
    quant_frame = build_quant_features(price_frame)
    price_frame = merge_quant_features(price_frame, quant_frame)
    technical_analysis = build_technical_analysis(price_frame)
    chart_snapshot = build_chart_snapshot(price_frame, bars=120)
    news_items = fetch_live_news(symbol)
    discussion_items = fetch_public_discussions(symbol)
    fear_greed = fetch_fear_greed_snapshot()
    if not discussion_items:
        discussion_items = [
            {
                "title": f"Fear & Greed proxy: {row.get('classification', 'neutral')}",
                "link": "",
                "published_at": row.get("timestamp"),
                "source": "alternative_me_fng_proxy",
                "sentiment": round(float(np.tanh((50.0 - _safe_float(row.get('value'), 50.0)) / 18.0)), 4),
                "classification": str(row.get("classification", "neutral")),
            }
            for row in fear_greed.get("history", [])[:3]
        ]
    macro = fetch_macro_context()

    price_index = price_frame.index.tz_localize(None) if getattr(price_frame.index, "tz", None) is not None else price_frame.index
    news_matrix, news_bias, news_intensity = _build_news_matrix(price_index, news_items)
    crowd_matrix, crowd_bias, crowd_extreme = _build_crowd_matrix(price_index, discussion_items, fear_greed)

    price_block = price_frame[PRICE_FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    min_rows = min(len(price_block), len(news_matrix), len(crowd_matrix))
    if min_rows < sequence_len:
        raise ValueError(f"Not enough live rows to build a sequence of length {sequence_len}. Only found {min_rows}.")

    fused = np.concatenate([price_block[:min_rows], news_matrix[:min_rows], crowd_matrix[:min_rows]], axis=1).astype(np.float32, copy=False)
    memory_bank_context = _memory_bank_context(fused)
    live_frame = price_frame.iloc[:min_rows].copy()
    latest = live_frame.iloc[-1].to_dict()
    recent_bars = (
        live_frame[["open", "high", "low", "close", "volume"]]
        .tail(72)
        .reset_index(drop=False)
        .rename(columns={"index": "timestamp"})
        .to_dict(orient="records")
    )
    for row in recent_bars:
        if row.get("timestamp") is not None:
            row["timestamp"] = pd.Timestamp(row["timestamp"]).isoformat()
    latest["recent_bars"] = recent_bars
    wltc_context = _build_wltc_context(recent_bars)
    mmm_context = _build_mmm_live_context(live_frame)
    mfg_context = _build_mfg_context(recent_bars)
    eci_context = _eci_display_context(live_frame.index[-1])
    latest["macro_bias"] = macro["macro_bias"]
    latest["macro_shock"] = macro["macro_shock"]
    latest["macro_driver"] = macro["driver"]
    latest["news_bias"] = round(float(news_bias), 4)
    latest["news_intensity"] = round(float(news_intensity), 4)
    latest["crowd_bias"] = round(float(crowd_bias), 4)
    latest["crowd_extreme"] = round(float(crowd_extreme), 4)
    latest["consensus_score"] = 0.0
    latest["hurst_overall"] = round(_safe_float(mmm_context.get("hurst_overall"), 0.5), 4)
    latest["hurst_positive"] = round(_safe_float(mmm_context.get("hurst_positive"), 0.5), 4)
    latest["hurst_negative"] = round(_safe_float(mmm_context.get("hurst_negative"), 0.5), 4)
    latest["hurst_asymmetry"] = round(_safe_float(mmm_context.get("hurst_asymmetry"), 0.0), 4)
    latest["market_memory_regime"] = str(mmm_context.get("market_memory_regime", "random_walk"))
    latest["mfg_disagreement"] = round(_safe_float(mfg_context.get("disagreement"), 0.0), 8)
    latest["mfg_consensus_drift"] = round(_safe_float(mfg_context.get("consensus_drift"), 0.0), 8)
    for persona_name, summary in wltc_context.items():
        latest[f"fear_index_{persona_name}"] = round(_safe_float(summary.get("fear_index"), 0.0), 4)
        latest[f"wltc_testosterone_{persona_name}"] = round(_safe_float(summary.get("testosterone_index"), 0.0), 4)
        latest[f"wltc_fundamental_tracking_{persona_name}"] = round(_safe_float(summary.get("fundamental_tracking"), 1.0), 4)
        latest[f"wltc_bid_aggressiveness_{persona_name}"] = round(_safe_float(summary.get("bid_aggressiveness"), 1.0), 4)
    analog_scorer = get_historical_analog_scorer()
    analog_history = (
        live_frame[PRICE_FEATURE_COLUMNS]
        .tail(analog_scorer.window_size)
        .astype(float)
        .to_dict(orient="records")
    )
    analog_snapshot = analog_scorer.score_window(analog_history)
    latest["analog_bias"] = round(float(analog_snapshot.directional_bias), 4)
    latest["analog_confidence"] = round(float(analog_snapshot.confidence), 4)
    latest["analog_support"] = int(analog_snapshot.support)
    latest["memory_bank_bias"] = round((float(memory_bank_context["bullish_probability"]) - 0.5) * 2.0, 4)
    latest["memory_bank_confidence"] = round(float(memory_bank_context["analog_confidence"]), 4)
    llm_context_input = {
        "symbol": symbol,
        "market": {
            "current_price": round(float(latest.get("close", 0.0)), 5),
            "atr_14": round(float(latest.get("atr_14", 0.0)), 5),
            "recent_closes": [round(float(value), 5) for value in live_frame["close"].tail(6).tolist()],
        },
        "simulation": {
            "macro_bias": latest["macro_bias"],
            "news_bias": latest["news_bias"],
            "crowd_bias": latest["crowd_bias"],
            "crowd_extreme": latest["crowd_extreme"],
            "hurst_overall": latest["hurst_overall"],
            "hurst_positive": latest["hurst_positive"],
            "hurst_negative": latest["hurst_negative"],
            "hurst_asymmetry": latest["hurst_asymmetry"],
            "eci_cone_width_modifier": round(_safe_float(eci_context.get("cone_width_modifier"), 0.0), 4),
        },
        "technical_analysis": {
            "structure": technical_analysis.get("structure"),
            "session": technical_analysis.get("session"),
            "location": technical_analysis.get("location"),
            "equilibrium": technical_analysis.get("equilibrium"),
            "quant_regime_strength": round(float(latest.get("quant_regime_strength", 0.0)), 4),
            "quant_transition_risk": round(float(latest.get("quant_transition_risk", 0.0)), 4),
            "quant_vol_realism": round(float(latest.get("quant_vol_realism", 0.0)), 4),
            "quant_fair_value_z": round(float(latest.get("quant_fair_value_z", 0.0)), 4),
            "nearest_support": technical_analysis.get("nearest_support"),
            "nearest_resistance": technical_analysis.get("nearest_resistance"),
            "order_blocks": technical_analysis.get("order_blocks", [])[:3],
            "fair_value_gaps": technical_analysis.get("fair_value_gaps", [])[:3],
        },
        "chart_snapshot_120": chart_snapshot,
        "macro": macro,
        "news_headlines": [item.get("title", "") for item in news_items[:5]],
        "crowd_items": [item.get("title", "") for item in discussion_items[:5]],
        "current_row": {
            "macro_bias": latest.get("macro_bias"),
            "macro_shock": latest.get("macro_shock"),
            "news_bias": latest.get("news_bias"),
            "news_intensity": latest.get("news_intensity"),
            "crowd_bias": latest.get("crowd_bias"),
            "crowd_extreme": latest.get("crowd_extreme"),
            "analog_bias": latest.get("analog_bias"),
            "analog_confidence": latest.get("analog_confidence"),
            "close": latest.get("close"),
            "atr_14": latest.get("atr_14"),
            "hurst_overall": latest.get("hurst_overall"),
            "hurst_positive": latest.get("hurst_positive"),
            "hurst_negative": latest.get("hurst_negative"),
            "hurst_asymmetry": latest.get("hurst_asymmetry"),
            "wltc_testosterone_retail": latest.get("wltc_testosterone_retail"),
            "wltc_testosterone_noise": latest.get("wltc_testosterone_noise"),
            "wltc_fundamental_tracking_retail": latest.get("wltc_fundamental_tracking_retail"),
            "wltc_fundamental_tracking_institutional": latest.get("wltc_fundamental_tracking_institutional"),
        },
        "wltc": wltc_context,
        "mmm": mmm_context,
        "mfg": mfg_context,
        "eci": {
            "cone_width_modifier": round(_safe_float(eci_context.get("cone_width_modifier"), 0.0), 4),
            "mins_to_next_high": round(_safe_float(eci_context.get("mins_to_next_high"), 0.0), 2),
            "mins_since_last_high": round(_safe_float(eci_context.get("mins_since_last_high"), 0.0), 2),
        },
    }

    using_nim = is_nvidia_nim_provider(llm_provider)
    llm_context = (
        {"available": False, "provider": "nvidia_nim", "reason": "deferred_to_kimi_judge", "content": {}}
        if using_nim
        else request_market_context(symbol, llm_context_input, provider=llm_provider, model=llm_model)
    )
    llm_content = _llm_content(llm_context)
    latest["llm_market_bias"] = round((_llm_numeric_prior(llm_content) - 0.5) * 2.0, 4)
    latest["llm_institutional_bias"] = round(_safe_float(llm_content.get("institutional_bias"), 0.0), 4)
    latest["llm_whale_bias"] = round(_safe_float(llm_content.get("whale_bias"), 0.0), 4)
    latest["llm_retail_bias"] = round(_safe_float(llm_content.get("retail_bias"), 0.0), 4)
    latest["llm_algo_bias"] = round(0.35 * latest["llm_institutional_bias"] + 0.15 * latest["llm_whale_bias"], 4)
    latest["llm_noise_bias"] = round(0.15 * latest["llm_retail_bias"], 4)

    personas = default_personas()
    persona_weights = _current_persona_weights(personas)
    persona_weight_tilts = _apply_llm_persona_tilts(personas, llm_content)
    feeds = {
        "news": news_items,
        "public_discussions": discussion_items,
        "fear_greed": fear_greed,
        "macro": macro,
    }
    bot_swarm = run_specialist_bots(
        symbol=symbol,
        current_row=latest,
        technical_analysis=technical_analysis,
        feeds=feeds,
        llm_content=llm_content,
    )
    latest["bot_swarm_bias"] = round((_safe_float(bot_swarm.get("aggregate", {}).get("bullish_probability"), 0.5) - 0.5) * 2.0, 4)
    latest["bot_swarm_confidence"] = round(_safe_float(bot_swarm.get("aggregate", {}).get("confidence"), 0.0), 4)
    latest["bot_swarm_disagreement"] = round(_safe_float(bot_swarm.get("aggregate", {}).get("disagreement"), 0.0), 4)
    style_biases = bot_swarm.get("aggregate", {}).get("style_biases", {}) if isinstance(bot_swarm, Mapping) else {}
    regime_affinity = bot_swarm.get("aggregate", {}).get("regime_affinity", {}) if isinstance(bot_swarm, Mapping) else {}
    latest["bot_trend_bias"] = round(_safe_float(style_biases.get("trend", {}).get("bias"), 0.0), 4)
    latest["bot_reversal_bias"] = round(_safe_float(style_biases.get("reversal", {}).get("bias"), 0.0), 4)
    latest["bot_structure_bias"] = round(_safe_float(style_biases.get("structure", {}).get("bias"), 0.0), 4)
    latest["bot_macro_bias"] = round(_safe_float(style_biases.get("macro", {}).get("bias"), 0.0), 4)
    latest["bot_shock_bias"] = round(_safe_float(style_biases.get("shock", {}).get("bias"), 0.0), 4)
    latest["bot_crowd_bias"] = round(_safe_float(style_biases.get("crowd", {}).get("bias"), 0.0), 4)
    latest["bot_regime_trend"] = round(_safe_float(regime_affinity.get("trend"), 0.0), 4)
    latest["bot_regime_reversal"] = round(_safe_float(regime_affinity.get("reversal"), 0.0), 4)
    latest["bot_regime_macro_shock"] = round(_safe_float(regime_affinity.get("macro_shock"), 0.0), 4)
    latest["bot_regime_balanced"] = round(_safe_float(regime_affinity.get("balanced"), 0.0), 4)
    horizon_predictions = {int(item.get("minutes", 0)): item for item in bot_swarm.get("aggregate", {}).get("horizon_predictions", [])}
    contradiction = classify_contradiction(
        prob_5m=_safe_float(horizon_predictions.get(5, {}).get("bullish_probability"), 0.5),
        prob_15m=_safe_float(horizon_predictions.get(15, {}).get("bullish_probability"), 0.5),
        prob_30m=_safe_float(horizon_predictions.get(30, {}).get("bullish_probability"), 0.5),
        conf_5m=_safe_float(bot_swarm.get("aggregate", {}).get("confidence"), 0.0),
        conf_15m=_safe_float(bot_swarm.get("aggregate", {}).get("confidence"), 0.0),
        conf_30m=_safe_float(bot_swarm.get("aggregate", {}).get("confidence"), 0.0),
    )
    contradiction_payload = {
        "type": contradiction.contradiction_type.value,
        "confidence": round(float(contradiction.confidence), 6),
        "cone_treatment": contradiction.cone_treatment,
        "long_bias_probability": round(
            float(
                0.5
                * _safe_float(horizon_predictions.get(15, {}).get("bullish_probability"), 0.5)
                + 0.5
                * _safe_float(horizon_predictions.get(30, {}).get("bullish_probability"), 0.5)
            ),
            6,
        ),
    }
    payload = _branch_payload(
        live_frame,
        latest,
        symbol,
        personas=personas,
        llm_content=llm_content,
        persona_weights=persona_weights,
        memory_bank_context=memory_bank_context,
        contradiction_context=contradiction_payload,
    )
    payload["persona_weight_tilts"] = persona_weight_tilts
    payload["persona_weights"] = persona_weights
    payload["feeds"] = feeds
    payload["bot_swarm"] = bot_swarm
    payload["llm_context"] = llm_context
    payload["current_row"] = latest
    payload["memory_bank_context"] = memory_bank_context
    payload["contradiction"] = contradiction_payload
    payload["wltc"] = wltc_context
    payload["mmm"] = mmm_context
    payload["mfg"] = mfg_context
    if isinstance(payload.get("simulation"), dict):
        payload["simulation"]["contradiction_type"] = contradiction.contradiction_type.value
        payload["simulation"]["contradiction_confidence"] = round(float(contradiction.confidence), 6)
        payload["simulation"]["cone_treatment"] = contradiction.cone_treatment
        payload["simulation"]["hurst_overall"] = latest["hurst_overall"]
        payload["simulation"]["hurst_positive"] = latest["hurst_positive"]
        payload["simulation"]["hurst_negative"] = latest["hurst_negative"]
        payload["simulation"]["hurst_asymmetry"] = latest["hurst_asymmetry"]
        payload["simulation"]["market_memory_regime"] = latest["market_memory_regime"]
        payload["simulation"]["mfg_disagreement"] = latest["mfg_disagreement"]
        payload["simulation"]["mfg_consensus_drift"] = latest["mfg_consensus_drift"]
        payload["simulation"]["testosterone_index"] = {
            name: round(_safe_float(summary.get("testosterone_index"), 0.0), 4)
            for name, summary in wltc_context.items()
        }
    payload["technical_analysis"] = technical_analysis
    payload["analog_context"] = analog_snapshot.to_dict()
    payload["branch_graph"] = _build_branch_graph(
        payload.get("branches", []),
        payload.get("highlighted_branches", {}),
        anchor_price=_safe_float(payload.get("market", {}).get("current_price"), 0.0),
    )
    judge_input = {
        "symbol": symbol,
        "simulation": payload.get("simulation", {}),
        "branches": payload.get("branches", [])[:6],
        "bot_swarm": {
            "aggregate": bot_swarm.get("aggregate", {}),
            "bots": bot_swarm.get("bots", []),
            "persona_reactions": bot_swarm.get("persona_reactions", {}),
        },
        "technical_analysis": technical_analysis,
        "news_feed": news_items[:8],
        "public_discussions": discussion_items[:8],
        "market_context": llm_content,
    }
    swarm_judge = (
        _fallback_swarm_judge(bot_swarm, payload.get("simulation", {}), technical_analysis)
        if using_nim
        else request_swarm_judgment(symbol, judge_input, provider=llm_provider, model=llm_model)
    )
    if not swarm_judge.get("available", False):
        swarm_judge = _fallback_swarm_judge(bot_swarm, payload.get("simulation", {}), technical_analysis)
    payload["swarm_judge"] = swarm_judge
    payload["chart_snapshot_120"] = chart_snapshot
    payload["llm_provider"] = llm_provider or "lm_studio"
    payload["llm_model"] = llm_model or ""
    payload["eci"] = eci_context
    return fused[-sequence_len:], payload


def build_live_simulation(
    symbol: str,
    sequence_len: int = SEQUENCE_LEN,
    llm_provider: str | None = None,
    llm_model: str | None = None,
) -> dict[str, Any]:
    sequence, payload = build_live_sequence(
        symbol,
        sequence_len=sequence_len,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )
    payload["sequence_shape"] = [int(sequence.shape[0]), int(sequence.shape[1])]
    return payload | {"sequence": sequence.tolist()}


def build_live_monitor(symbol: str) -> dict[str, Any]:
    candles = fetch_recent_market_candles(symbol)
    price_frame = engineer_price_features(candles)
    return build_simulation_comparison(symbol, price_frame)



