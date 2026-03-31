from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from config.project_config import (
    CROWD_EMBEDDINGS_NPY_PATH,
    CROWD_EVENTS_PATH,
    LEGACY_PRICE_FEATURES_CSV,
    LEGACY_PRICE_FEATURES_PARQUET,
    MACRO_FEATURES_PATH,
    NEWS_EMBEDDINGS_RAW_PATH,
    PERSONA_CONFIG_PATH,
    PERSONA_OUTPUTS_PATH,
    PERSONA_WEIGHT_HISTORY_PATH,
    PRICE_FEATURES_CSV_FALLBACK,
    PRICE_FEATURES_PATH,
    SIM_CONFIDENCE_PATH,
    SIM_TARGETS_PATH,
)
from src.pipeline.fusion import load_price_frame, save_numpy_artifact
from src.simulation.personas import default_personas, save_personas

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None


PERSONA_ORDER = ["retail", "institutional", "algo", "whale", "noise"]


def _require_pandas() -> Any:
    if pd is None:
        raise ImportError("pandas is required for persona pipelines.")
    return pd


def _resolve_price_frame():
    price_path = next(
        (
            path
            for path in [PRICE_FEATURES_PATH, PRICE_FEATURES_CSV_FALLBACK, LEGACY_PRICE_FEATURES_PARQUET, LEGACY_PRICE_FEATURES_CSV]
            if path.exists()
        ),
        None,
    )
    if price_path is None:
        raise FileNotFoundError("Price features are required before building persona outputs.")
    frame = load_price_frame(price_path)
    pandas = _require_pandas()
    frame = frame.copy()
    frame.index = pandas.to_datetime(frame.index, errors="coerce")
    frame = frame[~frame.index.isna()].sort_index()
    return frame, price_path


def _rolling_zscore(series, window: int = 252):
    pandas = _require_pandas()
    history_mean = series.rolling(window, min_periods=20).mean().shift(1)
    history_std = series.rolling(window, min_periods=20).std(ddof=0).shift(1)
    fallback_mean = series.expanding(min_periods=20).mean().shift(1)
    fallback_std = series.expanding(min_periods=20).std(ddof=0).shift(1)
    mean = history_mean.fillna(fallback_mean).bfill().fillna(0.0)
    std = history_std.fillna(fallback_std).replace(0.0, np.nan)
    return ((series - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def align_macro_to_price(price_index, macro_frame):
    pandas = _require_pandas()
    bars = pandas.DataFrame({"timestamp": pandas.to_datetime(price_index, errors="coerce")}).dropna().reset_index(drop=True)
    bars["timestamp"] = bars["timestamp"].astype("datetime64[ns]")
    if macro_frame.empty:
        return pandas.DataFrame({"timestamp": bars["timestamp"], "macro_bias": 0.0, "macro_shock": 0.0, "macro_driver": "macro_neutral"})

    macro = macro_frame.copy()
    date_column = "date" if "date" in macro.columns else macro.columns[0]
    macro["timestamp"] = pandas.to_datetime(macro[date_column], errors="coerce").astype("datetime64[ns]")
    macro = macro.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    value_columns = [column for column in macro.columns if column not in {date_column, "timestamp"}]
    for column in value_columns:
        macro[column] = pandas.to_numeric(macro[column], errors="coerce").ffill()

    def col(name: str):
        if name in macro.columns:
            return pandas.to_numeric(macro[name], errors="coerce").ffill()
        return pandas.Series(np.zeros(len(macro), dtype=np.float32))

    dollar = 0.65 * _rolling_zscore(col("DTWEXBGS")) + 0.35 * _rolling_zscore(col("UUP"))
    real_rates = _rolling_zscore(col("DFII10"))
    nominal_rates = _rolling_zscore(col("DGS10"))
    risk = _rolling_zscore(col("VIXCLS"))
    inflation = _rolling_zscore(col("T10YIE"))
    bonds = _rolling_zscore(col("TLT"))
    oil = _rolling_zscore(col("DCOILWTICO"))
    gold_proxy = 0.50 * _rolling_zscore(col("GC_F")) + 0.50 * _rolling_zscore(col("GLD"))

    macro_bias = np.tanh((-0.80 * dollar) + (-0.55 * real_rates) + (-0.25 * nominal_rates) + (0.50 * risk) + (0.30 * inflation) + (0.25 * bonds) + (0.15 * oil) + (0.35 * gold_proxy))
    macro_shock = np.clip(
        0.45 * col("VIXCLS").pct_change().abs().fillna(0.0)
        + 0.35 * col("DTWEXBGS").pct_change().abs().fillna(0.0)
        + 0.20 * col("DGS10").pct_change().abs().fillna(0.0),
        0.0,
        1.0,
    )

    components = {
        "dollar_weakness": (-dollar).to_numpy(dtype=np.float32),
        "real_yields_falling": (-real_rates).to_numpy(dtype=np.float32),
        "risk_aversion": risk.to_numpy(dtype=np.float32),
        "inflation_rising": inflation.to_numpy(dtype=np.float32),
        "gold_proxy_strength": gold_proxy.to_numpy(dtype=np.float32),
    }
    driver_names = list(components)
    driver_matrix = np.column_stack([np.abs(components[name]) for name in driver_names])
    driver_codes = driver_matrix.argmax(axis=1)
    driver_values = np.take_along_axis(
        np.column_stack([components[name] for name in driver_names]),
        driver_codes[:, None],
        axis=1,
    ).ravel()
    driver_labels = [
        f"{driver_names[index]}_bullish" if driver_values[position] >= 0 else f"{driver_names[index]}_bearish"
        for position, index in enumerate(driver_codes)
    ]

    aligned_source = macro[["timestamp"]].copy()
    aligned_source["macro_bias"] = macro_bias.astype(np.float32)
    aligned_source["macro_shock"] = np.asarray(macro_shock, dtype=np.float32)
    aligned_source["macro_driver"] = driver_labels
    aligned = pandas.merge_asof(
        bars.sort_values("timestamp"),
        aligned_source.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    aligned["macro_bias"] = aligned["macro_bias"].fillna(0.0)
    aligned["macro_shock"] = aligned["macro_shock"].fillna(0.0)
    aligned["macro_driver"] = aligned["macro_driver"].fillna("macro_neutral")
    return aligned


def _news_bias_from_embeddings(news_block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if news_block.size == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    columns = min(news_block.shape[1], 8)
    focus = news_block[:, :columns].astype(np.float32, copy=False)
    signed = focus[:, 0]
    if columns > 1:
        signed = signed + 0.55 * focus[:, 1]
    if columns > 2:
        signed = signed - 0.35 * focus[:, 2]
    if columns > 3:
        signed = signed + 0.25 * focus[:, 3]
    scale = float(np.nanstd(signed)) or 1.0
    bias = np.tanh(signed / scale).astype(np.float32)
    intensity = np.clip(np.linalg.norm(focus, axis=1) / max(1.0, np.sqrt(columns) * 1.5), 0.0, 1.0).astype(np.float32)
    return bias, intensity


def _crowd_bias_from_embeddings(crowd_block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if crowd_block.size == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    score_norm = crowd_block[:, 0]
    fear_bias = np.clip(0.5 - score_norm, -0.5, 0.5) * 2.0
    cftc_gold = crowd_block[:, 24] if crowd_block.shape[1] > 24 else 0.0
    cftc_eurusd = crowd_block[:, 25] if crowd_block.shape[1] > 25 else 0.0
    cftc_btc = crowd_block[:, 26] if crowd_block.shape[1] > 26 else 0.0
    bias = np.tanh(fear_bias + 0.45 * cftc_gold + 0.20 * cftc_eurusd + 0.10 * cftc_btc).astype(np.float32)
    extreme = np.clip(
        crowd_block[:, 14]
        + crowd_block[:, 15]
        + crowd_block[:, 16]
        + crowd_block[:, 19],
        0.0,
        1.0,
    ).astype(np.float32)
    return bias, extreme


def _price_scores(price_frame):
    close = price_frame
    trend = np.tanh(
        1.15 * close["ema_cross"].to_numpy(dtype=np.float32)
        + 0.85 * close["macd_hist"].to_numpy(dtype=np.float32)
        + 0.06 * (close["rsi_14"].to_numpy(dtype=np.float32) - 50.0)
        + 2.20 * close["return_3"].to_numpy(dtype=np.float32)
    ).astype(np.float32)
    mean_reversion = np.tanh(
        2.6 * (0.5 - close["bb_pct"].to_numpy(dtype=np.float32))
        + 0.05 * (50.0 - close["rsi_14"].to_numpy(dtype=np.float32))
        + 0.60 * close["dist_to_high"].to_numpy(dtype=np.float32)
        - 0.60 * close["dist_to_low"].to_numpy(dtype=np.float32)
    ).astype(np.float32)
    liquidity = np.tanh(
        0.90 * close["ll"].to_numpy(dtype=np.float32)
        - 0.90 * close["hh"].to_numpy(dtype=np.float32)
        + 0.65 * (1.0 - np.clip(close["dist_to_low"].to_numpy(dtype=np.float32), 0.0, 1.5))
        - 0.65 * (1.0 - np.clip(close["dist_to_high"].to_numpy(dtype=np.float32), 0.0, 1.5))
    ).astype(np.float32)
    momentum = np.tanh(
        0.08 * (close["rsi_7"].to_numpy(dtype=np.float32) - 50.0)
        + 0.90 * close["macd_hist"].to_numpy(dtype=np.float32)
        + 0.65 * (close["body_pct"].to_numpy(dtype=np.float32) - 0.5)
        + 1.75 * close["return_1"].to_numpy(dtype=np.float32)
    ).astype(np.float32)
    return trend, mean_reversion, liquidity, momentum


def _signals_from_score(score: np.ndarray, threshold: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    direction = np.where(score >= threshold, 1, np.where(score <= -threshold, -1, 0)).astype(np.int8)
    confidence = np.where(direction == 0, 0.0, np.clip(np.abs(score), 0.0, 0.99)).astype(np.float32)
    return direction, confidence


def _reason_labels(reason_codes: np.ndarray, labels: list[str]) -> np.ndarray:
    return np.asarray([labels[int(code)] for code in reason_codes], dtype=object)


def _rolling_accuracy_weight(direction, confidence, targets, capital_weight: float, window: int = 720):
    pandas = _require_pandas()
    active = direction != 0
    correct = ((direction == 1) & (targets > 0)) | ((direction == -1) & (targets <= 0))
    active_series = pandas.Series(active.astype(np.float32))
    correct_series = pandas.Series(np.where(active, correct.astype(np.float32), np.nan))
    confidence_series = pandas.Series(confidence.astype(np.float32))
    rolling_accuracy = correct_series.rolling(window, min_periods=25).mean().fillna(0.5)
    rolling_confidence = confidence_series.rolling(window, min_periods=25).mean().fillna(float(np.nanmean(confidence) or 0.5))
    weight = capital_weight * (0.35 + 0.65 * rolling_accuracy) * (0.70 + 0.30 * rolling_confidence)
    return rolling_accuracy.astype(np.float32), weight.astype(np.float32)


def build_persona_artifacts() -> dict[str, Any]:
    pandas = _require_pandas()
    price_frame, price_path = _resolve_price_frame()
    if not MACRO_FEATURES_PATH.exists():
        raise FileNotFoundError("macro_features.parquet is required before persona simulation.")
    if not NEWS_EMBEDDINGS_RAW_PATH.exists():
        raise FileNotFoundError("news_embeddings.npy is required before persona simulation.")
    if not CROWD_EMBEDDINGS_NPY_PATH.exists():
        raise FileNotFoundError("crowd_embeddings.npy is required before persona simulation.")

    macro_frame = pandas.read_parquet(MACRO_FEATURES_PATH)
    news_block = np.load(NEWS_EMBEDDINGS_RAW_PATH, mmap_mode="r")
    crowd_block = np.load(CROWD_EMBEDDINGS_NPY_PATH, mmap_mode="r")
    row_count = min(len(price_frame), len(news_block), len(crowd_block))
    price_frame = price_frame.iloc[:row_count].copy()
    news_block = np.asarray(news_block[:row_count], dtype=np.float32)
    crowd_block = np.asarray(crowd_block[:row_count], dtype=np.float32)

    macro_context = align_macro_to_price(price_frame.index[:row_count], macro_frame)
    news_bias, news_intensity = _news_bias_from_embeddings(news_block)
    crowd_bias, crowd_extreme = _crowd_bias_from_embeddings(crowd_block)
    trend, mean_reversion, liquidity, momentum = _price_scores(price_frame)
    emotion_chase = np.sign(trend + 1e-6) * crowd_extreme

    persona_signals = {
        "retail": {
            "price_trend": 0.55 * trend,
            "news_follow": 0.20 * news_bias,
            "crowd_chase": 0.25 * emotion_chase,
        },
        "institutional": {
            "macro_regime": 0.45 * macro_context["macro_bias"].to_numpy(dtype=np.float32),
            "news_regime": 0.15 * news_bias,
            "liquidity_value": 0.25 * liquidity,
            "mean_reversion": 0.15 * mean_reversion,
        },
        "algo": {
            "momentum": 0.45 * momentum,
            "trend_structure": 0.35 * trend,
            "liquidity_signal": 0.20 * liquidity,
        },
        "whale": {
            "macro_regime": 0.40 * macro_context["macro_bias"].to_numpy(dtype=np.float32),
            "contrarian_crowd": -0.35 * emotion_chase,
            "liquidity_accumulation": 0.25 * liquidity,
        },
        "noise": {
            "noise_wave": 0.65 * np.tanh(np.sin(np.arange(row_count) * 0.173) + np.cos(np.arange(row_count) * 0.037)).astype(np.float32),
            "headline_reaction": 0.20 * news_bias,
            "crowd_reaction": 0.15 * crowd_bias,
        },
    }

    if PERSONA_CONFIG_PATH.exists():
        try:
            personas_payload = json.loads(PERSONA_CONFIG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            personas_payload = {}
    else:
        personas_payload = {}

    if not personas_payload:
        personas = default_personas()
        save_personas(PERSONA_CONFIG_PATH, personas)
        personas_payload = {name: asdict(config) for name, config in personas.items()}

    targets = price_frame["target_direction"].to_numpy(dtype=np.float32)
    output = pandas.DataFrame({"timestamp": price_frame.index[:row_count]})
    weight_history = pandas.DataFrame({"timestamp": price_frame.index[:row_count]})
    normalized_weight_columns = []

    capital_weights = {
        name: float(personas_payload.get(name, {}).get("capital_weight", 0.2))
        for name in PERSONA_ORDER
    }

    for persona_name in PERSONA_ORDER:
        components = persona_signals[persona_name]
        labels = list(components)
        matrix = np.column_stack([value.astype(np.float32, copy=False) for value in components.values()])
        score = np.tanh(matrix.sum(axis=1)).astype(np.float32)
        direction, confidence = _signals_from_score(score)
        reason_codes = np.abs(matrix).argmax(axis=1)
        reasons = _reason_labels(reason_codes, labels)
        rolling_accuracy, raw_weight = _rolling_accuracy_weight(direction, confidence, targets, capital_weights[persona_name])

        output[f"{persona_name}_direction"] = direction
        output[f"{persona_name}_confidence"] = confidence
        output[f"{persona_name}_reason"] = reasons

        weight_history[f"{persona_name}_accuracy_rolling"] = rolling_accuracy
        weight_history[f"{persona_name}_weight_raw"] = raw_weight
        normalized_weight_columns.append(f"{persona_name}_weight_raw")

    raw_weight_matrix = weight_history[normalized_weight_columns].to_numpy(dtype=np.float32)
    weight_denominator = raw_weight_matrix.sum(axis=1, keepdims=True)
    weight_denominator[weight_denominator == 0.0] = 1.0
    normalized_weights = raw_weight_matrix / weight_denominator

    for position, persona_name in enumerate(PERSONA_ORDER):
        weight_history[f"{persona_name}_weight"] = normalized_weights[:, position].astype(np.float32)

    weighted_score = np.zeros(row_count, dtype=np.float32)
    impact_matrix = []
    for position, persona_name in enumerate(PERSONA_ORDER):
        direction = output[f"{persona_name}_direction"].to_numpy(dtype=np.float32)
        confidence = output[f"{persona_name}_confidence"].to_numpy(dtype=np.float32)
        impact = normalized_weights[:, position] * direction * confidence
        output[f"{persona_name}_impact"] = impact.astype(np.float32)
        impact_matrix.append(np.abs(impact))
        weighted_score += impact

    impact_stack = np.column_stack(impact_matrix)
    dominant_persona_codes = impact_stack.argmax(axis=1)
    dominant_personas = np.asarray([PERSONA_ORDER[int(code)] for code in dominant_persona_codes], dtype=object)
    output["macro_bias"] = macro_context["macro_bias"].to_numpy(dtype=np.float32)
    output["macro_shock"] = macro_context["macro_shock"].to_numpy(dtype=np.float32)
    output["news_bias"] = news_bias
    output["news_intensity"] = news_intensity
    output["crowd_bias"] = crowd_bias
    output["crowd_extreme"] = crowd_extreme
    output["consensus_score"] = weighted_score.astype(np.float32)
    output["simulated_direction"] = np.where(weighted_score >= 0.0, 1, -1).astype(np.int8)
    output["simulated_confidence"] = np.clip(np.abs(weighted_score), 0.0, 0.99).astype(np.float32)
    output["dominant_persona"] = dominant_personas
    output["dominant_driver"] = np.where(
        np.abs(macro_context["macro_bias"].to_numpy(dtype=np.float32)) >= np.maximum(np.abs(news_bias), np.abs(crowd_bias)),
        macro_context["macro_driver"].astype(str).to_numpy(),
        np.where(np.abs(news_bias) >= np.abs(crowd_bias), "headline_regime", "crowd_extreme"),
    )

    PERSONA_OUTPUTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PERSONA_WEIGHT_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(PERSONA_OUTPUTS_PATH, index=False)
    weight_history.to_parquet(PERSONA_WEIGHT_HISTORY_PATH, index=False)
    save_numpy_artifact(SIM_TARGETS_PATH, (output["simulated_direction"].to_numpy(dtype=np.float32) > 0).astype(np.float32))
    save_numpy_artifact(SIM_CONFIDENCE_PATH, output["simulated_confidence"].to_numpy(dtype=np.float32))

    source_summary = []
    if CROWD_EVENTS_PATH.exists():
        crowd_events = pandas.read_parquet(CROWD_EVENTS_PATH)
        if "source" in crowd_events.columns:
            source_summary.extend(sorted(crowd_events["source"].astype(str).dropna().unique().tolist()))

    report = {
        "rows": int(row_count),
        "price_source": str(price_path),
        "macro_source": str(MACRO_FEATURES_PATH),
        "news_source": str(NEWS_EMBEDDINGS_RAW_PATH),
        "crowd_source": str(CROWD_EMBEDDINGS_NPY_PATH),
        "persona_outputs_path": str(PERSONA_OUTPUTS_PATH),
        "weight_history_path": str(PERSONA_WEIGHT_HISTORY_PATH),
        "sim_target_path": str(SIM_TARGETS_PATH),
        "sim_confidence_path": str(SIM_CONFIDENCE_PATH),
        "personas": PERSONA_ORDER,
        "dominant_driver_counts": output["dominant_driver"].value_counts().head(10).to_dict(),
        "crowd_sources": source_summary,
    }
    return report
