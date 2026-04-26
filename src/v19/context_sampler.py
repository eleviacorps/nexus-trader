from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from config.project_config import PERSONA_OUTPUTS_PATH, QUANT_FEATURES_PATH, V10_BRANCH_ARCHIVE_FULL_PATH, V10_BRANCH_ARCHIVE_PATH, V17_MMM_FEATURES_PATH, V19_BRANCH_ARCHIVE_PATH

PIP_SIZE_XAUUSD = 0.1

REGIME_LABELS: tuple[str, ...] = (
    "trend_up",
    "trend_down",
    "range",
    "breakout",
    "panic_shock",
    "unknown",
)
SQT_LABELS: tuple[str, ...] = ("COLD", "NEUTRAL", "GOOD", "HOT")
STRUCTURE_LABELS: tuple[str, ...] = ("bullish", "bearish", "neutral", "unknown")
LOCATION_LABELS: tuple[str, ...] = ("premium", "discount", "equilibrium", "unknown")
BOT_SIGNAL_LABELS: tuple[str, ...] = ("bullish", "bearish", "neutral")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if np.isnan(number) or np.isinf(number):
            return float(default)
        return number
    except Exception:
        return float(default)


def _clip01(value: Any) -> float:
    return float(np.clip(_safe_float(value), 0.0, 1.0))


def _timestamp_series(frame: pd.DataFrame, column: str = "timestamp") -> pd.Series:
    if column in frame.columns:
        return pd.to_datetime(frame[column], utc=True, errors="coerce")
    if isinstance(frame.index, pd.DatetimeIndex):
        return pd.to_datetime(frame.index, utc=True, errors="coerce")
    return pd.to_datetime(pd.Series([pd.NaT] * len(frame)), utc=True, errors="coerce")


def _normalize_regime(raw: Any) -> str:
    text = str(raw or "unknown").strip().lower()
    if "panic" in text or "shock" in text:
        return "panic_shock"
    if "break" in text:
        return "breakout"
    if "down" in text or "bear" in text:
        return "trend_down"
    if "up" in text or "bull" in text:
        return "trend_up"
    if "range" in text or "chop" in text or "random" in text:
        return "range"
    return "unknown"


def _normalize_location(raw: Any) -> str:
    text = str(raw or "unknown").strip().lower()
    if text in {"premium", "discount", "equilibrium"}:
        return text
    return "unknown"


def _normalize_structure(raw: Any) -> str:
    text = str(raw or "unknown").strip().lower()
    if text in {"bullish", "bearish", "neutral"}:
        return text
    return "unknown"


def _normalize_bot_signal(raw: Any) -> str:
    text = str(raw or "neutral").strip().lower()
    if text in {"bullish", "bearish", "neutral"}:
        return text
    return "neutral"


def _normalize_direction(raw: Any) -> str:
    text = str(raw or "HOLD").strip().upper()
    if text in {"BUY", "SELL", "HOLD"}:
        return text
    if "BULL" in text or "UP" in text:
        return "BUY"
    if "BEAR" in text or "DOWN" in text:
        return "SELL"
    return "HOLD"


def _flatten_numeric(payload: Any, prefix: str = "") -> dict[str, float]:
    output: dict[str, float] = {}
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            output.update(_flatten_numeric(value, child_prefix))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            child_prefix = f"{prefix}[{index}]"
            output.update(_flatten_numeric(value, child_prefix))
    else:
        if isinstance(payload, bool):
            return output
        if isinstance(payload, (int, float)):
            output[prefix] = float(payload)
    return output


def _sqt_from_score(score: float) -> str:
    if score < 0.35:
        return "COLD"
    if score < 0.55:
        return "NEUTRAL"
    if score < 0.75:
        return "GOOD"
    return "HOT"


def _contradiction_type(row: Mapping[str, Any]) -> str:
    disagreement = _safe_float(row.get("analog_disagreement"), 0.0)
    macro_alignment = _clip01(row.get("macro_alignment"))
    news_consistency = _clip01(row.get("news_consistency"))
    crowd_consistency = _clip01(row.get("crowd_consistency"))
    if disagreement > 0.55 and max(macro_alignment, news_consistency, crowd_consistency) < 0.45:
        return "full_disagreement"
    if disagreement < 0.20 and min(macro_alignment, news_consistency, crowd_consistency) > 0.55:
        return "full_agreement"
    if disagreement > 0.40:
        return "partial_disagreement"
    return "mixed"


def _branch_aggregate_frame(path: str | None = None) -> pd.DataFrame:
    candidate = Path(path) if path is not None else (V10_BRANCH_ARCHIVE_FULL_PATH if V10_BRANCH_ARCHIVE_FULL_PATH.exists() else V10_BRANCH_ARCHIVE_PATH)
    archive = pd.read_parquet(candidate)
    archive["timestamp"] = _timestamp_series(archive)
    grouped = (
        archive.sort_values(["timestamp", "sample_id", "branch_id"])
        .groupby("timestamp", as_index=False)
        .agg(
            {
                "sample_id": "first",
                "dominant_regime": "first",
                "generator_probability": "max",
                "analog_similarity": "mean",
                "analog_disagreement": "mean",
                "news_consistency": "mean",
                "crowd_consistency": "mean",
                "macro_alignment": "mean",
                "branch_confidence": "mean",
                "path_error": "mean",
                "anchor_price": "first",
                "predicted_price_5m": "mean",
                "predicted_price_10m": "mean",
                "predicted_price_15m": "mean",
                "actual_price_15m": "mean",
                "actual_final_return": "mean",
                "model_direction_prob_15m": "mean",
                "model_hold_prob_15m": "mean",
                "model_confidence_prob_15m": "mean",
                "leaf_analog_confidence": "mean",
                "branch_volatility": "mean",
                "volatility_scale": "mean",
                "v10_diversity_score": "mean",
                "hmm_transition_risk": "mean",
                "volatility_realism": "mean",
                "fair_value_dislocation": "mean",
                "branch_direction": "first",
                "v10_regime_label": "first",
            }
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    grouped["dominant_regime"] = grouped["dominant_regime"].map(_normalize_regime)
    grouped["v10_regime_label"] = grouped["v10_regime_label"].map(_normalize_regime)
    grouped["contradiction_type"] = grouped.apply(_contradiction_type, axis=1)
    grouped["cone_realism"] = 1.0 - np.clip(np.abs(pd.to_numeric(grouped["path_error"], errors="coerce").fillna(0.0)), 0.0, 1.0)
    grouped["analog_confidence"] = pd.to_numeric(grouped["leaf_analog_confidence"], errors="coerce").fillna(
        pd.to_numeric(grouped["analog_similarity"], errors="coerce").fillna(0.0)
    )
    grouped["cone_width_pips_proxy"] = (
        np.abs(
            pd.to_numeric(grouped["predicted_price_15m"], errors="coerce").fillna(0.0)
            - pd.to_numeric(grouped["anchor_price"], errors="coerce").fillna(0.0)
        )
        / PIP_SIZE_XAUUSD
    )
    return grouped


def _persona_frame() -> pd.DataFrame:
    columns = [
        "timestamp",
        "macro_bias",
        "macro_shock",
        "news_bias",
        "news_intensity",
        "crowd_bias",
        "crowd_extreme",
        "consensus_score",
        "retail_impact",
        "institutional_impact",
        "algo_impact",
        "whale_impact",
        "noise_impact",
        "dominant_driver",
    ]
    frame = pd.read_parquet(PERSONA_OUTPUTS_PATH, columns=columns)
    frame["timestamp"] = _timestamp_series(frame)
    return frame.sort_values("timestamp").reset_index(drop=True)


def _quant_frame() -> pd.DataFrame:
    columns = [
        "quant_regime_strength",
        "quant_transition_risk",
        "quant_vol_realism",
        "quant_fair_value_z",
        "quant_kalman_fair_value",
        "quant_route_confidence",
        "quant_trend_score",
    ]
    frame = pd.read_parquet(QUANT_FEATURES_PATH, columns=columns).reset_index()
    timestamp_column = "datetime" if "datetime" in frame.columns else frame.columns[0]
    frame = frame.rename(columns={timestamp_column: "timestamp"})
    frame["timestamp"] = _timestamp_series(frame)
    return frame.sort_values("timestamp").reset_index(drop=True)


def _mmm_frame() -> pd.DataFrame:
    frame = pd.read_parquet(V17_MMM_FEATURES_PATH)
    frame["timestamp"] = _timestamp_series(frame)
    return frame.sort_values("timestamp").reset_index(drop=True)


@lru_cache(maxsize=1)
def load_context_source_frame() -> pd.DataFrame:
    if V19_BRANCH_ARCHIVE_PATH.exists():
        try:
            branch_archive = pd.read_parquet(V19_BRANCH_ARCHIVE_PATH)
            branch_archive["timestamp"] = _timestamp_series(branch_archive)
            return branch_archive.sort_values("timestamp").reset_index(drop=True)
        except Exception:
            pass
    base = _branch_aggregate_frame()
    try:
        persona = _persona_frame()
        quant = _quant_frame()
        mmm = _mmm_frame()
    except Exception:
        return base.sort_values("timestamp").reset_index(drop=True)
    working = pd.merge_asof(
        base.sort_values("timestamp"),
        persona.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("30min"),
    )
    working = pd.merge_asof(
        working.sort_values("timestamp"),
        quant.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("15min"),
    )
    working = pd.merge_asof(
        working.sort_values("timestamp"),
        mmm.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("12h"),
    )
    numeric_columns = working.select_dtypes(include=["number", "bool"]).columns
    for column in numeric_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0.0)
    for column in ["market_memory_regime", "dominant_driver"]:
        if column not in working.columns:
            working[column] = "unknown"
        working[column] = working[column].fillna("unknown").astype(str)
    return working.sort_values("timestamp").reset_index(drop=True)


def build_context_from_row(row: Mapping[str, Any], symbol: str = "XAUUSD") -> dict[str, Any]:
    timestamp = pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce")
    if pd.isna(timestamp):
        timestamp = pd.Timestamp.now(tz="UTC")
    current_price = _safe_float(row.get("anchor_price"), _safe_float(row.get("quant_kalman_fair_value"), 0.0))
    predicted_5 = _safe_float(row.get("predicted_price_5m"), current_price)
    predicted_10 = _safe_float(row.get("predicted_price_10m"), predicted_5)
    predicted_15 = _safe_float(row.get("predicted_price_15m"), predicted_10)
    overall_confidence = float(
        np.mean(
            [
                _clip01(row.get("model_confidence_prob_15m")),
                _clip01(row.get("generator_probability")),
                _clip01(row.get("analog_similarity")),
            ]
        )
    )
    cabr_score = float(
        np.mean(
            [
                _clip01(row.get("branch_confidence")),
                _clip01(row.get("analog_confidence")),
                _clip01(row.get("quant_route_confidence")),
            ]
        )
    )
    cpm_score = float(
        np.mean(
            [
                _clip01(row.get("generator_probability")),
                _clip01(row.get("volatility_realism")),
                np.clip(1.0 - _safe_float(row.get("hmm_transition_risk"), 0.0), 0.0, 1.0),
            ]
        )
    )
    regime = _normalize_regime(row.get("dominant_regime") or row.get("v10_regime_label") or row.get("market_memory_regime"))
    direction = "BUY" if predicted_15 >= current_price else "SELL"
    scenario_bias = "bullish" if direction == "BUY" else "bearish"
    contradiction = str(row.get("contradiction_type", "mixed"))
    cone_width_pips = abs(predicted_15 - current_price) / PIP_SIZE_XAUUSD
    volatility_anchor = max(_safe_float(row.get("branch_volatility"), 0.0), _safe_float(row.get("volatility_scale"), current_price * 0.0015))
    fair_value = _safe_float(row.get("quant_kalman_fair_value"), current_price)
    location = "discount" if current_price <= fair_value else "premium"
    structure = "bullish" if _safe_float(row.get("quant_trend_score"), 0.0) >= 0.0 else "bearish"
    support = current_price - max(volatility_anchor, current_price * 0.001)
    resistance = current_price + max(volatility_anchor, current_price * 0.001)
    sqt_label = _sqt_from_score(np.mean([overall_confidence, cabr_score, cpm_score]))
    news_bias = _safe_float(row.get("news_bias"), 0.0)
    crowd_bias = _safe_float(row.get("crowd_bias"), 0.0)
    macro_bias = _safe_float(row.get("macro_bias"), 0.0)
    mfg_disagreement = abs(_safe_float(row.get("analog_disagreement"), 0.0)) * 0.25 + _clip01(row.get("crowd_extreme")) * 0.10
    testosterone_retail = _clip01(abs(_safe_float(row.get("retail_impact"), 0.0)))
    testosterone_inst = _clip01(abs(_safe_float(row.get("institutional_impact"), 0.0)))
    return {
        "symbol": symbol,
        "market": {
            "current_price": round(current_price, 5),
            "atr_14": round(max(volatility_anchor, current_price * 0.001), 5),
            "candles": [
                {"minutes": 0, "timestamp": timestamp.isoformat(), "price": round(current_price, 5)},
                {"minutes": 5, "timestamp": (timestamp + pd.Timedelta(minutes=5)).isoformat(), "price": round(predicted_5, 5)},
                {"minutes": 10, "timestamp": (timestamp + pd.Timedelta(minutes=10)).isoformat(), "price": round(predicted_10, 5)},
                {"minutes": 15, "timestamp": (timestamp + pd.Timedelta(minutes=15)).isoformat(), "price": round(predicted_15, 5)},
            ],
        },
        "simulation": {
            "scenario_bias": scenario_bias,
            "direction": direction,
            "overall_confidence": round(overall_confidence, 6),
            "cabr_score": round(cabr_score, 6),
            "cpm_score": round(cpm_score, 6),
            "cone_width_pips": round(float(cone_width_pips), 3),
            "contradiction_type": contradiction,
            "detected_regime": regime,
            "hurst_overall": round(_safe_float(row.get("hurst_overall"), 0.5), 6),
            "hurst_positive": round(_safe_float(row.get("hurst_positive"), 0.5), 6),
            "hurst_negative": round(_safe_float(row.get("hurst_negative"), 0.5), 6),
            "hurst_asymmetry": round(_safe_float(row.get("hurst_asymmetry"), 0.0), 6),
            "testosterone_index": {
                "retail": round(testosterone_retail, 6),
                "institutional": round(testosterone_inst, 6),
            },
            "suggested_lot": 0.05,
            "cone_c_m": round(max(_safe_float(row.get("cone_realism"), 0.0), 0.0), 6),
            "entry_zone": [round(current_price - (0.08 * volatility_anchor), 2), round(current_price + (0.08 * volatility_anchor), 2)],
        },
        "technical_analysis": {
            "structure": structure,
            "location": location,
            "rsi_14": round(50.0 + (10.0 * np.clip(_safe_float(row.get("quant_trend_score"), 0.0), -1.0, 1.0)), 2),
            "atr_14": round(max(volatility_anchor, current_price * 0.001), 5),
            "equilibrium": round(fair_value, 5),
            "nearest_support": {"price": round(support, 5)},
            "nearest_resistance": {"price": round(resistance, 5)},
            "order_blocks": [
                {"price": round(support, 5), "strength": round(_clip01(row.get("analog_confidence")), 3), "type": "demand"},
                {"price": round(resistance, 5), "strength": round(_clip01(row.get("volatility_realism")), 3), "type": "supply"},
            ],
        },
        "bot_swarm": {
            "aggregate": {
                "signal": "bullish" if direction == "BUY" else "bearish",
                "bullish_probability": round(float(np.clip(0.5 + (0.5 * _safe_float(row.get("quant_trend_score"), 0.0)), 0.0, 1.0)), 6),
                "disagreement": round(float(np.clip(mfg_disagreement, 0.0, 1.0)), 6),
            }
        },
        "news_feed": [
            {
                "title": f"Synthetic {regime} context driven by {row.get('dominant_driver', 'mixed factors')}",
                "source": "historical_archive",
                "sentiment": round(float(np.clip(news_bias, -1.0, 1.0)), 4),
            }
        ],
        "public_discussions": [
            {
                "title": f"Crowd bias snapshot with consensus {round(_safe_float(row.get('consensus_score'), 0.0), 3)}",
                "source": "persona_outputs",
                "sentiment": round(float(np.clip(crowd_bias, -1.0, 1.0)), 4),
            }
        ],
        "sqt": {
            "label": sqt_label,
            "rolling_accuracy": round(float(np.clip(np.mean([overall_confidence, cabr_score]), 0.0, 1.0)), 6),
        },
        "mfg": {
            "disagreement": round(float(mfg_disagreement), 6),
            "consensus_drift": round(float(np.mean([macro_bias, news_bias, crowd_bias])), 6),
        },
        "v18_paths": {
            "consensus_path": [
                {"minutes": 5, "price": round(predicted_5, 5)},
                {"minutes": 10, "price": round(predicted_10, 5)},
                {"minutes": 15, "price": round(predicted_15, 5)},
            ],
            "minority_path": [
                {"minutes": 5, "price": round(current_price - (predicted_5 - current_price), 5)},
                {"minutes": 10, "price": round(current_price - (predicted_10 - current_price), 5)},
                {"minutes": 15, "price": round(current_price - (predicted_15 - current_price), 5)},
            ],
            "outer_upper": [{"minutes": 15, "price": round(resistance, 5)}],
            "outer_lower": [{"minutes": 15, "price": round(support, 5)}],
        },
    }


def feature_map_from_context(context: Mapping[str, Any]) -> dict[str, float]:
    numeric_map = _flatten_numeric(context)
    simulation = dict(context.get("simulation", {}) if isinstance(context, Mapping) else {})
    technical = dict(context.get("technical_analysis", {}) if isinstance(context, Mapping) else {})
    bot_swarm = dict((context.get("bot_swarm", {}) or {}).get("aggregate", {}) if isinstance(context, Mapping) else {})
    sqt = dict(context.get("sqt", {}) if isinstance(context, Mapping) else {})
    direction = _normalize_direction(simulation.get("direction", simulation.get("scenario_bias")))
    regime = _normalize_regime(simulation.get("detected_regime"))
    location = _normalize_location(technical.get("location"))
    structure = _normalize_structure(technical.get("structure"))
    bot_signal = _normalize_bot_signal(bot_swarm.get("signal"))
    sqt_label = str(sqt.get("label", simulation.get("sqt_label", "NEUTRAL"))).strip().upper()
    for candidate in ("BUY", "SELL", "HOLD"):
        numeric_map[f"cat.direction.{candidate.lower()}"] = 1.0 if direction == candidate else 0.0
    for candidate in REGIME_LABELS:
        numeric_map[f"cat.regime.{candidate}"] = 1.0 if regime == candidate else 0.0
    for candidate in SQT_LABELS:
        numeric_map[f"cat.sqt.{candidate.lower()}"] = 1.0 if sqt_label == candidate else 0.0
    for candidate in STRUCTURE_LABELS:
        numeric_map[f"cat.structure.{candidate}"] = 1.0 if structure == candidate else 0.0
    for candidate in LOCATION_LABELS:
        numeric_map[f"cat.location.{candidate}"] = 1.0 if location == candidate else 0.0
    for candidate in BOT_SIGNAL_LABELS:
        numeric_map[f"cat.bot_signal.{candidate}"] = 1.0 if bot_signal == candidate else 0.0
    return {key: float(value) for key, value in numeric_map.items() if key}


def context_to_feature_vector(context: Mapping[str, Any], feature_names: Sequence[str] | None = None) -> tuple[np.ndarray, list[str]]:
    feature_map = feature_map_from_context(context)
    ordered_names = list(feature_names) if feature_names is not None else sorted(feature_map)
    vector = np.asarray([feature_map.get(name, 0.0) for name in ordered_names], dtype=np.float32)
    return vector, ordered_names


@dataclass
class SimulationContextSampler:
    source_frame: pd.DataFrame | None = None
    seed: int = 42

    def _frame(self) -> pd.DataFrame:
        return self.source_frame.copy() if self.source_frame is not None else load_context_source_frame().copy()

    def sample_rows(self, n_samples: int = 50_000, balance_regimes: bool = True) -> pd.DataFrame:
        frame = self._frame()
        if frame.empty:
            return frame
        rng = np.random.default_rng(self.seed)
        n_samples = max(int(n_samples), 1)
        if not balance_regimes or "dominant_regime" not in frame.columns:
            if len(frame) <= n_samples:
                return frame.reset_index(drop=True)
            selection = rng.choice(len(frame), size=n_samples, replace=False)
            return frame.iloc[selection].sort_values("timestamp").reset_index(drop=True)
        frame["dominant_regime"] = frame["dominant_regime"].map(_normalize_regime)
        regimes = [regime for regime in sorted(frame["dominant_regime"].dropna().unique().tolist()) if regime]
        if not regimes:
            return frame.head(n_samples).reset_index(drop=True)
        per_regime = max(n_samples // len(regimes), 1)
        parts: list[pd.DataFrame] = []
        for regime in regimes:
            subset = frame.loc[frame["dominant_regime"] == regime].copy()
            if subset.empty:
                continue
            replace = len(subset) < per_regime
            sample = subset.sample(
                n=per_regime if replace or len(subset) >= per_regime else len(subset),
                replace=replace,
                random_state=self.seed,
            )
            parts.append(sample)
        sampled = pd.concat(parts, ignore_index=True) if parts else frame.head(n_samples).copy()
        if len(sampled) < n_samples:
            remaining = n_samples - len(sampled)
            additional = frame.sample(n=remaining, replace=len(frame) < remaining, random_state=self.seed)
            sampled = pd.concat([sampled, additional], ignore_index=True)
        return sampled.head(n_samples).sort_values("timestamp").reset_index(drop=True)

    def sample_contexts(self, n_samples: int = 50_000, balance_regimes: bool = True, symbol: str = "XAUUSD") -> list[dict[str, Any]]:
        rows = self.sample_rows(n_samples=n_samples, balance_regimes=balance_regimes)
        return [build_context_from_row(row._asdict(), symbol=symbol) for row in rows.itertuples(index=False)]
