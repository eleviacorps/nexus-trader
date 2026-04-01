from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None

try:
    from sklearn.mixture import GaussianMixture  # type: ignore
except ImportError:  # pragma: no cover
    GaussianMixture = None


QUANT_FEATURE_COLUMNS = [
    "quant_regime_code",
    "quant_regime_strength",
    "quant_regime_persistence",
    "quant_transition_risk",
    "quant_state_entropy",
    "quant_tail_risk",
    "quant_vol_forecast",
    "quant_vol_realism",
    "quant_fair_value_z",
    "quant_kalman_fair_value",
    "quant_kalman_dislocation",
    "quant_trend_score",
    "quant_state_prob_up",
    "quant_state_prob_down",
    "quant_state_prob_chop",
    "quant_state_prob_range",
    "quant_route_prob_up",
    "quant_route_prob_down",
    "quant_route_prob_chop",
    "quant_route_prob_range",
    "quant_route_confidence",
]


@dataclass(frozen=True)
class QuantRegimeReport:
    rows: int
    state_count: int
    feature_columns: list[str]
    regime_counts: dict[str, int]
    avg_transition_risk: float
    avg_vol_realism: float
    avg_regime_strength: float


def _require_pandas() -> Any:
    if pd is None:
        raise ImportError("pandas is required for quant hybrid features.")
    return pd


def _series(frame, column: str, default: float = 0.0):
    pandas = _require_pandas()
    if column in frame.columns:
        return pandas.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)
    return pandas.Series(np.full(len(frame), default, dtype=np.float32), index=frame.index)


def _zscore(series, window: int = 96):
    mean = series.rolling(window, min_periods=max(8, window // 8)).mean()
    std = series.rolling(window, min_periods=max(8, window // 8)).std(ddof=0).replace(0.0, np.nan)
    return ((series - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _fit_regime_probabilities(feature_matrix: np.ndarray, state_count: int) -> np.ndarray:
    if feature_matrix.size == 0:
        return np.zeros((0, state_count), dtype=np.float32)
    if GaussianMixture is None:
        scores = feature_matrix[:, 0]
        volatility = feature_matrix[:, 1] if feature_matrix.shape[1] > 1 else np.zeros(len(scores), dtype=np.float32)
        probs = np.zeros((len(scores), state_count), dtype=np.float32)
        bullish = scores >= 0.35
        bearish = scores <= -0.35
        chop = volatility >= np.quantile(volatility, 0.65) if len(volatility) else np.zeros(len(scores), dtype=bool)
        probs[:, 0] = bullish.astype(np.float32)
        probs[:, 1] = bearish.astype(np.float32)
        probs[:, 2] = (~bullish & ~bearish & chop).astype(np.float32)
        probs[:, 3] = 1.0 - probs[:, :3].sum(axis=1)
        probs = np.clip(probs, 0.0, 1.0)
        row_sum = probs.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0.0] = 1.0
        return (probs / row_sum).astype(np.float32)

    model = GaussianMixture(
        n_components=state_count,
        covariance_type="full",
        random_state=42,
        max_iter=300,
        reg_covar=1e-5,
    )
    model.fit(feature_matrix)
    return model.predict_proba(feature_matrix).astype(np.float32)


def _estimate_transition_matrix(regime_codes: np.ndarray, state_count: int) -> np.ndarray:
    counts = np.ones((state_count, state_count), dtype=np.float32)
    if regime_codes.size > 1:
        for previous, current in zip(regime_codes[:-1], regime_codes[1:]):
            prev_idx = int(np.clip(previous, 0, state_count - 1))
            curr_idx = int(np.clip(current, 0, state_count - 1))
            counts[prev_idx, curr_idx] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return (counts / row_sums).astype(np.float32)


def _route_regime_probabilities(emission_probs: np.ndarray, regime_codes: np.ndarray) -> np.ndarray:
    if emission_probs.size == 0:
        return np.zeros_like(emission_probs, dtype=np.float32)
    state_count = emission_probs.shape[1]
    transition_matrix = _estimate_transition_matrix(regime_codes.astype(np.int32, copy=False), state_count=state_count)
    routed = np.zeros_like(emission_probs, dtype=np.float32)
    routed[0] = emission_probs[0]
    for index in range(1, len(emission_probs)):
        prior = routed[index - 1] @ transition_matrix
        combined = (0.58 * emission_probs[index]) + (0.42 * prior)
        total = float(combined.sum())
        if total <= 0.0:
            combined = np.full(state_count, 1.0 / max(1, state_count), dtype=np.float32)
        else:
            combined = combined / total
        routed[index] = combined.astype(np.float32)
    return routed


def _kalman_fair_value(
    close: np.ndarray,
    anchor: np.ndarray,
    process_scale: np.ndarray,
    measurement_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    length = len(close)
    fair_value = np.zeros(length, dtype=np.float32)
    dislocation = np.zeros(length, dtype=np.float32)
    if length == 0:
        return fair_value, dislocation
    state = float(anchor[0] if np.isfinite(anchor[0]) else close[0])
    covariance = 1.0
    for index in range(length):
        measurement = float((0.55 * close[index]) + (0.45 * anchor[index]))
        process_var = max(1e-6, float(process_scale[index]))
        measurement_var = max(1e-6, float(measurement_scale[index]))
        covariance = covariance + process_var
        kalman_gain = covariance / (covariance + measurement_var)
        state = state + kalman_gain * (measurement - state)
        covariance = (1.0 - kalman_gain) * covariance
        fair_value[index] = float(state)
        dislocation[index] = float((close[index] - state) / max(abs(state), 1e-6))
    return fair_value.astype(np.float32), dislocation.astype(np.float32)


def build_quant_features(price_frame, *, state_count: int = 4):
    pandas = _require_pandas()
    frame = price_frame.copy()
    close = _series(frame, "close")
    return_1 = _series(frame, "return_1")
    return_6 = _series(frame, "return_6")
    return_12 = _series(frame, "return_12")
    atr_pct = _series(frame, "atr_pct").abs()
    bb_width = _series(frame, "bb_width").abs()
    ema_cross = _series(frame, "ema_cross")
    macd_hist = _series(frame, "macd_hist")
    rsi_14 = _series(frame, "rsi_14", 50.0)

    short_vol = return_1.abs().ewm(span=12, adjust=False).mean().fillna(0.0)
    long_vol = return_1.abs().ewm(span=72, adjust=False).mean().fillna(0.0)
    trend_score = np.tanh(
        (0.85 * ema_cross)
        + (0.65 * macd_hist)
        + (0.04 * (rsi_14 - 50.0))
        + (1.75 * return_6)
        + (1.10 * return_12)
    )
    vol_shock = np.abs(np.log(np.maximum(short_vol, 1e-6) / np.maximum(long_vol, 1e-6)))
    vol_realism = np.exp(-np.clip(vol_shock, 0.0, 3.0))
    fair_value_anchor = close.ewm(span=48, adjust=False).mean()
    fair_value_z = _zscore(np.log(np.maximum(close, 1e-6) / np.maximum(fair_value_anchor, 1e-6)), window=96)

    raw_features = np.column_stack(
        [
            trend_score.to_numpy(dtype=np.float32),
            short_vol.to_numpy(dtype=np.float32),
            long_vol.to_numpy(dtype=np.float32),
            atr_pct.to_numpy(dtype=np.float32),
            bb_width.to_numpy(dtype=np.float32),
            fair_value_z.to_numpy(dtype=np.float32),
        ]
    )
    finite_mask = np.isfinite(raw_features).all(axis=1)
    valid_rows = raw_features[finite_mask]
    if len(valid_rows) == 0:
        valid_rows = np.zeros((len(frame), raw_features.shape[1]), dtype=np.float32)
        finite_mask = np.ones(len(frame), dtype=bool)

    means = valid_rows.mean(axis=0, keepdims=True)
    stds = valid_rows.std(axis=0, keepdims=True)
    stds[stds < 1e-6] = 1.0
    normalized = np.zeros_like(raw_features, dtype=np.float32)
    normalized[finite_mask] = ((raw_features[finite_mask] - means) / stds).astype(np.float32)

    regime_probabilities = _fit_regime_probabilities(normalized[finite_mask], state_count=state_count)
    full_probabilities = np.zeros((len(frame), state_count), dtype=np.float32)
    full_probabilities[finite_mask] = regime_probabilities

    cluster_trend = regime_probabilities.T @ trend_score.to_numpy(dtype=np.float32)[finite_mask]
    cluster_vol = regime_probabilities.T @ short_vol.to_numpy(dtype=np.float32)[finite_mask]
    cluster_weight = np.maximum(regime_probabilities.sum(axis=0), 1e-6)
    cluster_trend = cluster_trend / cluster_weight
    cluster_vol = cluster_vol / cluster_weight

    up_cluster = int(np.argmax(cluster_trend))
    down_cluster = int(np.argmin(cluster_trend))
    remaining = [idx for idx in range(state_count) if idx not in {up_cluster, down_cluster}]
    if remaining:
        chop_cluster = remaining[int(np.argmax(cluster_vol[remaining]))]
        range_cluster = next((idx for idx in remaining if idx != chop_cluster), chop_cluster)
    else:
        chop_cluster = up_cluster
        range_cluster = down_cluster

    regime_code = full_probabilities.argmax(axis=1).astype(np.float32)
    regime_strength = full_probabilities.max(axis=1).astype(np.float32)
    regime_entropy = -np.sum(full_probabilities * np.log(np.clip(full_probabilities, 1e-6, 1.0)), axis=1) / np.log(max(2, state_count))
    routed_probabilities = _route_regime_probabilities(full_probabilities, regime_code.astype(np.int32, copy=False))
    route_confidence = routed_probabilities.max(axis=1).astype(np.float32)
    up_prob = full_probabilities[:, up_cluster]
    down_prob = full_probabilities[:, down_cluster]
    chop_prob = full_probabilities[:, chop_cluster]
    range_prob = full_probabilities[:, range_cluster]
    route_up_prob = routed_probabilities[:, up_cluster]
    route_down_prob = routed_probabilities[:, down_cluster]
    route_chop_prob = routed_probabilities[:, chop_cluster]
    route_range_prob = routed_probabilities[:, range_cluster]

    regime_switch = np.zeros(len(frame), dtype=np.float32)
    if len(regime_code) > 1:
        regime_switch[1:] = (regime_code[1:] != regime_code[:-1]).astype(np.float32)
    regime_persistence = (
        pandas.Series(1.0 - regime_switch, index=frame.index)
        .rolling(12, min_periods=1)
        .mean()
        .clip(0.0, 1.0)
        .to_numpy(dtype=np.float32)
    )
    transition_risk = np.clip(
        0.55 * (1.0 - regime_strength)
        + 0.30 * pandas.Series(regime_switch, index=frame.index).rolling(8, min_periods=1).mean().to_numpy(dtype=np.float32)
        + 0.15 * regime_entropy.astype(np.float32)
        - 0.10 * regime_persistence,
        0.0,
        1.0,
    )
    tail_risk = np.clip(
        0.40 * (1.0 - vol_realism.to_numpy(dtype=np.float32))
        + 0.35 * np.clip(np.abs(fair_value_z.to_numpy(dtype=np.float32)) / 3.0, 0.0, 1.0)
        + 0.15 * transition_risk.astype(np.float32)
        + 0.10 * regime_entropy.astype(np.float32),
        0.0,
        1.0,
    )
    kalman_fair_value, kalman_dislocation = _kalman_fair_value(
        close.to_numpy(dtype=np.float32),
        fair_value_anchor.to_numpy(dtype=np.float32),
        process_scale=(short_vol.to_numpy(dtype=np.float32) + atr_pct.to_numpy(dtype=np.float32) + 1e-6),
        measurement_scale=(long_vol.to_numpy(dtype=np.float32) + bb_width.to_numpy(dtype=np.float32) + 1e-6),
    )

    quant_frame = pandas.DataFrame(
        {
            "quant_regime_code": regime_code,
            "quant_regime_strength": regime_strength,
            "quant_regime_persistence": regime_persistence.astype(np.float32),
            "quant_transition_risk": transition_risk.astype(np.float32),
            "quant_state_entropy": regime_entropy.astype(np.float32),
            "quant_tail_risk": tail_risk.astype(np.float32),
            "quant_vol_forecast": short_vol.to_numpy(dtype=np.float32),
            "quant_vol_realism": vol_realism.to_numpy(dtype=np.float32),
            "quant_fair_value_z": fair_value_z.to_numpy(dtype=np.float32),
            "quant_kalman_fair_value": kalman_fair_value,
            "quant_kalman_dislocation": kalman_dislocation,
            "quant_trend_score": trend_score.to_numpy(dtype=np.float32),
            "quant_state_prob_up": up_prob.astype(np.float32),
            "quant_state_prob_down": down_prob.astype(np.float32),
            "quant_state_prob_chop": chop_prob.astype(np.float32),
            "quant_state_prob_range": range_prob.astype(np.float32),
            "quant_route_prob_up": route_up_prob.astype(np.float32),
            "quant_route_prob_down": route_down_prob.astype(np.float32),
            "quant_route_prob_chop": route_chop_prob.astype(np.float32),
            "quant_route_prob_range": route_range_prob.astype(np.float32),
            "quant_route_confidence": route_confidence.astype(np.float32),
        },
        index=frame.index,
    )
    return quant_frame.fillna(0.0).astype(np.float32)


def merge_quant_features(price_frame, quant_frame):
    pandas = _require_pandas()
    if quant_frame is None or len(quant_frame) == 0:
        return price_frame
    merged = price_frame.copy()
    aligned = quant_frame.copy()
    aligned.index = pandas.to_datetime(aligned.index, errors="coerce")
    merged.index = pandas.to_datetime(merged.index, errors="coerce")
    for column in QUANT_FEATURE_COLUMNS:
        if column in aligned.columns:
            merged[column] = aligned[column].reindex(merged.index).ffill().fillna(0.0).astype(np.float32)
    return merged


def summarize_quant_frame(quant_frame) -> QuantRegimeReport:
    regime_codes = quant_frame["quant_regime_code"].round().astype(int).tolist() if len(quant_frame) else []
    counts: dict[str, int] = {}
    for code in regime_codes:
        counts[str(code)] = counts.get(str(code), 0) + 1
    return QuantRegimeReport(
        rows=int(len(quant_frame)),
        state_count=4,
        feature_columns=list(QUANT_FEATURE_COLUMNS),
        regime_counts=counts,
        avg_transition_risk=float(quant_frame["quant_transition_risk"].mean()) if len(quant_frame) else 0.0,
        avg_vol_realism=float(quant_frame["quant_vol_realism"].mean()) if len(quant_frame) else 0.0,
        avg_regime_strength=float(quant_frame["quant_regime_strength"].mean()) if len(quant_frame) else 0.0,
    )
