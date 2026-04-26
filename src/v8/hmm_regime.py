from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None

try:
    from hmmlearn.hmm import GaussianHMM  # type: ignore
except ImportError:  # pragma: no cover
    GaussianHMM = None

try:
    from sklearn.mixture import GaussianMixture  # type: ignore
except ImportError:  # pragma: no cover
    GaussianMixture = None


REGIME_LABELS_V8: tuple[str, ...] = (
    "bullish_trend",
    "bearish_trend",
    "mean_reversion",
    "range",
    "breakout",
    "false_breakout",
    "panic_news_shock",
    "low_volatility_drift",
)


@dataclass(frozen=True)
class HMMRegimeReport:
    rows: int
    state_count: int
    provider: str
    dominant_regime_counts: dict[str, int]
    avg_regime_confidence: float
    avg_transition_probability: float
    avg_persistence: float


def _require_pandas():
    if pd is None:
        raise ImportError("pandas is required for v8 HMM regime features.")
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


def _build_regime_inputs(frame) -> np.ndarray:
    close = _series(frame, "close")
    return_1 = _series(frame, "return_1")
    atr_pct = _series(frame, "atr_pct").abs()
    trend_score = _series(frame, "quant_trend_score", _series(frame, "ema_cross"))
    volume_ratio = _series(frame, "volume_ratio", 1.0)
    dislocation = _series(frame, "quant_kalman_dislocation", 0.0)
    realized_vol = return_1.abs().rolling(12, min_periods=3).mean().fillna(0.0)
    spread_proxy = np.abs(dislocation) + atr_pct
    features = np.column_stack(
        [
            _zscore(return_1, 128).to_numpy(dtype=np.float32),
            _zscore(atr_pct, 128).to_numpy(dtype=np.float32),
            _zscore(realized_vol, 128).to_numpy(dtype=np.float32),
            _zscore(trend_score, 128).to_numpy(dtype=np.float32),
            _zscore(volume_ratio, 128).to_numpy(dtype=np.float32),
            _zscore(spread_proxy, 128).to_numpy(dtype=np.float32),
            _zscore(np.log(np.maximum(close, 1e-6)), 128).to_numpy(dtype=np.float32),
        ]
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _cluster_with_hmm(features: np.ndarray, state_count: int) -> tuple[np.ndarray, np.ndarray, str]:
    fit_features = features
    if len(features) > 120_000:
        sample_idx = np.linspace(0, len(features) - 1, 120_000, dtype=np.int64)
        fit_features = features[sample_idx]
    if GaussianHMM is not None:
        model = GaussianHMM(n_components=state_count, covariance_type="diag", n_iter=120, random_state=42)
        model.fit(fit_features)
        return model.predict_proba(features).astype(np.float32), np.asarray(model.transmat_, dtype=np.float32), "hmmlearn"
    if GaussianMixture is not None:
        model = GaussianMixture(n_components=state_count, covariance_type="full", random_state=42, max_iter=250, reg_covar=1e-5)
        model.fit(fit_features)
        probabilities = model.predict_proba(features).astype(np.float32)
        codes = probabilities.argmax(axis=1)
        transition = np.ones((state_count, state_count), dtype=np.float32)
        for previous, current in zip(codes[:-1], codes[1:]):
            transition[int(previous), int(current)] += 1.0
        transition /= transition.sum(axis=1, keepdims=True)
        return probabilities, transition.astype(np.float32), "gaussian_mixture"
    probs = np.full((len(features), state_count), 1.0 / max(1, state_count), dtype=np.float32)
    transition = np.full((state_count, state_count), 1.0 / max(1, state_count), dtype=np.float32)
    return probs, transition, "fallback_uniform"


def _map_states(probabilities: np.ndarray, features: np.ndarray) -> list[str]:
    dominant = probabilities.argmax(axis=1)
    state_means = []
    for state in range(probabilities.shape[1]):
        mask = dominant == state
        state_means.append(features[mask].mean(axis=0) if mask.any() else np.zeros(features.shape[1], dtype=np.float32))
    state_means = np.asarray(state_means, dtype=np.float32)
    trend = state_means[:, 0] + 0.7 * state_means[:, 3]
    vol = np.abs(state_means[:, 1]) + np.abs(state_means[:, 2]) + np.abs(state_means[:, 5])
    mapping = ["range"] * probabilities.shape[1]
    up_state = int(np.argmax(trend))
    down_state = int(np.argmin(trend))
    mapping[up_state] = "bullish_trend"
    mapping[down_state] = "bearish_trend"
    remaining = [idx for idx in range(probabilities.shape[1]) if idx not in {up_state, down_state}]
    if remaining:
        mapping[max(remaining, key=lambda idx: vol[idx])] = "panic_news_shock"
    return mapping


def build_hmm_regime_frame(price_frame, *, state_count: int = 8):
    pandas = _require_pandas()
    features = _build_regime_inputs(price_frame)
    probabilities, transition, provider = _cluster_with_hmm(features, state_count=state_count)
    mapping = _map_states(probabilities, features)
    dominant_state = probabilities.argmax(axis=1)
    dominant_labels = [mapping[int(code)] for code in dominant_state]
    persistence = np.zeros(len(probabilities), dtype=np.float32)
    transition_probability = np.zeros(len(probabilities), dtype=np.float32)
    for idx in range(len(probabilities)):
        state = int(dominant_state[idx])
        persistence[idx] = float(transition[state, state])
        transition_probability[idx] = float(1.0 - transition[state, state])
    output = pandas.DataFrame(index=price_frame.index)
    for state_idx in range(probabilities.shape[1]):
        output[f"hmm_state_{state_idx}_prob"] = probabilities[:, state_idx].astype(np.float32)
    output["hmm_regime_code"] = dominant_state.astype(np.float32)
    output["hmm_regime_confidence"] = probabilities.max(axis=1).astype(np.float32)
    output["hmm_regime_persistence"] = persistence.astype(np.float32)
    output["hmm_transition_probability"] = transition_probability.astype(np.float32)
    output["hmm_dominant_regime"] = dominant_labels
    output.attrs["provider"] = provider
    output.attrs["state_mapping"] = mapping
    return output


def summarize_hmm_frame(hmm_frame) -> HMMRegimeReport:
    dominant_labels = hmm_frame["hmm_dominant_regime"].astype(str).tolist() if len(hmm_frame) else []
    counts: dict[str, int] = {}
    for label in dominant_labels:
        counts[label] = counts.get(label, 0) + 1
    return HMMRegimeReport(
        rows=int(len(hmm_frame)),
        state_count=int(sum(1 for column in hmm_frame.columns if str(column).startswith("hmm_state_") and str(column).endswith("_prob"))),
        provider=str(hmm_frame.attrs.get("provider", "unknown")),
        dominant_regime_counts=counts,
        avg_regime_confidence=float(hmm_frame["hmm_regime_confidence"].mean()) if len(hmm_frame) else 0.0,
        avg_transition_probability=float(hmm_frame["hmm_transition_probability"].mean()) if len(hmm_frame) else 0.0,
        avg_persistence=float(hmm_frame["hmm_regime_persistence"].mean()) if len(hmm_frame) else 0.0,
    )
