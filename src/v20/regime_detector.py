from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


DEFAULT_STATE_NAMES = {
    0: "low_vol_range",
    1: "trending_up",
    2: "trending_down",
    3: "breakout",
    4: "mean_revert",
    5: "panic",
}


def _safe_numeric_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    for column in columns:
        output[column] = pd.to_numeric(frame.get(column), errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    return output


def _label_states(feature_frame: pd.DataFrame, states: np.ndarray) -> dict[int, str]:
    labeled: dict[int, str] = {}
    summary = feature_frame.copy()
    summary["state"] = states
    group = summary.groupby("state").agg(
        mean_return=("log_return", "mean"),
        mean_vol=("realized_vol_20", "mean"),
        jump_rate=("macro_jump_flag", "mean"),
    )
    remaining = set(group.index.tolist())
    if not remaining:
        return DEFAULT_STATE_NAMES.copy()
    vol_order = group["mean_vol"].sort_values()
    labeled[int(vol_order.index[0])] = "low_vol_range"
    remaining.discard(int(vol_order.index[0]))
    if remaining:
        up_state = int(group.loc[list(remaining), "mean_return"].idxmax())
        labeled[up_state] = "trending_up"
        remaining.discard(up_state)
    if remaining:
        down_state = int(group.loc[list(remaining), "mean_return"].idxmin())
        labeled[down_state] = "trending_down"
        remaining.discard(down_state)
    if remaining:
        panic_state = int(group.loc[list(remaining), "jump_rate"].idxmax())
        labeled[panic_state] = "panic"
        remaining.discard(panic_state)
    if remaining:
        breakout_state = int(group.loc[list(remaining), "mean_vol"].idxmax())
        labeled[breakout_state] = "breakout"
        remaining.discard(breakout_state)
    for state in remaining:
        labeled[int(state)] = "mean_revert"
    for state in range(6):
        labeled.setdefault(state, DEFAULT_STATE_NAMES.get(state, f"state_{state}"))
    return labeled


@dataclass
class RegimeDetector:
    model: GaussianHMM
    state_names: dict[int, str]
    feature_columns: tuple[str, ...]

    def transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        numeric = _safe_numeric_frame(features_df, list(self.feature_columns))
        X = numeric.to_numpy(dtype=np.float64)
        states = self.model.predict(X)
        _, posteriors = self.model.score_samples(X)
        duration: list[int] = []
        running = 0
        prev_state: int | None = None
        for state in states.tolist():
            if prev_state is None or state != prev_state:
                running = 1
            else:
                running += 1
            prev_state = int(state)
            duration.append(running)
        output = pd.DataFrame(index=features_df.index)
        output["hmm_state"] = states.astype(np.int64)
        for idx in range(posteriors.shape[1]):
            output[f"hmm_prob_{idx}"] = posteriors[:, idx].astype(np.float32)
        output["hmm_duration"] = np.asarray(duration, dtype=np.int64)
        output["hmm_state_name"] = [self.state_names.get(int(state), f"state_{int(state)}") for state in states]
        return output

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(
                {
                    "model": self.model,
                    "state_names": self.state_names,
                    "feature_columns": self.feature_columns,
                },
                handle,
            )

    @classmethod
    def load(cls, path: str | Path) -> "RegimeDetector":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        return cls(
            model=payload["model"],
            state_names=dict(payload["state_names"]),
            feature_columns=tuple(payload["feature_columns"]),
        )


def train_hmm(features_df: pd.DataFrame, n_states: int = 6) -> tuple[RegimeDetector, np.ndarray, np.ndarray]:
    feature_columns = ["log_return", "realized_vol_20", "volume_zscore", "macro_vol_regime_class", "macro_jump_flag"]
    numeric = _safe_numeric_frame(features_df, feature_columns)
    X = numeric.to_numpy(dtype=np.float64)
    model = GaussianHMM(
        n_components=int(n_states),
        covariance_type="full",
        n_iter=200,
        random_state=42,
    )
    model.fit(X)
    states = model.predict(X)
    _, posteriors = model.score_samples(X)
    state_names = _label_states(
        pd.DataFrame(
            {
                "log_return": numeric["log_return"],
                "realized_vol_20": numeric["realized_vol_20"],
                "macro_jump_flag": numeric["macro_jump_flag"],
            },
            index=features_df.index,
        ),
        states,
    )
    detector = RegimeDetector(model=model, state_names=state_names, feature_columns=tuple(feature_columns))
    return detector, states, posteriors
