from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

TARGET_EXCLUDE = {
    "target_up_15m",
    "target_up_30m",
    "future_return_15m",
    "future_return_30m",
    "range_forward_15m",
}


@dataclass(frozen=True)
class V21SequenceBundle:
    feature_columns: list[str]
    dataset: TensorDataset
    feature_mean: np.ndarray
    feature_std: np.ndarray


def _vol_bucket(values: pd.Series) -> pd.Series:
    quantiles = values.quantile([0.33, 0.66]).to_list() if len(values) else [0.0, 0.0]
    low, high = float(quantiles[0]), float(quantiles[1])
    return pd.Series(np.where(values <= low, 0, np.where(values <= high, 1, 2)), index=values.index, dtype=np.int64)


def _range_bucket(values: pd.Series) -> pd.Series:
    quantiles = values.quantile([0.33, 0.66]).to_list() if len(values) else [0.0, 0.0]
    low, high = float(quantiles[0]), float(quantiles[1])
    return pd.Series(np.where(values <= low, 0, np.where(values <= high, 1, 2)), index=values.index, dtype=np.int64)


def build_v21_sequence_bundle(frame: pd.DataFrame, *, sequence_len: int, max_rows: int, clip_value: float = 8.0) -> V21SequenceBundle:
    numeric = frame.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    if max_rows > 0:
        numeric = numeric.tail(max_rows).copy()
    numeric = numeric.dropna().reset_index(drop=True)
    feature_columns = [column for column in numeric.columns if column not in TARGET_EXCLUDE]
    if len(numeric) <= sequence_len:
        raise ValueError("Not enough rows after cleaning to build V21 sequences.")

    stat_cut = max(int(len(numeric) * 0.9), sequence_len + 1)
    stat_frame = numeric.iloc[:stat_cut][feature_columns]
    feature_mean = stat_frame.mean(axis=0).to_numpy(dtype=np.float32)
    feature_std = stat_frame.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0).to_numpy(dtype=np.float32)
    feature_std[feature_std < 1e-6] = 1.0

    values = numeric[feature_columns].to_numpy(dtype=np.float32)
    values = np.clip((values - feature_mean) / feature_std, -float(clip_value), float(clip_value))

    regime_ids = pd.to_numeric(numeric.get("hmm_state"), errors="coerce").fillna(0).clip(lower=0, upper=5).astype(np.int64)
    dir_15 = pd.to_numeric(numeric.get("target_up_15m"), errors="coerce").fillna(0.0).astype(np.float32)
    dir_30 = pd.to_numeric(numeric.get("target_up_30m"), errors="coerce").fillna(0.0).astype(np.float32)
    vol_proxy = _vol_bucket(pd.to_numeric(numeric.get("future_return_30m"), errors="coerce").fillna(0.0).abs())
    range_bucket = _range_bucket(pd.to_numeric(numeric.get("range_forward_15m"), errors="coerce").fillna(0.0))

    regime_values = regime_ids.to_numpy(dtype=np.int64)
    dir15_values = dir_15.to_numpy(dtype=np.float32)
    dir30_values = dir_30.to_numpy(dtype=np.float32)
    vol_values = vol_proxy.to_numpy(dtype=np.int64)
    range_values = range_bucket.to_numpy(dtype=np.int64)

    x_rows: list[np.ndarray] = []
    regime_rows: list[np.ndarray] = []
    y_dir15: list[float] = []
    y_dir30: list[float] = []
    y_vol: list[int] = []
    y_regime: list[int] = []
    y_range: list[int] = []
    for index in range(sequence_len, len(numeric)):
        x_rows.append(values[index - sequence_len : index])
        regime_rows.append(regime_values[index - sequence_len : index])
        y_dir15.append(float(dir15_values[index]))
        y_dir30.append(float(dir30_values[index]))
        y_vol.append(int(vol_values[index]))
        y_regime.append(int(regime_values[index]))
        y_range.append(int(range_values[index]))

    dataset = TensorDataset(
        torch.tensor(np.asarray(x_rows), dtype=torch.float32),
        torch.tensor(np.asarray(regime_rows), dtype=torch.long),
        torch.tensor(np.asarray(y_dir15), dtype=torch.float32),
        torch.tensor(np.asarray(y_dir30), dtype=torch.float32),
        torch.tensor(np.asarray(y_vol), dtype=torch.long),
        torch.tensor(np.asarray(y_regime), dtype=torch.long),
        torch.tensor(np.asarray(y_range), dtype=torch.long),
    )
    return V21SequenceBundle(
        feature_columns=feature_columns,
        dataset=dataset,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )


__all__ = ["V21SequenceBundle", "TARGET_EXCLUDE", "build_v21_sequence_bundle"]
