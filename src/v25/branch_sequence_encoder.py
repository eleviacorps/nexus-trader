from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class EncodedBranchPath:
    embedding: list[float]
    volatility: float
    acceleration: float
    trend: float


class BranchSequenceEncoder:
    """
    Lightweight GRU-style encoder for short branch paths.
    This keeps inference local and deterministic without introducing heavy runtime deps.
    """

    def __init__(self, input_size: int = 3, hidden_size: int = 12, seed: int = 24) -> None:
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        rng = np.random.default_rng(seed)
        scale = 0.18
        self.w_z = rng.normal(0.0, scale, size=(self.hidden_size, self.input_size))
        self.u_z = rng.normal(0.0, scale, size=(self.hidden_size, self.hidden_size))
        self.w_r = rng.normal(0.0, scale, size=(self.hidden_size, self.input_size))
        self.u_r = rng.normal(0.0, scale, size=(self.hidden_size, self.hidden_size))
        self.w_h = rng.normal(0.0, scale, size=(self.hidden_size, self.input_size))
        self.u_h = rng.normal(0.0, scale, size=(self.hidden_size, self.hidden_size))

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))

    @staticmethod
    def _to_path_array(path: Sequence[float] | Sequence[dict[str, float]]) -> np.ndarray:
        if not path:
            return np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
        if isinstance(path[0], dict):
            values = []
            for point in path:
                if not isinstance(point, dict):
                    continue
                value = point.get("price", point.get("value", point.get("target_price", 0.0)))
                try:
                    values.append(float(value))
                except Exception:
                    values.append(0.0)
            if not values:
                values = [0.0]
            return np.asarray(values, dtype=np.float64)
        return np.asarray([float(item) for item in path], dtype=np.float64)

    def encode_path(self, path: Sequence[float] | Sequence[dict[str, float]]) -> EncodedBranchPath:
        prices = self._to_path_array(path)
        if prices.size < 2:
            return EncodedBranchPath(
                embedding=[0.0 for _ in range(self.hidden_size)],
                volatility=0.0,
                acceleration=0.0,
                trend=0.0,
            )
        diffs = np.diff(prices)
        trend = float((prices[-1] - prices[0]) / max(abs(prices[0]), 1e-6))
        volatility = float(np.std(diffs))
        first_order = np.pad(diffs, (0, 1), mode="edge")
        second_order = np.diff(first_order, prepend=first_order[0])
        acceleration = float(np.mean(np.abs(second_order)))

        # [delta, volatility, acceleration] tokens
        sequence = np.vstack(
            [
                first_order / max(np.std(first_order), 1e-6),
                np.full_like(first_order, volatility),
                np.abs(second_order),
            ]
        ).T.astype(np.float64, copy=False)
        hidden = np.zeros(self.hidden_size, dtype=np.float64)
        for token in sequence:
            z = self._sigmoid(self.w_z @ token + self.u_z @ hidden)
            r = self._sigmoid(self.w_r @ token + self.u_r @ hidden)
            h_candidate = np.tanh(self.w_h @ token + self.u_h @ (r * hidden))
            hidden = (1.0 - z) * hidden + z * h_candidate
        hidden = np.tanh(hidden).astype(np.float64)
        return EncodedBranchPath(
            embedding=[float(v) for v in hidden.tolist()],
            volatility=volatility,
            acceleration=acceleration,
            trend=trend,
        )

    def encode_batch(self, paths: Iterable[Sequence[float] | Sequence[dict[str, float]]]) -> np.ndarray:
        rows = [self.encode_path(path).embedding for path in paths]
        if not rows:
            return np.zeros((0, self.hidden_size), dtype=np.float64)
        return np.asarray(rows, dtype=np.float64)

