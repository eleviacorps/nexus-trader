from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BranchEnsembleRanker:
    """
    V25.1 branch ranking blend:
      0.40 * CABR
    + 0.25 * branch sequence realism model
    + 0.20 * historical analog similarity
    + 0.15 * realized historical expectancy of similar trades
    """

    w_cabr: float = 0.40
    w_sequence_realism: float = 0.25
    w_analog_similarity: float = 0.20
    w_historical_expectancy: float = 0.15

    @staticmethod
    def _clip_01(value: Any) -> float:
        try:
            return float(np.clip(float(value), 0.0, 1.0))
        except Exception:
            return 0.0

    def score(
        self,
        *,
        cabr: float,
        sequence_realism: float,
        analog_similarity: float,
        historical_expectancy_norm: float,
    ) -> float:
        return float(
            (self.w_cabr * self._clip_01(cabr))
            + (self.w_sequence_realism * self._clip_01(sequence_realism))
            + (self.w_analog_similarity * self._clip_01(analog_similarity))
            + (self.w_historical_expectancy * self._clip_01(historical_expectancy_norm))
        )

    def apply_dataframe(
        self,
        frame: pd.DataFrame,
        *,
        cabr_col: str,
        sequence_realism_col: str,
        analog_col: str,
        historical_expectancy_col: str,
        output_col: str = "branch_ensemble_score_v25_1",
    ) -> pd.DataFrame:
        if frame.empty:
            output = frame.copy()
            output[output_col] = np.asarray([], dtype=np.float64)
            return output
        output = frame.copy()
        cabr = pd.to_numeric(output[cabr_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        seq = pd.to_numeric(output[sequence_realism_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        analog = pd.to_numeric(output[analog_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        hist = pd.to_numeric(output[historical_expectancy_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        values = (
            (self.w_cabr * np.clip(cabr, 0.0, 1.0))
            + (self.w_sequence_realism * np.clip(seq, 0.0, 1.0))
            + (self.w_analog_similarity * np.clip(analog, 0.0, 1.0))
            + (self.w_historical_expectancy * np.clip(hist, 0.0, 1.0))
        )
        output[output_col] = values.astype(np.float64, copy=False)
        return output
