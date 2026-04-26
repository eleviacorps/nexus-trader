from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class ConditionalPredictabilityMapper:
    BASELINE_PREDICTORS = (
        "momentum",
        "rsi_extreme",
        "ema_cross",
        "macd",
        "bb_reversion",
        "volume",
    )

    def _value(self, row: Any, name: str, default: float = 0.0) -> float:
        for key in (name, f"bcfe_{name}", f"context_{name}"):
            if hasattr(row, "get"):
                value = row.get(key, None)
            else:
                value = getattr(row, key, None)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if np.isnan(numeric):
                continue
            return numeric
        return float(default)

    def _predict_momentum(self, row: Any) -> bool:
        return self._value(row, "return_1", 0.0) > 0.0

    def _predict_rsi_extreme(self, row: Any) -> bool | None:
        rsi = self._value(row, "rsi_14", 50.0)
        if rsi > 68.0:
            return False
        if rsi < 32.0:
            return True
        return None

    def _predict_ema_cross(self, row: Any) -> bool | None:
        cross = self._value(row, "ema_cross", 0.0)
        if abs(cross) < 1e-9:
            return None
        return cross > 0.0

    def _predict_macd(self, row: Any) -> bool | None:
        hist = self._value(row, "macd_hist", 0.0)
        if abs(hist) < 1e-6:
            return None
        return hist > 0.0

    def _predict_bb_reversion(self, row: Any) -> bool | None:
        bb = self._value(row, "bb_pct", 0.5)
        if bb > 0.92:
            return False
        if bb < 0.08:
            return True
        return None

    def _predict_volume_confirm(self, row: Any) -> bool | None:
        volume_ratio = self._value(row, "volume_ratio", 1.0)
        if volume_ratio < 1.5:
            return None
        return self._value(row, "return_1", 0.0) > 0.0

    def baseline_votes(self, row: Any) -> dict[str, bool]:
        predictors = {
            "momentum": self._predict_momentum(row),
            "rsi_extreme": self._predict_rsi_extreme(row),
            "ema_cross": self._predict_ema_cross(row),
            "macd": self._predict_macd(row),
            "bb_reversion": self._predict_bb_reversion(row),
            "volume": self._predict_volume_confirm(row),
        }
        return {name: vote for name, vote in predictors.items() if vote is not None}

    def score_row(self, row: Any) -> dict[str, Any]:
        active = self.baseline_votes(row)
        if not active:
            return {
                "predictability": 0.5,
                "agreement": 0.0,
                "n_active": 0,
                "directional_bias": 0.0,
                "votes": {},
            }

        bullish_share = float(np.mean([1.0 if vote else 0.0 for vote in active.values()]))
        agreement = float(abs(bullish_share - 0.5) * 2.0)
        active_ratio = float(len(active) / len(self.BASELINE_PREDICTORS))
        predictability = float(np.clip(0.5 + (0.5 * agreement * active_ratio), 0.5, 1.0))
        directional_bias = float((bullish_share * 2.0) - 1.0)
        return {
            "predictability": predictability,
            "agreement": agreement,
            "n_active": int(len(active)),
            "directional_bias": directional_bias,
            "votes": active,
        }

    def label_bar(self, row: Any, actual_direction: bool) -> dict[str, Any]:
        live = self.score_row(row)
        active = live["votes"]
        if not active:
            return live
        correct = sum(1 for vote in active.values() if bool(vote) == bool(actual_direction))
        return {
            "predictability": float(correct / len(active)),
            "agreement": float(live["agreement"]),
            "n_active": int(len(active)),
            "directional_bias": float(live["directional_bias"]),
            "votes": active,
        }

    def label_archive(self, features_df: pd.DataFrame, *, return_column: str = "return_1") -> pd.DataFrame:
        predictability_scores: list[float] = []
        agreement_scores: list[float] = []
        n_active_scores: list[int] = []
        directional_bias_scores: list[float] = []

        for idx in range(len(features_df) - 1):
            row = features_df.iloc[idx]
            actual_direction = float(features_df.iloc[idx + 1][return_column]) > 0.0
            result = self.label_bar(row, actual_direction)
            predictability_scores.append(float(result["predictability"]))
            agreement_scores.append(float(result["agreement"]))
            n_active_scores.append(int(result["n_active"]))
            directional_bias_scores.append(float(result["directional_bias"]))

        predictability_scores.append(0.5)
        agreement_scores.append(0.0)
        n_active_scores.append(0)
        directional_bias_scores.append(0.0)

        result_df = features_df.copy()
        result_df["cpm_predictability"] = np.asarray(predictability_scores, dtype=np.float32)
        result_df["cpm_agreement"] = np.asarray(agreement_scores, dtype=np.float32)
        result_df["cpm_n_active"] = np.asarray(n_active_scores, dtype=np.int16)
        result_df["cpm_directional_bias"] = np.asarray(directional_bias_scores, dtype=np.float32)
        return result_df

    def summarize_distribution(self, labeled: pd.DataFrame) -> dict[str, float | int]:
        scores = labeled.get("cpm_predictability", pd.Series(dtype=float)).astype(float)
        if scores.empty:
            return {
                "row_count": 0,
                "mean_predictability": 0.0,
                "share_above_0_60": 0.0,
                "share_above_0_70": 0.0,
                "share_above_0_80": 0.0,
            }
        return {
            "row_count": int(len(scores)),
            "mean_predictability": float(scores.mean()),
            "share_above_0_60": float((scores > 0.60).mean()),
            "share_above_0_70": float((scores > 0.70).mean()),
            "share_above_0_80": float((scores > 0.80).mean()),
        }
