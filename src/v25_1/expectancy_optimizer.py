from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.v24_4_2.recovery_runtime import classify_regime
from src.v25_1.adaptive_expectancy_gate import AdaptiveExpectancyGate
from src.v25_1.branch_ensemble_ranker import BranchEnsembleRanker
from src.v25_1.regime_specific_branch_ranker import RegimeSpecificBranchRanker
from src.v25_1.trade_cluster_filter import TradeClusterFilter


@dataclass(frozen=True)
class ExpectancyConfig:
    trend_up: float
    trend_down: float
    breakout: float
    range_value: float
    unknown: float
    chop: float
    sell_threshold_buffer: float
    cluster_radius: float

    def threshold_map(self) -> dict[str, float]:
        return {
            "trend_up": float(self.trend_up),
            "trend_down": float(self.trend_down),
            "breakout": float(self.breakout),
            "range": float(self.range_value),
            "unknown": float(self.unknown),
            "chop": float(self.chop),
        }


@dataclass(frozen=True)
class ExpectancyOptimizationResult:
    generated_at: str
    config: dict[str, Any]
    aggregate_metrics: dict[str, Any]
    window_metrics: list[dict[str, Any]]
    optimization_objective: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExpectancyOptimizer:
    def __init__(
        self,
        *,
        ranker: RegimeSpecificBranchRanker,
        ensemble_ranker: BranchEnsembleRanker | None = None,
        random_seed: int = 2501,
    ):
        self.ranker = ranker
        self.ensemble_ranker = ensemble_ranker or BranchEnsembleRanker()
        self.random = np.random.default_rng(random_seed)

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _clip_01(value: float) -> float:
        return float(np.clip(float(value), 0.0, 1.0))

    def _proxy_feature_row(self, row: Mapping[str, Any], regime: str) -> dict[str, float]:
        strategic = self._clip_01(self._safe_float(row.get("strategic_confidence"), 0.5))
        cpm = self._clip_01(self._safe_float(row.get("cpm_score"), strategic))
        cabr = self._clip_01(self._safe_float(row.get("cabr_score"), strategic))
        atr_pct = max(self._safe_float(row.get("atr_pct"), 0.001), 1e-6)
        macro_class = self._safe_float(row.get("macro_vol_regime_class"), 0.0)
        return_3 = self._safe_float(row.get("return_3"), 0.0)
        return_12 = self._safe_float(row.get("return_12"), 0.0)
        direction = str(row.get("action", "HOLD")).upper()
        direction_sign = 1.0 if direction == "BUY" else -1.0
        analog_proxy = self._clip_01(0.5 + (0.35 * (cpm - 0.5)) + (0.15 * np.sign(return_12) * direction_sign))
        return {
            "regime_bucket": regime,
            "cabr_score": cabr,
            "analog_similarity": analog_proxy,
            "volatility_realism": self._clip_01(1.0 - min(atr_pct / 0.0032, 1.0)),
            "hmm_regime_match": self._clip_01(1.0 - min(macro_class / 4.0, 1.0)),
            "branch_disagreement": self._clip_01(abs(return_3) / max(atr_pct * 1.3, 1e-6)),
            "news_consistency": self._clip_01(0.5 + (0.25 * np.sign(return_3 * return_12))),
            "crowd_consistency": self._clip_01(0.5 + (0.2 * np.sign(return_12) * direction_sign)),
            "branch_volatility": float(np.clip(atr_pct * 1000.0, 0.0, 8.0)),
            "branch_move_zscore_abs": float(np.clip(abs(return_3) / max(atr_pct, 1e-6), 0.0, 8.0)),
        }

    @staticmethod
    def _realized_metrics(row: Mapping[str, Any], direction: str) -> tuple[float, float]:
        sign = 1.0 if str(direction).upper() == "BUY" else -1.0
        future_return = float(pd.to_numeric(pd.Series([row.get("future_return_15m")]), errors="coerce").fillna(0.0).iloc[0])
        stop_pips = max(float(pd.to_numeric(pd.Series([row.get("stop_pips")]), errors="coerce").fillna(0.0).iloc[0]), 1e-6)
        entry_price = float(pd.to_numeric(pd.Series([row.get("reference_close")]), errors="coerce").fillna(0.0).iloc[0])
        # Legacy scaled-R for drawdown comparability.
        stop_distance = max(stop_pips * 0.1, 1e-4)
        realized_r_scaled = (future_return * sign) / stop_distance
        # Pip-based R for meaningful expectancy.
        realized_pips = future_return * entry_price * 10.0 * sign
        realized_r_pip = realized_pips / stop_pips
        return float(realized_r_scaled), float(realized_r_pip)

    def _simulate_window(
        self,
        frame: pd.DataFrame,
        *,
        config: ExpectancyConfig,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        if frame.empty:
            return frame.copy(), {
                "number_of_trades": 0,
                "participation_rate": 0.0,
                "win_rate": 0.0,
                "expectancy_R": 0.0,
                "max_drawdown": 0.0,
            }
        regime_thresholds = config.threshold_map()
        gate = AdaptiveExpectancyGate(
            regime_thresholds=regime_thresholds,
            sell_threshold_buffer=float(config.sell_threshold_buffer),
            cluster_filter=TradeClusterFilter(max_age_minutes=45, price_radius=float(config.cluster_radius)),
        )
        accepted_rows: list[dict[str, Any]] = []
        historical_expectancy: dict[tuple[str, str], list[float]] = {}
        for row in frame.sort_values("signal_time_utc").to_dict(orient="records"):
            timestamp = pd.Timestamp(row.get("signal_time_utc"))
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
            else:
                timestamp = timestamp.tz_convert("UTC")
            regime = classify_regime(row)
            direction = str(row.get("action", "HOLD")).upper()
            if direction not in {"BUY", "SELL"}:
                continue
            proxy = self._proxy_feature_row(row, regime)
            sequence_realism = float(self.ranker.predict(proxy).probability)
            analog_similarity = float(proxy["analog_similarity"])
            key = (regime, direction)
            history = historical_expectancy.get(key, [])
            historical_mean = float(np.mean(history)) if history else 0.0
            hist_norm = float(np.clip(0.5 + (historical_mean * 0.06), 0.0, 1.0))
            branch_score = self.ensemble_ranker.score(
                cabr=float(proxy["cabr_score"]),
                sequence_realism=sequence_realism,
                analog_similarity=analog_similarity,
                historical_expectancy_norm=hist_norm,
            )
            strategic = self._clip_01(self._safe_float(row.get("strategic_confidence"), 0.5))
            cpm = self._clip_01(self._safe_float(row.get("cpm_score"), strategic))
            base_quality = (0.35 * strategic) + (0.25 * cpm) + (0.20 * float(proxy["cabr_score"])) + (0.20 * branch_score)
            sell_expectancy = float(np.mean(historical_expectancy.get((regime, "SELL"), [0.0])))
            gate_decision = gate.evaluate(
                regime=regime,
                direction=direction,
                score=base_quality,
                timestamp=timestamp.to_pydatetime(),
                entry_price=self._safe_float(row.get("reference_close"), 0.0),
                sell_regime_expectancy=sell_expectancy,
            )
            if not gate_decision.allow:
                continue
            realized_scaled, realized_pip = self._realized_metrics(row, direction)
            gate.record_outcome(realized_pip)
            historical_expectancy.setdefault(key, []).append(realized_pip)
            enriched = dict(row)
            enriched["variant"] = "v25_1"
            enriched["variant_signal"] = direction
            enriched["regime_label_v25_1"] = regime
            enriched["branch_sequence_realism"] = sequence_realism
            enriched["analog_similarity_v25_1"] = analog_similarity
            enriched["historical_expectancy_norm_v25_1"] = hist_norm
            enriched["branch_ensemble_score_v25_1"] = branch_score
            enriched["execution_score_v25_1"] = float(base_quality)
            enriched["buy_threshold_v25_1"] = gate_decision.buy_threshold
            enriched["sell_threshold_v25_1"] = gate_decision.sell_threshold
            enriched["threshold_used_v25_1"] = gate_decision.threshold_used
            enriched["gate_reason_v25_1"] = gate_decision.reason
            enriched["realized_r_scaled_v25_1"] = realized_scaled
            enriched["realized_r_pip_v25_1"] = realized_pip
            accepted_rows.append(enriched)

        trades = pd.DataFrame.from_records(accepted_rows) if accepted_rows else frame.iloc[0:0].copy()
        candidate_count = int(len(frame))
        trade_count = int(len(trades))
        if trade_count == 0:
            metrics = {
                "number_of_trades": 0,
                "participation_rate": 0.0,
                "win_rate": 0.0,
                "expectancy_R": 0.0,
                "max_drawdown": 0.0,
            }
            return trades, metrics
        realized_scaled = pd.to_numeric(trades["realized_r_scaled_v25_1"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        realized_pip = pd.to_numeric(trades["realized_r_pip_v25_1"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        equity = np.cumsum(realized_scaled)
        drawdown = np.maximum.accumulate(equity) - equity
        metrics = {
            "number_of_trades": trade_count,
            "participation_rate": float(trade_count / max(candidate_count, 1)),
            "win_rate": float(np.mean(realized_pip > 0.0)),
            "expectancy_R": float(np.mean(realized_pip)),
            "max_drawdown": float(drawdown.max()) if drawdown.size else 0.0,
        }
        return trades, metrics

    @staticmethod
    def _aggregate_window_metrics(window_metrics: list[dict[str, Any]]) -> dict[str, Any]:
        if not window_metrics:
            return {
                "number_of_trades": 0,
                "participation_rate": 0.0,
                "win_rate": 0.0,
                "expectancy_R": 0.0,
                "max_drawdown": 0.0,
            }
        weights = np.asarray([max(int(item.get("number_of_trades", 0)), 1) for item in window_metrics], dtype=np.float64)
        participation = float(np.mean([item.get("participation_rate", 0.0) for item in window_metrics]))
        win_rate = float(np.average([item.get("win_rate", 0.0) for item in window_metrics], weights=weights))
        expectancy = float(np.average([item.get("expectancy_R", 0.0) for item in window_metrics], weights=weights))
        max_drawdown = float(max([item.get("max_drawdown", 0.0) for item in window_metrics]))
        total_trades = int(sum(int(item.get("number_of_trades", 0)) for item in window_metrics))
        return {
            "number_of_trades": total_trades,
            "participation_rate": participation,
            "win_rate": win_rate,
            "expectancy_R": expectancy,
            "max_drawdown": max_drawdown,
        }

    def evaluate(
        self,
        windows: list[tuple[str, pd.DataFrame]],
        *,
        config: ExpectancyConfig,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, pd.DataFrame]]:
        per_window: list[dict[str, Any]] = []
        trades_by_window: dict[str, pd.DataFrame] = {}
        for label, frame in windows:
            trades, metrics = self._simulate_window(frame, config=config)
            trades_by_window[label] = trades
            per_window.append({"label": label, "metrics": metrics})
        aggregate = self._aggregate_window_metrics([item["metrics"] for item in per_window])
        return aggregate, per_window, trades_by_window

    def optimize(
        self,
        windows: list[tuple[str, pd.DataFrame]],
        *,
        trials: int = 1400,
    ) -> ExpectancyOptimizationResult:
        best: tuple[float, ExpectancyConfig, dict[str, Any], list[dict[str, Any]]] | None = None
        for _ in range(int(trials)):
            config = ExpectancyConfig(
                trend_up=float(self.random.uniform(0.80, 0.89)),
                trend_down=float(self.random.uniform(0.70, 0.86)),
                breakout=float(self.random.uniform(0.74, 0.90)),
                range_value=float(self.random.uniform(0.72, 0.88)),
                unknown=float(self.random.uniform(0.68, 0.82)),
                chop=float(self.random.uniform(0.75, 0.93)),
                sell_threshold_buffer=float(self.random.uniform(0.01, 0.05)),
                cluster_radius=float(self.random.uniform(0.20, 0.50)),
            )
            aggregate, per_window, _ = self.evaluate(windows, config=config)
            participation = float(aggregate.get("participation_rate", 0.0))
            win_rate = float(aggregate.get("win_rate", 0.0))
            expectancy = float(aggregate.get("expectancy_R", 0.0))
            drawdown = float(aggregate.get("max_drawdown", 1.0))
            band_penalty = abs(participation - 0.20) * 3.0
            if not (0.15 <= participation <= 0.25):
                band_penalty += 0.75
            objective = (
                (1.4 * win_rate)
                + (0.25 * min(expectancy, 1.2))
                - (0.8 * drawdown)
                - band_penalty
            )
            if best is None or objective > best[0]:
                best = (objective, config, aggregate, per_window)
        if best is None:
            raise RuntimeError("No optimization trials were evaluated.")
        objective, config, aggregate, per_window = best
        return ExpectancyOptimizationResult(
            generated_at=datetime.now(tz=UTC).isoformat(),
            config=asdict(config),
            aggregate_metrics=aggregate,
            window_metrics=per_window,
            optimization_objective=float(objective),
        )
