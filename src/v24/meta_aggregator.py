from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from config.project_config import V24_META_AGGREGATOR_CONFIG, V24_META_AGGREGATOR_PATH
from src.v24.models import MetaAggregatorModel, MetaAggregatorModelConfig
from src.v24.world_state import WorldState


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if np.isnan(number) or np.isinf(number):
            return float(default)
        return float(number)
    except Exception:
        return float(default)


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(value))))


@dataclass(frozen=True)
class MetaAggregatorConfig:
    rr_floor: float = 1.5
    atr_reference: float = 0.0015
    drawdown_reference: float = 0.03
    danger_weight: float = 0.75
    uncertainty_weight: float = 0.50
    heuristic_prior_weight: float = 0.25

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TradeQualityEstimate:
    expected_value: float
    profit_probability: float
    expected_rr: float
    expected_drawdown: float
    uncertainty_score: float
    danger_score: float
    abstain_probability: float
    quality_score: float
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HeuristicMetaAggregator:
    """
    Phase-1 V24 bridge: converts repaired V22 runtime features into trade-quality outputs.
    """

    def __init__(self, config: MetaAggregatorConfig | None = None) -> None:
        self.config = config or MetaAggregatorConfig()

    def _flat(self, world_state: WorldState | Mapping[str, Any]) -> dict[str, float]:
        if isinstance(world_state, WorldState):
            return world_state.to_flat_features()
        if isinstance(world_state, Mapping):
            return {str(key): _safe_float(value, 0.0) for key, value in world_state.items()}
        raise TypeError("world_state must be a WorldState or mapping")

    def predict(
        self,
        world_state: WorldState | Mapping[str, Any],
        *,
        sequence_features: Sequence[Sequence[float]] | np.ndarray | None = None,
    ) -> TradeQualityEstimate:
        _ = sequence_features
        values = self._flat(world_state)
        rr_ratio = max(_safe_float(values.get("execution_context_rr_ratio"), 0.0), 0.0)
        cabr_score = _safe_float(values.get("nexus_features_cabr_score"), 0.5)
        cpm_score = _safe_float(values.get("nexus_features_cpm_score"), 0.5)
        confidence_score = _safe_float(values.get("nexus_features_confidence_tier_score"), 0.0)
        direction_sign = _safe_float(values.get("execution_context_direction_sign"), 0.0)
        hmm_confidence = _safe_float(values.get("quant_models_hmm_confidence"), 0.5)
        macro_vol = _safe_float(values.get("quant_models_macro_vol_regime_class"), 0.0)
        atr_pct = _safe_float(values.get("market_structure_atr_pct"), 0.0)
        return_3 = _safe_float(values.get("market_structure_return_3"), 0.0)
        return_12 = _safe_float(values.get("market_structure_return_12"), 0.0)
        rolling_win_rate = _safe_float(values.get("runtime_state_rolling_win_rate_10"), 0.5)
        consecutive_losses = _safe_float(values.get("runtime_state_consecutive_losses"), 0.0)
        daily_drawdown = abs(min(0.0, _safe_float(values.get("runtime_state_daily_drawdown_pct"), 0.0)))
        recent_direction_bias = _safe_float(values.get("runtime_state_recent_direction_bias"), direction_sign)
        ensemble_risk = _safe_float(values.get("execution_context_v22_risk_score"), 0.35)
        agreement_rate = _safe_float(values.get("execution_context_v22_agreement_rate"), 0.5)
        meta_label_prob = _safe_float(values.get("execution_context_v22_meta_label_prob"), 0.5)

        momentum_edge = direction_sign * np.tanh(return_3 * 1200.0)
        persistence_edge = direction_sign * np.tanh(return_12 * 700.0)
        direction_penalty = max(0.0, -(direction_sign * recent_direction_bias))
        base_edge = (
            1.10 * ((cabr_score - 0.5) * 2.0)
            + 0.95 * ((cpm_score - 0.5) * 2.0)
            + 0.70 * ((confidence_score - 0.5) * 2.0)
            + 0.60 * momentum_edge
            + 0.35 * persistence_edge
            + 0.30 * ((rr_ratio / max(self.config.rr_floor, 1e-6)) - 1.0)
            + 0.20 * ((hmm_confidence - 0.5) * 2.0)
            + 0.20 * ((rolling_win_rate - 0.5) * 2.0)
            + 0.10 * ((meta_label_prob - 0.5) * 2.0)
            - 0.35 * direction_penalty
            - 0.20 * max(0.0, macro_vol - 2.0)
        )
        profit_probability = _sigmoid(base_edge)
        uncertainty_score = float(
            np.clip(
                (0.35 * max(0.0, 0.60 - hmm_confidence) * 2.5)
                + (0.20 * (1.0 - agreement_rate))
                + (0.20 * ensemble_risk)
                + (0.15 * max(0.0, self.config.rr_floor - rr_ratio))
                + (0.10 * abs(momentum_edge - persistence_edge))
                + (0.10 * max(0.0, 0.50 - meta_label_prob)),
                0.0,
                1.0,
            )
        )
        danger_score = float(
            np.clip(
                (0.26 * ensemble_risk)
                + (0.18 * max(0.0, macro_vol - 2.0))
                + (0.18 * max(0.0, (atr_pct / max(self.config.atr_reference, 1e-6)) - 1.0))
                + (0.14 * min(1.0, consecutive_losses / 4.0))
                + (0.14 * min(1.0, daily_drawdown / max(self.config.drawdown_reference, 1e-6)))
                + (0.10 * max(0.0, self.config.rr_floor - rr_ratio)),
                0.0,
                1.0,
            )
        )
        expected_rr = max(0.0, rr_ratio * (0.85 + (0.15 * hmm_confidence)) * (1.0 - (0.18 * uncertainty_score)))
        expected_drawdown = float(
            np.clip(
                (atr_pct * (6.0 + (4.0 * danger_score)))
                + (daily_drawdown * 0.50)
                + (0.002 * consecutive_losses),
                0.0,
                0.25,
            )
        )
        expected_value = float((profit_probability * expected_rr) - ((1.0 - profit_probability) * max(expected_drawdown, 0.5)))
        abstain_probability = _sigmoid(
            ((danger_score - 0.50) * 4.0)
            + ((uncertainty_score - 0.52) * 3.5)
            + (1.4 * max(0.0, self.config.rr_floor - expected_rr))
            + (2.0 * max(0.0, 0.55 - profit_probability))
        )
        quality_score = float(
            np.clip(
                expected_value - (self.config.danger_weight * danger_score) - (self.config.uncertainty_weight * uncertainty_score),
                -2.0,
                2.0,
            )
        )
        notes: list[str] = []
        if rr_ratio < self.config.rr_floor:
            notes.append("rr_below_floor")
        if macro_vol >= 3.0:
            notes.append("macro_volatility")
        if hmm_confidence < 0.55:
            notes.append("weak_regime_confidence")
        if ensemble_risk > 0.60:
            notes.append("elevated_v22_risk")
        return TradeQualityEstimate(
            expected_value=round(expected_value, 6),
            profit_probability=round(profit_probability, 6),
            expected_rr=round(expected_rr, 6),
            expected_drawdown=round(expected_drawdown, 6),
            uncertainty_score=round(uncertainty_score, 6),
            danger_score=round(danger_score, 6),
            abstain_probability=round(float(abstain_probability), 6),
            quality_score=round(quality_score, 6),
            notes=tuple(notes),
        )


class LearnedMetaAggregator:
    """
    Phase-2 V24 learned meta-aggregator with heuristic prior blending.
    """

    def __init__(
        self,
        model: MetaAggregatorModel,
        *,
        static_feature_names: Sequence[str],
        sequence_feature_names: Sequence[str],
        static_mean: Sequence[float],
        static_std: Sequence[float],
        sequence_mean: Sequence[float],
        sequence_std: Sequence[float],
        config: Mapping[str, Any] | None = None,
        heuristic: HeuristicMetaAggregator | None = None,
    ) -> None:
        self.model = model.eval()
        self.static_feature_names = tuple(str(item) for item in static_feature_names)
        self.sequence_feature_names = tuple(str(item) for item in sequence_feature_names)
        self.static_mean = np.asarray(static_mean, dtype=np.float32).reshape(1, -1)
        self.static_std = np.asarray(static_std, dtype=np.float32).reshape(1, -1)
        self.sequence_mean = np.asarray(sequence_mean, dtype=np.float32).reshape(1, 1, -1)
        self.sequence_std = np.asarray(sequence_std, dtype=np.float32).reshape(1, 1, -1)
        self.runtime_config = dict(config or {})
        self.heuristic = heuristic or HeuristicMetaAggregator()
        self.model_config = self.runtime_config.get("model_config", {})
        self.device = next(self.model.parameters()).device

    @classmethod
    def from_artifacts(
        cls,
        checkpoint_path: Path | str = V24_META_AGGREGATOR_PATH,
        config_path: Path | str = V24_META_AGGREGATOR_CONFIG,
    ) -> "LearnedMetaAggregator":
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
        runtime_config = json.loads(Path(config_path).read_text(encoding="utf-8")) if Path(config_path).exists() else {}
        model_config_payload = checkpoint.get("model_config") or runtime_config.get("model_config")
        if not model_config_payload:
            raise ValueError("Meta-aggregator checkpoint is missing model_config.")
        model = MetaAggregatorModel(MetaAggregatorModelConfig(**model_config_payload))
        model.load_state_dict(checkpoint["model_state"])
        return cls(
            model,
            static_feature_names=checkpoint["static_feature_names"],
            sequence_feature_names=checkpoint["sequence_feature_names"],
            static_mean=checkpoint["static_mean"],
            static_std=checkpoint["static_std"],
            sequence_mean=checkpoint["sequence_mean"],
            sequence_std=checkpoint["sequence_std"],
            config=runtime_config,
        )

    def _static_vector(self, world_state: WorldState | Mapping[str, Any]) -> np.ndarray:
        if isinstance(world_state, WorldState):
            flat = world_state.to_flat_features()
        elif isinstance(world_state, Mapping):
            flat = {str(key): _safe_float(value, 0.0) for key, value in world_state.items()}
        else:
            raise TypeError("world_state must be a WorldState or mapping")
        return np.asarray([_safe_float(flat.get(name), 0.0) for name in self.static_feature_names], dtype=np.float32)

    def _sequence_matrix(self, sequence_features: Sequence[Sequence[float]] | np.ndarray | None) -> np.ndarray:
        sequence = np.asarray(sequence_features if sequence_features is not None else [], dtype=np.float32)
        if sequence.ndim == 1:
            sequence = sequence.reshape(-1, len(self.sequence_feature_names)) if sequence.size else np.zeros((0, len(self.sequence_feature_names)), dtype=np.float32)
        if sequence.ndim != 2 or sequence.shape[1] != len(self.sequence_feature_names):
            sequence = np.zeros((int(self.model.config.seq_len), len(self.sequence_feature_names)), dtype=np.float32)
        if sequence.shape[0] < int(self.model.config.seq_len):
            pad = np.zeros((int(self.model.config.seq_len) - sequence.shape[0], sequence.shape[1]), dtype=np.float32)
            sequence = np.vstack([pad, sequence])
        elif sequence.shape[0] > int(self.model.config.seq_len):
            sequence = sequence[-int(self.model.config.seq_len) :, :]
        return sequence.astype(np.float32)

    def _normalize_static(self, vector: np.ndarray) -> np.ndarray:
        return (vector.reshape(1, -1) - self.static_mean) / np.maximum(self.static_std, 1e-6)

    def _normalize_sequence(self, matrix: np.ndarray) -> np.ndarray:
        return (matrix.reshape(1, matrix.shape[0], matrix.shape[1]) - self.sequence_mean) / np.maximum(self.sequence_std, 1e-6)

    def _estimate_from_outputs(self, prediction: Mapping[str, Any], heuristic: TradeQualityEstimate) -> TradeQualityEstimate:
        heuristic_weight = float(np.clip(self.runtime_config.get("heuristic_prior_weight", 0.25), 0.0, 1.0))
        learned_weight = 1.0 - heuristic_weight
        learned = {
            "expected_value": _safe_float(prediction.get("expected_value"), heuristic.expected_value),
            "profit_probability": _safe_float(prediction.get("win_probability"), heuristic.profit_probability),
            "expected_rr": _safe_float(prediction.get("realized_rr"), heuristic.expected_rr),
            "expected_drawdown": _safe_float(prediction.get("expected_drawdown"), heuristic.expected_drawdown),
            "danger_score": _safe_float(prediction.get("danger_score"), heuristic.danger_score),
            "uncertainty_score": _safe_float(prediction.get("uncertainty"), heuristic.uncertainty_score),
            "abstain_probability": _safe_float(prediction.get("abstain_probability"), heuristic.abstain_probability),
        }
        blended = {
            "expected_value": (learned_weight * learned["expected_value"]) + (heuristic_weight * heuristic.expected_value),
            "profit_probability": (learned_weight * learned["profit_probability"]) + (heuristic_weight * heuristic.profit_probability),
            "expected_rr": (learned_weight * learned["expected_rr"]) + (heuristic_weight * heuristic.expected_rr),
            "expected_drawdown": (learned_weight * learned["expected_drawdown"]) + (heuristic_weight * heuristic.expected_drawdown),
            "danger_score": (learned_weight * learned["danger_score"]) + (heuristic_weight * heuristic.danger_score),
            "uncertainty_score": (learned_weight * learned["uncertainty_score"]) + (heuristic_weight * heuristic.uncertainty_score),
            "abstain_probability": (learned_weight * learned["abstain_probability"]) + (heuristic_weight * heuristic.abstain_probability),
        }
        quality_score = blended["expected_value"] - (self.heuristic.config.danger_weight * blended["danger_score"]) - (
            self.heuristic.config.uncertainty_weight * blended["uncertainty_score"]
        )
        notes = tuple(dict.fromkeys(list(heuristic.notes) + ["learned_meta_aggregator", "heuristic_prior_blend"]))
        return TradeQualityEstimate(
            expected_value=round(float(blended["expected_value"]), 6),
            profit_probability=round(float(np.clip(blended["profit_probability"], 0.0, 1.0)), 6),
            expected_rr=round(float(max(0.0, blended["expected_rr"])), 6),
            expected_drawdown=round(float(np.clip(blended["expected_drawdown"], 0.0, 0.25)), 6),
            uncertainty_score=round(float(np.clip(blended["uncertainty_score"], 0.0, 1.0)), 6),
            danger_score=round(float(np.clip(blended["danger_score"], 0.0, 1.0)), 6),
            abstain_probability=round(float(np.clip(blended["abstain_probability"], 0.0, 1.0)), 6),
            quality_score=round(float(np.clip(quality_score, -2.0, 2.0)), 6),
            notes=notes,
        )

    @torch.inference_mode()
    def predict(
        self,
        world_state: WorldState | Mapping[str, Any],
        *,
        sequence_features: Sequence[Sequence[float]] | np.ndarray | None = None,
    ) -> TradeQualityEstimate:
        heuristic_estimate = self.heuristic.predict(world_state, sequence_features=sequence_features)
        static_vector = self._static_vector(world_state)
        sequence_matrix = self._sequence_matrix(sequence_features)
        static_tensor = torch.from_numpy(self._normalize_static(static_vector)).to(self.device, dtype=torch.float32)
        sequence_tensor = torch.from_numpy(self._normalize_sequence(sequence_matrix)).to(self.device, dtype=torch.float32)
        prediction = self.model.predict(sequence_tensor, static_tensor)
        return self._estimate_from_outputs(prediction, heuristic_estimate)


def load_meta_aggregator(
    *,
    preference: str = "auto",
    checkpoint_path: Path | str = V24_META_AGGREGATOR_PATH,
    config_path: Path | str = V24_META_AGGREGATOR_CONFIG,
) -> HeuristicMetaAggregator | LearnedMetaAggregator:
    mode = str(preference or "auto").strip().lower()
    if mode == "heuristic":
        return HeuristicMetaAggregator()
    checkpoint_file = Path(checkpoint_path)
    config_file = Path(config_path)
    if checkpoint_file.exists() and config_file.exists():
        try:
            return LearnedMetaAggregator.from_artifacts(checkpoint_file, config_file)
        except Exception:
            if mode == "learned":
                raise
    return HeuristicMetaAggregator()


__all__ = [
    "HeuristicMetaAggregator",
    "LearnedMetaAggregator",
    "MetaAggregatorConfig",
    "TradeQualityEstimate",
    "load_meta_aggregator",
]
