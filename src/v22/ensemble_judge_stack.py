from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return float(default)
        return float(number)
    except Exception:
        return float(default)


def _softmax(logits: np.ndarray) -> np.ndarray:
    centered = logits - np.max(logits, axis=-1, keepdims=True)
    exp_values = np.exp(centered)
    return exp_values / np.clip(np.sum(exp_values, axis=-1, keepdims=True), 1e-9, None)


@dataclass(frozen=True)
class LinearMetaLabeler:
    weights: tuple[float, ...]
    bias: float = 0.0

    def predict_proba(self, X: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        array = np.asarray(X, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        weights = np.asarray(self.weights, dtype=np.float32)
        if array.shape[1] < weights.shape[0]:
            padded = np.zeros((array.shape[0], weights.shape[0]), dtype=np.float32)
            padded[:, : array.shape[1]] = array
            array = padded
        elif array.shape[1] > weights.shape[0]:
            weights = np.pad(weights, (0, array.shape[1] - weights.shape[0]), mode="constant")
        logits = array @ weights + float(self.bias)
        positive = 1.0 / (1.0 + np.exp(-logits))
        return np.stack([1.0 - positive, positive], axis=-1)


class EnsembleJudgeStack:
    ACTIONS = ("BUY", "SELL", "HOLD")

    def __init__(
        self,
        students: Sequence[Any],
        conformal_quantiles: Sequence[float] | np.ndarray,
        meta_model: Any | None = None,
        *,
        risk_threshold: float = 0.65,
        meta_threshold: float = 0.40,
        disagreement_threshold: float = 0.8,
    ) -> None:
        self.students = list(students)
        self.quantiles = np.asarray(conformal_quantiles, dtype=np.float32).reshape(-1)
        if self.quantiles.shape[0] != 3:
            raise ValueError("conformal_quantiles must have exactly three action entries.")
        self.meta = meta_model or LinearMetaLabeler(weights=(1.4, -1.0, -1.2, 0.8, -1.1, -0.8), bias=-0.1)
        self.risk_threshold = float(risk_threshold)
        self.meta_threshold = float(meta_threshold)
        self.disagreement_threshold = float(disagreement_threshold)

    def _student_outputs(self, series: Any, quant_features: Any) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for student in self.students:
            raw = student(series, quant_features)
            if not isinstance(raw, Mapping):
                raise TypeError("Each student must return a mapping with action_logits, risk_pred, disagree_prob.")
            outputs.append(dict(raw))
        return outputs

    def _conformal_adjust(self, probs_stack: np.ndarray) -> np.ndarray:
        mean_probs = np.mean(probs_stack, axis=0)
        adjusted = np.clip(mean_probs + self.quantiles, 1e-6, None)
        return adjusted / np.sum(adjusted)

    def _agreement_cap(self, agreement_count: int) -> float:
        if agreement_count >= 5:
            return 0.10
        if agreement_count >= 4:
            return 0.07
        if agreement_count >= 3:
            return 0.05
        return 0.01

    def predict(self, series: Any, quant_features: Any, context_features: Sequence[float] | np.ndarray | None = None) -> dict[str, Any]:
        return self.predict_from_outputs(self._student_outputs(series, quant_features), context_features=context_features)

    def predict_from_outputs(
        self,
        outputs: Sequence[Mapping[str, Any]],
        *,
        context_features: Sequence[float] | np.ndarray | None = None,
    ) -> dict[str, Any]:
        if not outputs:
            return {"action": "HOLD", "reason": "no_students", "confidence": 0.0, "max_lot": 0.01}
        logits_stack = []
        risk_preds = []
        disagree_preds = []
        for output in outputs:
            logits = output.get("action_logits")
            logits_array = logits.detach().cpu().numpy() if isinstance(logits, torch.Tensor) else np.asarray(logits, dtype=np.float32)
            logits_stack.append(logits_array.reshape(-1))
            risk_preds.append(_safe_float(output.get("risk_pred"), 1.0))
            disagree_preds.append(_safe_float(output.get("disagree_prob"), 1.0))
        logits_np = np.vstack(logits_stack).astype(np.float32)
        probs_stack = _softmax(logits_np)
        calibrated = self._conformal_adjust(probs_stack)
        votes = np.argmax(logits_np, axis=-1)
        vote_counts = np.bincount(votes, minlength=len(self.ACTIONS))
        agreement_count = int(np.max(vote_counts))
        agreement_index = int(np.argmax(vote_counts))
        agreement_rate = float(agreement_count / len(outputs))
        vote_std = float(np.std(votes.astype(np.float32)))
        risk_score = float(np.mean(risk_preds))
        disagree_score = float(np.mean(disagree_preds))
        meta_input = np.concatenate(
            [
                calibrated.astype(np.float32),
                np.asarray([agreement_rate, risk_score, disagree_score], dtype=np.float32),
                np.asarray(context_features or [], dtype=np.float32).reshape(-1),
            ],
            axis=0,
        )
        meta_prob = float(self.meta.predict_proba(meta_input)[0, 1])
        conformal_set_size = int(np.sum(calibrated >= max(float(np.max(calibrated) - float(np.max(self.quantiles))), 0.15)))
        max_lot = self._agreement_cap(agreement_count)
        base_payload = {
            "agreement_action": self.ACTIONS[agreement_index],
            "agreement_count": agreement_count,
            "agreement_rate": round(agreement_rate, 6),
            "ensemble_vote_std": round(vote_std, 6),
            "meta_label_prob": round(meta_prob, 6),
            "risk_score": round(risk_score, 6),
            "mean_disagree_prob": round(disagree_score, 6),
            "conformal_set_size": conformal_set_size,
            "max_lot": max_lot,
        }
        if vote_std > self.disagreement_threshold:
            return {"action": "HOLD", "reason": "ensemble_disagree", "confidence": 0.0} | base_payload
        if risk_score > self.risk_threshold:
            return {"action": "HOLD", "reason": "high_risk", "confidence": 0.0} | base_payload
        if meta_prob < self.meta_threshold:
            return {"action": "HOLD", "reason": "meta_reject", "confidence": round(meta_prob, 6)} | base_payload
        action_index = int(np.argmax(calibrated))
        return {
            "action": self.ACTIONS[action_index],
            "reason": "ok",
            "confidence": round(float(np.max(calibrated)), 6),
            "calibrated_probs": calibrated.round(6).tolist(),
        } | base_payload


__all__ = ["EnsembleJudgeStack", "LinearMetaLabeler"]
