from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.pipeline.fusion import GATE_CONTEXT_COLUMNS
from src.training.train_tft import GATE_FEATURE_NAMES, horizon_agreement_features, split_multihorizon_heads_numpy

try:
    from xgboost import XGBClassifier  # type: ignore
except ImportError:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier  # type: ignore
except ImportError:  # pragma: no cover
    LGBMClassifier = None

try:
    from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore
except ImportError:  # pragma: no cover
    HistGradientBoostingClassifier = None


def prepare_meta_gate_training_data(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    context_features: np.ndarray | None = None,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    features = horizon_agreement_features(probabilities, threshold=threshold, context_features=context_features)
    context_feature_names = list(GATE_CONTEXT_COLUMNS) if context_features is not None else []
    feature_names = list(GATE_FEATURE_NAMES) + context_feature_names
    horizon_count = probabilities.shape[1] if probabilities.shape[1] <= 8 else probabilities.shape[1] // 3
    direction_probabilities, hold_probabilities, confidence_probabilities = split_multihorizon_heads_numpy(probabilities, horizon_count)
    direction_targets, hold_targets, confidence_targets = split_multihorizon_heads_numpy(targets, horizon_count)
    if horizon_count >= 2:
        strategic_dir_pred = (direction_probabilities[:, -2:].mean(axis=1) >= threshold).astype(np.float32)
        strategic_dir_target = (direction_targets[:, -2:].mean(axis=1) >= 0.5).astype(np.float32)
        strategic_hold_target = (hold_targets[:, -2:].mean(axis=1) >= 0.5).astype(np.float32)
        strategic_conf_target = confidence_targets[:, -2:].mean(axis=1)
        strategic_hold_pred = hold_probabilities[:, -2:].mean(axis=1)
        target_direction_agreement = ((direction_targets[:, -1] >= 0.5) == (direction_targets[:, -2] >= 0.5)).astype(np.float32)
    else:
        strategic_dir_pred = (direction_probabilities[:, 0] >= threshold).astype(np.float32)
        strategic_dir_target = direction_targets[:, 0]
        strategic_hold_target = hold_targets[:, 0]
        strategic_conf_target = confidence_targets[:, 0]
        strategic_hold_pred = hold_probabilities[:, 0]
        target_direction_agreement = np.ones(len(strategic_dir_target), dtype=np.float32)

    labels = (
        (strategic_dir_pred == strategic_dir_target).astype(np.float32)
        * (strategic_hold_target < 0.45).astype(np.float32)
        * (strategic_hold_pred < 0.58).astype(np.float32)
        * (strategic_conf_target >= 0.42).astype(np.float32)
        * (target_direction_agreement >= 0.5).astype(np.float32)
    ).astype(np.float32)

    if context_features is not None and len(context_feature_names) == np.asarray(context_features).shape[1]:
        context = np.asarray(context_features, dtype=np.float32)
        idx = {name: position for position, name in enumerate(context_feature_names)}
        structural_filter = (
            (context[:, idx["gate_ctx_transition_risk"]] < 0.70)
            & (context[:, idx["gate_ctx_tail_risk"]] < 0.72)
            & (context[:, idx["gate_ctx_vol_realism"]] > 0.16)
            & (context[:, idx["gate_ctx_route_confidence"]] > 0.18)
            & (context[:, idx["gate_ctx_kalman_dislocation_abs"]] < 0.92)
        ).astype(np.float32)
        labels = (labels * structural_filter).astype(np.float32)

    return features, labels, {
        "feature_names": feature_names,
        "context_feature_names": context_feature_names,
        "positive_rate": float(labels.mean()) if labels.size else 0.0,
    }


def _build_model() -> tuple[Any, str]:
    if XGBClassifier is not None:
        return (
            XGBClassifier(
                n_estimators=140,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
            ),
            "xgboost",
        )
    if LGBMClassifier is not None:
        return (
            LGBMClassifier(
                n_estimators=140,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="binary",
                random_state=42,
                verbose=-1,
            ),
            "lightgbm",
        )
    if HistGradientBoostingClassifier is not None:
        return (
            HistGradientBoostingClassifier(
                max_depth=4,
                max_iter=160,
                learning_rate=0.05,
                random_state=42,
            ),
            "hist_gradient_boosting",
        )
    raise ImportError("A boosted classifier backend is required for the meta gate.")


def train_meta_gate(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    context_features: np.ndarray | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    features, labels, metadata = prepare_meta_gate_training_data(
        probabilities,
        targets,
        context_features=context_features,
        threshold=threshold,
    )
    if len(np.unique(labels)) < 2:
        return {
            "available": False,
            "feature_names": metadata["feature_names"],
            "context_feature_names": metadata["context_feature_names"],
            "positive_rate": metadata["positive_rate"],
            "provider": "none",
        }
    try:
        model, provider = _build_model()
    except Exception:
        return {
            "available": False,
            "feature_names": metadata["feature_names"],
            "context_feature_names": metadata["context_feature_names"],
            "positive_rate": metadata["positive_rate"],
            "provider": "none",
        }
    model.fit(features, labels)
    probabilities_out = np.clip(model.predict_proba(features)[:, 1], 0.0, 1.0).astype(np.float32)
    threshold_value = float(np.quantile(probabilities_out, 0.86)) if probabilities_out.size else 0.5
    return {
        "available": True,
        "provider": provider,
        "feature_names": metadata["feature_names"],
        "context_feature_names": metadata["context_feature_names"],
        "positive_rate": metadata["positive_rate"],
        "threshold": threshold_value,
        "train_participation": float((probabilities_out >= threshold_value).mean()) if probabilities_out.size else 0.0,
        "train_precision": float(labels[probabilities_out >= threshold_value].mean()) if np.any(probabilities_out >= threshold_value) else 0.0,
        "model": model,
    }


def apply_meta_gate(
    probabilities: np.ndarray,
    meta_gate: Mapping[str, Any],
    *,
    context_features: np.ndarray | None = None,
) -> np.ndarray | None:
    if not meta_gate or not meta_gate.get("available", False):
        return None
    model = meta_gate.get("model")
    if model is None:
        return None
    features = horizon_agreement_features(probabilities, context_features=context_features)
    expected = len(meta_gate.get("feature_names", []))
    if expected and features.shape[1] != expected:
        base = horizon_agreement_features(probabilities, context_features=None)
        if base.shape[1] == expected:
            features = base
        else:
            raise ValueError("Meta gate feature dimension does not match the saved model.")
    return np.clip(model.predict_proba(features)[:, 1], 0.0, 1.0).astype(np.float32)


def combine_gate_scores(
    precision_scores: np.ndarray | None,
    meta_scores: np.ndarray | None,
    regret_scores: np.ndarray | None = None,
) -> np.ndarray | None:
    if precision_scores is None and meta_scores is None and regret_scores is None:
        return None
    available = [
        np.asarray(values, dtype=np.float32)
        for values in (precision_scores, meta_scores, regret_scores)
        if values is not None
    ]
    if len(available) == 1:
        return available[0]
    precision = np.asarray(precision_scores, dtype=np.float32) if precision_scores is not None else None
    meta = np.asarray(meta_scores, dtype=np.float32) if meta_scores is not None else None
    regret = np.asarray(regret_scores, dtype=np.float32) if regret_scores is not None else None
    if precision is None:
        precision = 0.5 * available[0] + 0.5 * available[-1]
    if meta is None:
        meta = 0.5 * available[0] + 0.5 * available[-1]
    if regret is None:
        regret = 0.5 * available[0] + 0.5 * available[-1]
    geometric = np.cbrt(np.clip(precision, 0.0, 1.0) * np.clip(meta, 0.0, 1.0) * np.clip(regret, 0.0, 1.0))
    arithmetic = (precision + meta + regret) / 3.0
    softened_floor = np.maximum(0.55 * np.minimum(np.minimum(precision, meta), regret), 0.0)
    return np.clip((0.20 * geometric) + (0.40 * arithmetic) + (0.20 * precision) + (0.10 * meta) + (0.10 * softened_floor), 0.0, 1.0).astype(np.float32)


def save_meta_gate(path: Path, meta_gate: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(dict(meta_gate), handle)


def load_meta_gate(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return payload if isinstance(payload, dict) else None
