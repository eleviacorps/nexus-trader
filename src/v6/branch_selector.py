from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from src.v6.branch_features import BRANCH_FEATURE_NAMES, compute_branch_feature_dict
from src.v6.historical_retrieval import HistoricalRetrievalResult
from src.v6.regime_detection import RegimeDetectionResult
from src.v6.volatility_constraints import VolatilityEnvelope

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


@dataclass(frozen=True)
class BranchSelectionResult:
    selected_index: int
    selected_branch_id: Any
    selected_score: float
    scores: list[float]
    rationale: list[dict[str, Any]]


class BranchSelectorModel:
    def __init__(self, payload: Mapping[str, Any] | None = None) -> None:
        self.payload = dict(payload or {})

    @staticmethod
    def _build_model() -> tuple[Any, str]:
        if XGBClassifier is not None:
            return (
                XGBClassifier(
                    n_estimators=180,
                    max_depth=4,
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
                    n_estimators=180,
                    max_depth=4,
                    learning_rate=0.05,
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
                    max_iter=180,
                    learning_rate=0.05,
                    random_state=42,
                ),
                "hist_gradient_boosting",
            )
        raise ImportError("No supported branch selector backend is installed.")

    @classmethod
    def fit(cls, features: np.ndarray, labels: np.ndarray) -> "BranchSelectorModel":
        features = np.asarray(features, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32)
        if len(np.unique(labels)) < 2:
            return cls({"available": False, "feature_names": list(BRANCH_FEATURE_NAMES), "provider": "none"})
        model, provider = cls._build_model()
        model.fit(features, labels)
        return cls(
            {
                "available": True,
                "provider": provider,
                "feature_names": list(BRANCH_FEATURE_NAMES),
                "model": model,
            }
        )

    def score(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if not self.payload.get("available", False):
            return self.fallback_score(features)
        model = self.payload.get("model")
        return np.clip(model.predict_proba(features)[:, 1], 0.0, 1.0).astype(np.float32)

    @staticmethod
    def fallback_score(features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        idx = {name: pos for pos, name in enumerate(BRANCH_FEATURE_NAMES)}
        linear = (
            0.28 * features[:, idx["regime_match"]]
            + 0.18 * features[:, idx["volatility_match"]]
            + 0.11 * features[:, idx["historical_similarity"]]
            + 0.10 * features[:, idx["news_match"]]
            + 0.08 * features[:, idx["crowd_match"]]
            + 0.07 * features[:, idx["orderflow_match"]]
            + 0.08 * features[:, idx["branch_survival_prior"]]
            + 0.06 * features[:, idx["momentum_persistence"]]
            + 0.04 * features[:, idx["trend_strength"]]
            - 0.10 * np.abs(features[:, idx["move_zscore"]])
            - 0.12 * features[:, idx["implausibility_penalty"]]
            - 0.10 * features[:, idx["constraint_violation"]]
            - 0.04 * features[:, idx["spread_widening"]]
        )
        return (1.0 / (1.0 + np.exp(-linear))).astype(np.float32)


def rank_branches_with_selector(
    branches: Sequence[Mapping[str, Any]],
    current_row: Mapping[str, Any],
    regime: RegimeDetectionResult,
    envelopes: Mapping[int, VolatilityEnvelope],
    retrieval: HistoricalRetrievalResult | None = None,
    selector: BranchSelectorModel | None = None,
) -> BranchSelectionResult:
    selector = selector or BranchSelectorModel()
    feature_dicts: list[dict[str, float]] = []
    feature_matrix: list[np.ndarray] = []
    for branch in branches:
        horizon_guess = 30
        predicted_prices = list(branch.get("predicted_prices", []))
        if len(predicted_prices) <= 1:
            horizon_guess = 5
        elif len(predicted_prices) <= 3:
            horizon_guess = 15
        envelope = envelopes.get(horizon_guess) or next(iter(envelopes.values()))
        feature_dict = compute_branch_feature_dict(branch, current_row, regime, envelope, retrieval)
        feature_matrix.append(np.asarray([feature_dict[name] for name in BRANCH_FEATURE_NAMES], dtype=np.float32))
        feature_dicts.append(feature_dict)
    matrix = np.vstack(feature_matrix) if feature_matrix else np.empty((0, len(BRANCH_FEATURE_NAMES)), dtype=np.float32)
    scores = selector.score(matrix) if len(matrix) else np.empty(0, dtype=np.float32)
    if scores.size == 0:
        return BranchSelectionResult(selected_index=0, selected_branch_id=None, selected_score=0.0, scores=[], rationale=[])
    best_index = int(np.argmax(scores))
    rationale = []
    for branch, feature_dict, score in zip(branches, feature_dicts, scores):
        ordered_drivers = sorted(feature_dict.items(), key=lambda item: abs(item[1]), reverse=True)[:5]
        rationale.append(
            {
                "path_id": branch.get("path_id"),
                "selector_score": round(float(score), 6),
                "top_drivers": [{name: round(float(value), 6)} for name, value in ordered_drivers],
            }
        )
    return BranchSelectionResult(
        selected_index=best_index,
        selected_branch_id=branches[best_index].get("path_id"),
        selected_score=float(scores[best_index]),
        scores=[float(value) for value in scores],
        rationale=rationale,
    )
