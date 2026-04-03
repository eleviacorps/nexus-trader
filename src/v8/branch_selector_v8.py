from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None

try:
    from xgboost import XGBRanker, XGBClassifier  # type: ignore
except ImportError:  # pragma: no cover
    XGBRanker = None
    XGBClassifier = None

try:
    from lightgbm import LGBMRanker, LGBMClassifier  # type: ignore
except ImportError:  # pragma: no cover
    LGBMRanker = None
    LGBMClassifier = None

try:
    from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore
except ImportError:  # pragma: no cover
    HistGradientBoostingClassifier = None


BRANCH_SELECTOR_FEATURES_V8: tuple[str, ...] = (
    "generator_probability",
    "hmm_regime_match",
    "hmm_persistence",
    "hmm_transition_risk",
    "volatility_realism",
    "branch_move_zscore",
    "fair_value_dislocation",
    "mean_reversion_pressure",
    "analog_similarity",
    "analog_disagreement",
    "news_consistency",
    "crowd_consistency",
    "macro_alignment",
    "branch_direction",
    "branch_move_size",
    "branch_volatility",
    "vwap_distance",
    "atr_normalized_move",
    "branch_entropy",
    "branch_confidence",
)


@dataclass(frozen=True)
class BranchArchiveSummary:
    samples: int
    branches: int
    winner_positive_rate: float
    avg_path_error: float


class BranchSelectorV8:
    def __init__(self, payload: Mapping[str, Any] | None = None) -> None:
        self.payload = dict(payload or {})

    @staticmethod
    def _build_model() -> tuple[Any, str]:
        if XGBRanker is not None:
            return XGBRanker(n_estimators=220, max_depth=5, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, objective="rank:pairwise", random_state=42), "xgboost_ranker"
        if LGBMRanker is not None:
            return LGBMRanker(n_estimators=220, max_depth=5, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, objective="lambdarank", random_state=42, verbose=-1), "lightgbm_ranker"
        if XGBClassifier is not None:
            return XGBClassifier(n_estimators=220, max_depth=5, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, objective="binary:logistic", eval_metric="logloss", random_state=42), "xgboost_classifier"
        if LGBMClassifier is not None:
            return LGBMClassifier(n_estimators=220, max_depth=5, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, objective="binary", random_state=42, verbose=-1), "lightgbm_classifier"
        if HistGradientBoostingClassifier is not None:
            return HistGradientBoostingClassifier(max_depth=5, max_iter=220, learning_rate=0.05, random_state=42), "hist_gradient_boosting"
        raise ImportError("No supported v8 selector backend is installed.")

    @classmethod
    def fit(cls, frame) -> "BranchSelectorV8":
        features = frame[list(BRANCH_SELECTOR_FEATURES_V8)].to_numpy(dtype=np.float32)
        labels = frame["winner_label"].to_numpy(dtype=np.float32)
        groups = frame.groupby("sample_id").size().to_numpy(dtype=np.int32)
        if len(np.unique(labels)) < 2:
            return cls({"available": False, "provider": "none", "feature_names": list(BRANCH_SELECTOR_FEATURES_V8)})
        try:
            model, provider = cls._build_model()
        except ImportError:
            return cls({"available": False, "provider": "fallback", "feature_names": list(BRANCH_SELECTOR_FEATURES_V8)})
        if "ranker" in provider:
            model.fit(features, labels, group=groups)
        else:
            model.fit(features, labels)
        return cls({"available": True, "provider": provider, "feature_names": list(BRANCH_SELECTOR_FEATURES_V8), "model": model})

    def score(self, frame) -> np.ndarray:
        features = frame[list(BRANCH_SELECTOR_FEATURES_V8)].to_numpy(dtype=np.float32)
        if not self.payload.get("available", False):
            return self.fallback_score(frame)
        model = self.payload.get("model")
        provider = str(self.payload.get("provider", ""))
        scores = model.predict(features) if "ranker" in provider else model.predict_proba(features)[:, 1]
        return np.asarray(scores, dtype=np.float32)

    @staticmethod
    def fallback_score(frame) -> np.ndarray:
        linear = (
            0.24 * frame["hmm_regime_match"].to_numpy(dtype=np.float32)
            + 0.18 * frame["volatility_realism"].to_numpy(dtype=np.float32)
            + 0.14 * frame["analog_similarity"].to_numpy(dtype=np.float32)
            + 0.10 * frame["macro_alignment"].to_numpy(dtype=np.float32)
            + 0.08 * frame["news_consistency"].to_numpy(dtype=np.float32)
            + 0.08 * frame["crowd_consistency"].to_numpy(dtype=np.float32)
            + 0.08 * frame["branch_confidence"].to_numpy(dtype=np.float32)
            - 0.10 * np.abs(frame["branch_move_zscore"].to_numpy(dtype=np.float32))
            - 0.08 * np.abs(frame["fair_value_dislocation"].to_numpy(dtype=np.float32))
            - 0.06 * frame["analog_disagreement"].to_numpy(dtype=np.float32)
        )
        return (1.0 / (1.0 + np.exp(-linear))).astype(np.float32)


def score_branch_row_v8(row: Mapping[str, Any], selected_score: float, winning_score: float) -> dict[str, Any]:
    return {
        "selected_score": round(float(selected_score), 6),
        "winning_score": round(float(winning_score), 6),
        "selected_margin": round(float(selected_score - winning_score), 6),
        "selected_branch_id": row.get("branch_id"),
        "winner_label": int(row.get("winner_label", 0)),
    }


def build_branch_archive_frame(rows: list[dict[str, Any]]):
    if pd is None:
        raise ImportError("pandas is required for v8 branch archive handling.")
    return pd.DataFrame(rows)


def summarize_branch_archive(frame) -> BranchArchiveSummary:
    return BranchArchiveSummary(
        samples=int(frame["sample_id"].nunique()) if len(frame) else 0,
        branches=int(len(frame)),
        winner_positive_rate=float(frame["winner_label"].mean()) if len(frame) else 0.0,
        avg_path_error=float(frame["path_error"].mean()) if len(frame) else 0.0,
    )


def train_branch_selector_v8(frame, output_path: Path) -> dict[str, Any]:
    selector = BranchSelectorV8.fit(frame)
    payload = dict(selector.payload)
    model = payload.pop("model", None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if model is not None:
        import pickle
        with output_path.open("wb") as handle:
            pickle.dump({**payload, "model": model}, handle)
    else:
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {key: value for key, value in payload.items() if key != "model"}


def load_branch_selector_v8(path: Path) -> BranchSelectorV8:
    if not path.exists():
        return BranchSelectorV8({"available": False, "provider": "none", "feature_names": list(BRANCH_SELECTOR_FEATURES_V8)})
    if path.suffix.lower() == ".json":
        return BranchSelectorV8(json.loads(path.read_text(encoding="utf-8")))
    import pickle
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        return BranchSelectorV8(payload if isinstance(payload, dict) else {})
    except Exception:
        return BranchSelectorV8(json.loads(path.read_text(encoding="utf-8")))


def evaluate_branch_selector(frame, selector: BranchSelectorV8) -> dict[str, Any]:
    frame = frame.copy()
    frame["selector_score"] = selector.score(frame)
    top1_correct = []
    top3_containment = []
    selected_path_errors = []
    baseline_errors = []
    minority_useful = []
    event_win = []
    regime_stats: dict[str, dict[str, float]] = {}
    for _, group in frame.groupby("sample_id", sort=False):
        ranked = group.sort_values("selector_score", ascending=False)
        winner = group.sort_values("path_error", ascending=True).iloc[0]
        selected = ranked.iloc[0]
        baseline = group.sort_values("generator_probability", ascending=False).iloc[0]
        top1_correct.append(float(int(selected["branch_id"] == winner["branch_id"])))
        top3_containment.append(float(int(winner["branch_id"] in set(ranked.head(3)["branch_id"].tolist()))))
        selected_path_errors.append(float(selected["path_error"]))
        baseline_errors.append(float(baseline["path_error"]))
        minority = ranked.iloc[1] if len(ranked) > 1 else selected
        minority_useful.append(float(int(np.sign(minority["actual_final_return"]) == np.sign(winner["actual_final_return"]) and minority["branch_direction"] != selected["branch_direction"])))
        event_win.append(float(int(np.sign(selected["actual_final_return"]) == np.sign(selected["branch_direction"]))))
        regime = str(selected.get("dominant_regime", "unknown"))
        bucket = regime_stats.setdefault(regime, {"count": 0.0, "top1": 0.0, "event_win": 0.0})
        bucket["count"] += 1.0
        bucket["top1"] += top1_correct[-1]
        bucket["event_win"] += event_win[-1]
    for bucket in regime_stats.values():
        count = max(bucket["count"], 1.0)
        bucket["top1_accuracy"] = round(bucket["top1"] / count, 6)
        bucket["event_win_rate"] = round(bucket["event_win"] / count, 6)
    return {
        "sample_count": int(frame["sample_id"].nunique()) if len(frame) else 0,
        "branch_rows": int(len(frame)),
        "top1_branch_accuracy": round(float(np.mean(top1_correct)) if top1_correct else 0.0, 6),
        "top3_branch_containment": round(float(np.mean(top3_containment)) if top3_containment else 0.0, 6),
        "average_selected_path_error": round(float(np.mean(selected_path_errors)) if selected_path_errors else 0.0, 6),
        "average_generator_baseline_error": round(float(np.mean(baseline_errors)) if baseline_errors else 0.0, 6),
        "selector_error_improvement": round(float(np.mean(baseline_errors) - np.mean(selected_path_errors)) if selected_path_errors else 0.0, 6),
        "event_driven_15m_win_rate": round(float(np.mean(event_win)) if event_win else 0.0, 6),
        "minority_branch_usefulness": round(float(np.mean(minority_useful)) if minority_useful else 0.0, 6),
        "regime_performance": regime_stats,
    }
