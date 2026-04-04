from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None

try:
    from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore
except ImportError:  # pragma: no cover
    HistGradientBoostingClassifier = None

from src.v9.branch_features_v9 import BRANCH_FEATURES_V9
from src.v9.selector_torch import score_selector_torch, train_selector_torch


ANALOG_FEATURES = (
    "analog_density",
    "analog_disagreement_v9",
    "analog_weighted_accuracy",
    "regime_match_x_analog",
    "analog_density_x_regime_persistence",
    "memory_bank_confidence",
    "memory_bank_alignment",
)

QUANT_FEATURES = (
    "garch_zscore",
    "fair_value_distance",
    "fair_value_mean_reversion_prob",
    "hmm_regime_probability",
    "regime_transition_risk_v9",
    "volatility_realism_x_fair_value",
)


def _require_pandas() -> None:
    if pd is None:  # pragma: no cover
        raise ImportError("pandas is required for selector experiments.")


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    lower = float(values.min())
    upper = float(values.max())
    if upper - lower <= 1e-6:
        return np.full_like(values, 0.5, dtype=np.float32)
    return (values - lower) / (upper - lower)


def _time_split(frame, validation_fraction: float = 0.2):
    sample_ids = np.asarray(sorted(frame["sample_id"].unique().tolist()), dtype=np.int64)
    validation_count = min(max(int(len(sample_ids) * validation_fraction), 1), max(len(sample_ids) - 1, 1))
    split_at = len(sample_ids) - validation_count
    train_ids = set(sample_ids[:split_at].tolist())
    valid_ids = set(sample_ids[split_at:].tolist())
    train_frame = frame.loc[frame["sample_id"].isin(train_ids)].copy().reset_index(drop=True)
    valid_frame = frame.loc[frame["sample_id"].isin(valid_ids)].copy().reset_index(drop=True)
    return train_frame, valid_frame


def _fit_binary_ranker(train_frame, score_frame, feature_names: tuple[str, ...], target_col: str) -> np.ndarray:
    usable = tuple(feature for feature in feature_names if feature in train_frame.columns)
    if not usable:
        return score_frame["composite_score"].to_numpy(dtype=np.float32)
    if HistGradientBoostingClassifier is None:
        return score_frame["composite_score"].to_numpy(dtype=np.float32)
    labels = train_frame[target_col].to_numpy(dtype=np.float32)
    if len(np.unique(labels)) < 2:
        return score_frame["composite_score"].to_numpy(dtype=np.float32)
    model = HistGradientBoostingClassifier(max_depth=6, max_iter=250, learning_rate=0.05, random_state=42)
    model.fit(train_frame[list(usable)].to_numpy(dtype=np.float32), labels)
    return np.asarray(model.predict_proba(score_frame[list(usable)].to_numpy(dtype=np.float32))[:, 1], dtype=np.float32)


def _fit_regime_specific_ranker(train_frame, valid_frame, feature_names: tuple[str, ...], target_col: str) -> np.ndarray:
    usable = tuple(feature for feature in feature_names if feature in train_frame.columns)
    if not usable or HistGradientBoostingClassifier is None:
        return valid_frame["composite_score"].to_numpy(dtype=np.float32)
    scores = np.zeros(len(valid_frame), dtype=np.float32)
    global_scores = _fit_binary_ranker(train_frame, valid_frame, usable, target_col)
    for regime, score_rows in valid_frame.groupby("dominant_regime", sort=False).groups.items():
        regime_train = train_frame.loc[train_frame["dominant_regime"] == regime]
        if len(regime_train) < 256 or len(np.unique(regime_train[target_col].to_numpy(dtype=np.float32))) < 2:
            scores[np.asarray(list(score_rows), dtype=np.int64)] = global_scores[np.asarray(list(score_rows), dtype=np.int64)]
            continue
        model = HistGradientBoostingClassifier(max_depth=5, max_iter=180, learning_rate=0.05, random_state=42)
        model.fit(regime_train[list(usable)].to_numpy(dtype=np.float32), regime_train[target_col].to_numpy(dtype=np.float32))
        local_scores = model.predict_proba(valid_frame.loc[score_rows, list(usable)].to_numpy(dtype=np.float32))[:, 1]
        scores[np.asarray(list(score_rows), dtype=np.int64)] = np.asarray(local_scores, dtype=np.float32)
    return scores


def evaluate_selector_scores(frame, score_column: str) -> dict[str, Any]:
    top1 = []
    top3 = []
    cone = []
    event_win = []
    unit_pnl = []
    minority_rescue = []
    regime_summary: dict[str, dict[str, float]] = {}
    for _, sample in frame.groupby("sample_id", sort=False):
        ranked = sample.sort_values(score_column, ascending=False)
        selected = ranked.iloc[0]
        best = sample.sort_values("composite_score", ascending=False).iloc[0]
        sample_level = sample.iloc[0]
        top1.append(float(int(selected["branch_id"] == best["branch_id"])))
        top3.append(float(int(best["branch_id"] in set(ranked.head(3)["branch_id"].tolist()))))
        cone.append(float(bool(sample_level.get("inside_confidence_cone", False))))
        pnl = float(selected["branch_direction"]) * float(selected["actual_final_return"])
        unit_pnl.append(pnl)
        event_win.append(float(int(np.sign(pnl) > 0)))
        minority_branch = int(sample_level.get("minority_rescue_branch", -1))
        minority_rescue.append(float(int(minority_branch >= 0 and int(selected["branch_id"]) == minority_branch)))
        regime = str(selected.get("dominant_regime", "unknown"))
        bucket = regime_summary.setdefault(regime, {"count": 0.0, "top1": 0.0, "top3": 0.0, "event": 0.0})
        bucket["count"] += 1.0
        bucket["top1"] += top1[-1]
        bucket["top3"] += top3[-1]
        bucket["event"] += event_win[-1]
    for bucket in regime_summary.values():
        count = max(bucket["count"], 1.0)
        bucket["top1_accuracy"] = round(bucket["top1"] / count, 6)
        bucket["top3_containment"] = round(bucket["top3"] / count, 6)
        bucket["event_win_rate"] = round(bucket["event"] / count, 6)
    return {
        "sample_count": int(frame["sample_id"].nunique()),
        "branch_rows": int(len(frame)),
        "top1_branch_accuracy": round(float(np.mean(top1)) if top1 else 0.0, 6),
        "top3_branch_containment": round(float(np.mean(top3)) if top3 else 0.0, 6),
        "confidence_cone_containment_rate": round(float(np.mean(cone)) if cone else 0.0, 6),
        "event_driven_15m_win_rate": round(float(np.mean(event_win)) if event_win else 0.0, 6),
        "event_driven_15m_avg_unit_pnl": round(float(np.mean(unit_pnl)) if unit_pnl else 0.0, 6),
        "minority_branch_rescue_rate": round(float(np.mean(minority_rescue)) if minority_rescue else 0.0, 6),
        "regime_breakdown": regime_summary,
    }


def run_selector_experiments(frame, *, validation_fraction: float = 0.2) -> dict[str, Any]:
    _require_pandas()
    working = frame.copy().reset_index(drop=True)
    train_frame, valid_frame = _time_split(working, validation_fraction=validation_fraction)
    results: dict[str, Any] = {
        "split": {
            "train_samples": int(train_frame["sample_id"].nunique()),
            "validation_samples": int(valid_frame["sample_id"].nunique()),
            "validation_fraction": float(validation_fraction),
        }
    }

    valid_frame["selector_a_score"] = _fit_binary_ranker(train_frame, valid_frame, tuple(BRANCH_FEATURES_V9), "composite_winner_label")
    results["selector_a"] = evaluate_selector_scores(valid_frame, "selector_a_score")

    valid_frame["selector_c_score"] = _fit_regime_specific_ranker(train_frame, valid_frame, tuple(BRANCH_FEATURES_V9), "composite_winner_label")
    results["selector_c"] = evaluate_selector_scores(valid_frame, "selector_c_score")

    valid_frame["selector_d_score"] = _fit_binary_ranker(train_frame, valid_frame, tuple(BRANCH_FEATURES_V9), "is_top_3_branch")
    results["selector_d"] = evaluate_selector_scores(valid_frame, "selector_d_score")

    analog_features = tuple(feature for feature in ANALOG_FEATURES if feature in working.columns)
    valid_frame["selector_e_score"] = _fit_binary_ranker(train_frame, valid_frame, analog_features, "composite_winner_label")
    results["selector_e"] = evaluate_selector_scores(valid_frame, "selector_e_score")

    quant_features = tuple(feature for feature in QUANT_FEATURES if feature in working.columns)
    valid_frame["selector_f_score"] = _fit_binary_ranker(train_frame, valid_frame, quant_features, "composite_winner_label")
    results["selector_f"] = evaluate_selector_scores(valid_frame, "selector_f_score")

    torch_model, torch_report = train_selector_torch(train_frame, feature_names=BRANCH_FEATURES_V9, epochs=6, batch_size=512, validation_fraction=validation_fraction)
    valid_frame["selector_b_score"] = score_selector_torch(torch_model, valid_frame, BRANCH_FEATURES_V9, device=torch_report.device)
    results["selector_b"] = evaluate_selector_scores(valid_frame, "selector_b_score") | {
        "device": torch_report.device,
        "feature_count": torch_report.feature_count,
        "train_loss": round(float(torch_report.train_loss), 6),
        "validation_loss": round(float(torch_report.validation_loss), 6),
    }

    ensemble = (
        0.24 * _normalize_scores(valid_frame["selector_a_score"].to_numpy(dtype=np.float32))
        + 0.18 * _normalize_scores(valid_frame["selector_b_score"].to_numpy(dtype=np.float32))
        + 0.18 * _normalize_scores(valid_frame["selector_c_score"].to_numpy(dtype=np.float32))
        + 0.15 * _normalize_scores(valid_frame["selector_d_score"].to_numpy(dtype=np.float32))
        + 0.15 * _normalize_scores(valid_frame["selector_e_score"].to_numpy(dtype=np.float32))
        + 0.10 * _normalize_scores(valid_frame["selector_f_score"].to_numpy(dtype=np.float32))
    )
    valid_frame["selector_g_score"] = ensemble.astype(np.float32)
    results["selector_g"] = evaluate_selector_scores(valid_frame, "selector_g_score")
    return results


def save_selector_experiments(path_json: Path, path_md: Path, results: dict[str, Any]) -> None:
    path_json.parent.mkdir(parents=True, exist_ok=True)
    path_md.parent.mkdir(parents=True, exist_ok=True)
    path_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = ["# V9 Selector Experiment Results", ""]
    split = results.get("split", {})
    if split:
        lines.extend(
            [
                "## Split",
                f"- train_samples: {split.get('train_samples')}",
                f"- validation_samples: {split.get('validation_samples')}",
                f"- validation_fraction: {split.get('validation_fraction')}",
                "",
            ]
        )
    for name, payload in results.items():
        if name == "split":
            continue
        lines.append(f"## {name}")
        for key, value in payload.items():
            if isinstance(value, dict):
                continue
            lines.append(f"- {key}: {value}")
        lines.append("")
    path_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
