from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
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

from config.project_config import (
    V11_PCOP_STAGE10_MODEL_PATH,
    V11_PCOP_STAGE5_MODEL_PATH,
    V11_SELECTOR_MODEL_PATH,
    V11_SETL_MODEL_PATH,
)
from src.backtest.engine import capital_backtest_from_unit_pnl, fixed_risk_capital_backtest_from_unit_pnl
from src.v9.branch_features_v9 import BRANCH_FEATURES_V9
from src.v11.crowd_state_machine import build_crowd_state_history
from src.v11.path_conditioned_outcome import apply_pcop_model, reweight_branches, train_pcop_model
from src.v11.persistent_world_model import roll_world_state_history
from src.v11.setl import SETL_FEATURES, build_setl_features, optimize_setl_threshold, score_setl_model, train_setl_model


@dataclass(frozen=True)
class VariantBacktestResult:
    trade_count: int
    hold_count: int
    participation_rate: float
    win_rate: float
    avg_unit_pnl: float
    cumulative_unit_pnl: float


def _require_pandas() -> None:
    if pd is None:  # pragma: no cover
        raise ImportError("pandas is required for V11 research backtesting.")


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    lower = float(values.min())
    upper = float(values.max())
    if upper - lower <= 1e-6:
        return np.full_like(values, 0.5, dtype=np.float32)
    return (values - lower) / (upper - lower)


def _time_split(frame, validation_fraction: float = 0.2) -> tuple[Any, Any]:
    sample_ids = np.asarray(sorted(frame["sample_id"].unique().tolist()), dtype=np.int64)
    validation_count = min(max(int(len(sample_ids) * validation_fraction), 1), max(len(sample_ids) - 1, 1))
    split_at = len(sample_ids) - validation_count
    train_ids = set(sample_ids[:split_at].tolist())
    valid_ids = set(sample_ids[split_at:].tolist())
    return (
        frame.loc[frame["sample_id"].isin(train_ids)].copy().reset_index(drop=True),
        frame.loc[frame["sample_id"].isin(valid_ids)].copy().reset_index(drop=True),
    )


def augment_v11_context(frame) -> Any:
    _require_pandas()
    working = frame.copy().reset_index(drop=True)
    crowd = build_crowd_state_history(working)
    world = roll_world_state_history(working, crowd)
    working = working.merge(crowd, on="sample_id", how="left")
    working = working.merge(world, on="sample_id", how="left")
    sample_stats = (
        working.groupby("sample_id", sort=False)
        .agg(
            cone_width_15m=("predicted_price_15m", lambda values: float(np.max(values) - np.min(values))),
            minority_share=("branch_direction", lambda values: float(np.mean(np.asarray(values, dtype=np.float32) < 0.0))),
        )
        .reset_index()
    )
    working = working.merge(sample_stats, on="sample_id", how="left")
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce").astype(str)
    return working


def _selector_feature_names(frame) -> tuple[str, ...]:
    candidates = list(BRANCH_FEATURES_V9) + [
        "v10_diversity_score",
        "cesm_state_id",
        "cesm_confidence",
        "cesm_transition_score",
        "pmwm_institutional_positioning",
        "pmwm_retail_sentiment_momentum",
        "pmwm_structural_memory_strength",
        "pmwm_regime_persistence",
        "pmwm_smart_money_fingerprint",
        "cone_width_15m",
        "minority_share",
    ]
    return tuple(name for name in candidates if name in frame.columns)


def train_selector_model(frame) -> tuple[Any, tuple[str, ...]]:
    feature_names = _selector_feature_names(frame)
    if HistGradientBoostingClassifier is None or not feature_names:
        return None, feature_names
    labels = frame["composite_winner_label"].to_numpy(dtype=np.float32)
    if len(np.unique(labels)) < 2:
        return None, feature_names
    model = HistGradientBoostingClassifier(max_depth=6, max_iter=260, learning_rate=0.05, random_state=42)
    model.fit(frame[list(feature_names)].fillna(0.0).to_numpy(dtype=np.float32), labels)
    return model, feature_names


def score_selector_model(model, frame, *, feature_names: tuple[str, ...]) -> np.ndarray:
    if model is None or not feature_names:
        return frame["composite_score"].to_numpy(dtype=np.float32)
    values = frame[list(feature_names)].fillna(0.0).to_numpy(dtype=np.float32)
    return np.asarray(model.predict_proba(values)[:, 1], dtype=np.float32)


def _entry_price(row: Any, stage_bars: int) -> float:
    if stage_bars == 0:
        return float(row["entry_open_price"])
    if stage_bars == 5:
        return float(row["actual_price_5m"])
    if stage_bars == 10:
        return float(row["actual_price_10m"])
    raise ValueError("stage_bars must be one of 0, 5, 10.")


def _unit_pnl(entry_price: float, exit_price: float, direction: float, volatility_scale: float, confidence: float) -> float:
    gross_return = float(direction) * ((float(exit_price) - float(entry_price)) / max(float(entry_price), 1e-6))
    if gross_return > 0:
        gross_unit = 1.0
    elif gross_return < 0:
        gross_unit = -1.0
    else:
        gross_unit = 0.0
    fee_penalty = 0.0004
    slippage_penalty = 0.0002 + (0.0002 * min(max(float(volatility_scale), 0.0), 4.0) / 4.0) + (0.0001 * (1.0 - min(max(float(confidence), 0.0), 1.0)))
    return float(gross_unit - fee_penalty - (2.0 * slippage_penalty))


def _select_rows(frame, score_column: str) -> Any:
    ranked = frame.sort_values(["sample_id", score_column, "branch_confidence"], ascending=[True, False, False], kind="mergesort")
    return ranked.groupby("sample_id", sort=False).head(1).copy().reset_index(drop=True)


def build_stage_dataset(frame, *, stage_bars: int, selector_score_column: str, pcop_score_column: str | None = None):
    working = frame.copy()
    score_column = pcop_score_column or selector_score_column
    selected = _select_rows(working, score_column)
    selected = build_setl_features(selected, stage_bars=stage_bars)
    entry_prices = np.asarray([_entry_price(row, stage_bars) for row in selected.to_dict(orient="records")], dtype=np.float32)
    actual_exit = selected["actual_price_15m"].to_numpy(dtype=np.float32)
    trade_direction = selected["setl_trade_direction"].to_numpy(dtype=np.float32)
    confidence = np.clip(
        0.45 * selected["branch_confidence"].to_numpy(dtype=np.float32)
        + 0.30 * selected.get(pcop_score_column or "", 0.5 if pcop_score_column else selected[selector_score_column]).to_numpy(dtype=np.float32)
        if pcop_score_column
        else 0.65 * selected["branch_confidence"].to_numpy(dtype=np.float32) + 0.35 * selected[selector_score_column].to_numpy(dtype=np.float32),
        0.0,
        1.0,
    )
    selected["setl_target_net_unit_pnl"] = np.asarray(
        [
            _unit_pnl(entry, exit_price, direction, vol, conf)
            for entry, exit_price, direction, vol, conf in zip(
                entry_prices,
                actual_exit,
                trade_direction,
                selected["volatility_scale"].to_numpy(dtype=np.float32),
                confidence,
                strict=False,
            )
        ],
        dtype=np.float32,
    )
    return selected


def _variant_report(trade_mask: np.ndarray, pnl: np.ndarray) -> dict[str, Any]:
    active = np.asarray(trade_mask, dtype=bool)
    pnl = np.asarray(pnl, dtype=np.float32)
    filtered = np.where(active, pnl, 0.0).astype(np.float32)
    trade_count = int(active.sum())
    hold_count = int(len(active) - trade_count)
    wins = (filtered > 0) & active
    capital = {
        "usd_10": capital_backtest_from_unit_pnl(filtered, initial_capital=10.0, risk_fraction=0.02),
        "usd_1000": capital_backtest_from_unit_pnl(filtered, initial_capital=1000.0, risk_fraction=0.02),
        "usd_10_fixed_risk": fixed_risk_capital_backtest_from_unit_pnl(filtered, initial_capital=10.0, risk_fraction=0.02),
        "usd_1000_fixed_risk": fixed_risk_capital_backtest_from_unit_pnl(filtered, initial_capital=1000.0, risk_fraction=0.02),
    }
    return {
        "trade_count": trade_count,
        "hold_count": hold_count,
        "participation_rate": round(float(np.mean(active)) if len(active) else 0.0, 6),
        "win_rate": round(float(wins.sum() / max(trade_count, 1)), 6),
        "avg_unit_pnl": round(float(np.mean(filtered[active])) if trade_count else 0.0, 6),
        "cumulative_unit_pnl": round(float(filtered.sum()), 6),
        "capital_backtests": capital,
    }


def _save_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def run_v11_backtest(frame, *, validation_fraction: float = 0.2) -> dict[str, Any]:
    _require_pandas()
    working = augment_v11_context(frame)
    train_frame, valid_frame = _time_split(working, validation_fraction=validation_fraction)

    selector_model, selector_features = train_selector_model(train_frame)
    train_frame["selector_score"] = score_selector_model(selector_model, train_frame, feature_names=selector_features)
    valid_frame["selector_score"] = score_selector_model(selector_model, valid_frame, feature_names=selector_features)

    pcop5_model, pcop5_features = train_pcop_model(train_frame.assign(selector_score=train_frame["selector_score"]), stage_bars=5)
    train_frame["pcop_survival_score_5m"] = apply_pcop_model(pcop5_model, train_frame.assign(selector_score=train_frame["selector_score"]), stage_bars=5, feature_names=pcop5_features)
    valid_frame["pcop_survival_score_5m"] = apply_pcop_model(pcop5_model, valid_frame.assign(selector_score=valid_frame["selector_score"]), stage_bars=5, feature_names=pcop5_features)
    train_frame = reweight_branches(train_frame, survival_scores=train_frame["pcop_survival_score_5m"].to_numpy(dtype=np.float32)).rename(columns={"pcop_conditioned_score": "pcop_conditioned_score_5m"})
    valid_frame = reweight_branches(valid_frame, survival_scores=valid_frame["pcop_survival_score_5m"].to_numpy(dtype=np.float32)).rename(columns={"pcop_conditioned_score": "pcop_conditioned_score_5m"})

    pcop10_model, pcop10_features = train_pcop_model(train_frame.assign(selector_score=train_frame["selector_score"]), stage_bars=10)
    train_frame["pcop_survival_score_10m"] = apply_pcop_model(pcop10_model, train_frame.assign(selector_score=train_frame["selector_score"]), stage_bars=10, feature_names=pcop10_features)
    valid_frame["pcop_survival_score_10m"] = apply_pcop_model(pcop10_model, valid_frame.assign(selector_score=valid_frame["selector_score"]), stage_bars=10, feature_names=pcop10_features)
    train_frame["pcop_conditioned_score_10m"] = (0.55 * train_frame["selector_score"].to_numpy(dtype=np.float32)) + (0.45 * train_frame["pcop_survival_score_10m"].to_numpy(dtype=np.float32))
    valid_frame["pcop_conditioned_score_10m"] = (0.55 * valid_frame["selector_score"].to_numpy(dtype=np.float32)) + (0.45 * valid_frame["pcop_survival_score_10m"].to_numpy(dtype=np.float32))

    train_open = build_stage_dataset(train_frame, stage_bars=0, selector_score_column="selector_score")
    train_5m = build_stage_dataset(train_frame, stage_bars=5, selector_score_column="selector_score", pcop_score_column="pcop_conditioned_score_5m")
    train_10m = build_stage_dataset(train_frame, stage_bars=10, selector_score_column="selector_score", pcop_score_column="pcop_conditioned_score_10m")
    train_setl = pd.concat([train_open, train_5m, train_10m], ignore_index=True)

    setl_model, setl_features = train_setl_model(train_setl, feature_names=SETL_FEATURES)
    _save_pickle(V11_SELECTOR_MODEL_PATH, {"model": selector_model, "features": selector_features})
    _save_pickle(V11_PCOP_STAGE5_MODEL_PATH, {"model": pcop5_model, "features": pcop5_features})
    _save_pickle(V11_PCOP_STAGE10_MODEL_PATH, {"model": pcop10_model, "features": pcop10_features})
    _save_pickle(V11_SETL_MODEL_PATH, {"model": setl_model, "features": setl_features})

    train_setl_scores = score_setl_model(setl_model, train_setl, feature_names=setl_features)
    threshold = optimize_setl_threshold(train_setl_scores, train_setl["setl_target_net_unit_pnl"].to_numpy(dtype=np.float32))

    valid_open = build_stage_dataset(valid_frame, stage_bars=0, selector_score_column="selector_score")
    valid_5m = build_stage_dataset(valid_frame, stage_bars=5, selector_score_column="selector_score", pcop_score_column="pcop_conditioned_score_5m")
    valid_10m = build_stage_dataset(valid_frame, stage_bars=10, selector_score_column="selector_score", pcop_score_column="pcop_conditioned_score_10m")
    for dataset in (valid_open, valid_5m, valid_10m):
        dataset["setl_expected_pnl"] = score_setl_model(setl_model, dataset, feature_names=setl_features)

    baseline = _variant_report(np.ones(len(valid_open), dtype=bool), valid_open["setl_target_net_unit_pnl"].to_numpy(dtype=np.float32))
    setl_open_mask = valid_open["setl_expected_pnl"].to_numpy(dtype=np.float32) >= float(threshold.threshold)
    setl_open = _variant_report(setl_open_mask, valid_open["setl_target_net_unit_pnl"].to_numpy(dtype=np.float32))
    pcop5_mask = valid_5m["setl_expected_pnl"].to_numpy(dtype=np.float32) >= float(threshold.threshold)
    pcop5 = _variant_report(pcop5_mask, valid_5m["setl_target_net_unit_pnl"].to_numpy(dtype=np.float32))
    pcop10_mask = valid_10m["setl_expected_pnl"].to_numpy(dtype=np.float32) >= float(threshold.threshold)
    pcop10 = _variant_report(pcop10_mask, valid_10m["setl_target_net_unit_pnl"].to_numpy(dtype=np.float32))

    merged = (
        valid_open[["sample_id", "setl_expected_pnl", "setl_target_net_unit_pnl"]].rename(columns={"setl_expected_pnl": "pnl_0", "setl_target_net_unit_pnl": "real_0"})
        .merge(valid_5m[["sample_id", "setl_expected_pnl", "setl_target_net_unit_pnl"]].rename(columns={"setl_expected_pnl": "pnl_5", "setl_target_net_unit_pnl": "real_5"}), on="sample_id", how="inner")
        .merge(valid_10m[["sample_id", "setl_expected_pnl", "setl_target_net_unit_pnl"]].rename(columns={"setl_expected_pnl": "pnl_10", "setl_target_net_unit_pnl": "real_10"}), on="sample_id", how="inner")
    )
    stage_scores = merged[["pnl_0", "pnl_5", "pnl_10"]].to_numpy(dtype=np.float32)
    stage_choice = np.argmax(stage_scores, axis=1)
    chosen_expected = stage_scores[np.arange(len(stage_scores)), stage_choice]
    chosen_realized = np.choose(stage_choice, [merged["real_0"].to_numpy(dtype=np.float32), merged["real_5"].to_numpy(dtype=np.float32), merged["real_10"].to_numpy(dtype=np.float32)]).astype(np.float32)
    full_mask = chosen_expected >= float(threshold.threshold)
    full_v11 = _variant_report(full_mask, chosen_realized)
    full_v11["stage_usage"] = {
        "open": int((stage_choice == 0).sum()),
        "pcop_5m": int((stage_choice == 1).sum()),
        "pcop_10m": int((stage_choice == 2).sum()),
    }

    summary = {
        "split": {
            "train_samples": int(train_frame["sample_id"].nunique()),
            "validation_samples": int(valid_frame["sample_id"].nunique()),
            "validation_fraction": float(validation_fraction),
        },
        "setl_threshold": {
            "threshold": round(float(threshold.threshold), 6),
            "train_participation_rate": round(float(threshold.participation_rate), 6),
            "train_avg_unit_pnl": round(float(threshold.avg_unit_pnl), 6),
        },
        "variants": {
            "selector_only_open": baseline,
            "setl_open": setl_open,
            "pcop_5m_setl": pcop5,
            "pcop_10m_setl": pcop10,
            "full_v11": full_v11,
        },
        "context": {
            "crowd_states_seen": sorted(set(valid_frame["cesm_state"].astype(str).unique().tolist())),
            "pmwm_mean_regime_persistence": round(float(valid_frame["pmwm_regime_persistence"].mean()), 6),
            "pcop_stage5_mean_survival": round(float(valid_frame["pcop_survival_score_5m"].mean()), 6),
            "pcop_stage10_mean_survival": round(float(valid_frame["pcop_survival_score_10m"].mean()), 6),
        },
    }
    return summary


def render_v11_markdown(summary: dict[str, Any]) -> str:
    lines = ["# V11 Research Backtest", ""]
    split = summary.get("split", {})
    if split:
        lines.extend(
            [
                "## Split",
                f"- train_samples: `{split.get('train_samples', 0)}`",
                f"- validation_samples: `{split.get('validation_samples', 0)}`",
                f"- validation_fraction: `{split.get('validation_fraction', 0.0)}`",
                "",
            ]
        )
    lines.append("## Variants")
    for name, payload in summary.get("variants", {}).items():
        lines.append(f"### {name}")
        lines.append(f"- trade_count: `{payload.get('trade_count', 0)}`")
        lines.append(f"- participation_rate: `{payload.get('participation_rate', 0.0)}`")
        lines.append(f"- win_rate: `{payload.get('win_rate', 0.0)}`")
        lines.append(f"- avg_unit_pnl: `{payload.get('avg_unit_pnl', 0.0)}`")
        if "stage_usage" in payload:
            lines.append(f"- stage_usage: `{json.dumps(payload['stage_usage'])}`")
        lines.append("")
    return "\n".join(lines).strip() + "\n"
