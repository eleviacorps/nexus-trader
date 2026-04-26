from __future__ import annotations

import json
import sys
from collections import deque
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import MODELS_DIR, OUTPUTS_DIR
from scripts.validate_v24_4_1_codex import (  # type: ignore
    _build_or_load_signals,
    _default_windows,
    _enrich_signals,
    _load_feature_frame,
)
from src.v24_4_2.recovery_runtime import evaluate_candidates_v24_4_2, safe_float
from src.v24_4_2.threshold_optimizer import ThresholdConfig
from src.v25.tradeability_model import TradeabilityModel


MODEL_PATH = MODELS_DIR / "v25" / "tradeability_model.json"
REPORT_PATH = OUTPUTS_DIR / "v25" / "tradeability_training_report.json"
BEST_CONFIG_PATH = OUTPUTS_DIR / "v24_4_2" / "best_threshold_config.json"


def _load_best_config() -> ThresholdConfig:
    if BEST_CONFIG_PATH.exists():
        payload = json.loads(BEST_CONFIG_PATH.read_text(encoding="utf-8"))
        cfg = payload.get("config", {})
        return ThresholdConfig(
            trend_up=float(cfg.get("trend_up", 0.54)),
            trend_down=float(cfg.get("trend_down", 0.64)),
            breakout=float(cfg.get("breakout", 0.58)),
            range_value=float(cfg.get("range_value", 0.60)),
            cooldown_decay=float(cfg.get("cooldown_decay", 0.75)),
            cluster_radius=float(cfg.get("cluster_radius", 0.25)),
            size_multiplier=float(cfg.get("size_multiplier", 1.0)),
        )
    return ThresholdConfig(0.54, 0.64, 0.58, 0.60, 0.75, 0.25, 1.0)


def _build_dataset(config: ThresholdConfig) -> tuple[list[dict[str, Any]], list[float]]:
    items: list[dict[str, Any]] = []
    labels: list[float] = []
    for window in _default_windows():
        frame = _load_feature_frame(window.start, window.end, prelude_days=45)
        signals = _build_or_load_signals(window, frame)
        candidates = _enrich_signals(signals, frame)
        if candidates.empty:
            continue
        trades, _ = evaluate_candidates_v24_4_2(candidates, frame, config)
        if trades.empty:
            continue
        streak = 0.0
        sell_times: deque[pd.Timestamp] = deque(maxlen=16)
        buy_times: deque[pd.Timestamp] = deque(maxlen=16)
        for row in trades.sort_values("signal_time_utc").to_dict(orient="records"):
            timestamp = pd.Timestamp(row.get("signal_time_utc"))
            direction = str(row.get("variant_signal", "HOLD")).upper()
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
            else:
                timestamp = timestamp.tz_convert("UTC")

            side_times = sell_times if direction == "SELL" else buy_times
            while side_times and (timestamp - side_times[0]) > timedelta(minutes=45):
                side_times.popleft()
            cluster_count = float(len(side_times))

            confidence = float(np.clip(safe_float(row.get("strategic_confidence"), 0.5), 0.0, 1.0))
            branch_quality = float(
                np.clip(
                    (0.60 * float(np.clip(safe_float(row.get("cpm_score"), confidence), 0.0, 1.0)))
                    + (0.40 * confidence),
                    0.0,
                    1.0,
                )
            )
            item = {
                "admission_score": float(np.clip(safe_float(row.get("admission_score"), 0.0), 0.0, 1.0)),
                "regime": str(row.get("regime_label_v24_4_2", "unknown")),
                "direction": direction,
                "spread": max(0.0, safe_float(row.get("spread_estimate"), 0.0)),
                "slippage": max(0.0, safe_float(row.get("slippage_estimate"), 0.0)),
                "cabr_score": float(np.clip(safe_float(row.get("cabr_score"), 0.5), 0.0, 1.0)),
                "branch_quality": branch_quality,
                "claude_confidence": confidence,
                "recent_streak": float(streak),
                "cluster_count": cluster_count,
                "meta_year": int(timestamp.year),
                "meta_label": "trade_won_after_costs",
            }
            realized_r = safe_float(row.get("realized_r_scaled"), 0.0)
            label = 1.0 if realized_r > 0.0 else 0.0
            items.append(item)
            labels.append(label)

            side_times.append(timestamp)
            if label > 0.5:
                streak = max(0.0, streak) + 1.0
            else:
                streak = min(0.0, streak) - 1.0

    return items, labels


def _metrics(model: TradeabilityModel, items: list[dict[str, Any]], labels: list[float]) -> dict[str, Any]:
    if not items:
        return {
            "rows": 0,
            "mean_probability": 0.0,
            "precision_at_threshold": 0.0,
            "recall_at_threshold": 0.0,
            "accuracy_at_threshold": 0.0,
            "predicted_positive_rate": 0.0,
        }
    probs = np.asarray([model.predict_probability(item) for item in items], dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    preds = (probs > model.threshold).astype(np.float64)
    tp = float(np.sum((preds == 1.0) & (y == 1.0)))
    fp = float(np.sum((preds == 1.0) & (y == 0.0)))
    fn = float(np.sum((preds == 0.0) & (y == 1.0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = float(np.mean(preds == y))
    return {
        "rows": int(len(items)),
        "mean_probability": float(np.mean(probs)),
        "precision_at_threshold": float(precision),
        "recall_at_threshold": float(recall),
        "accuracy_at_threshold": float(accuracy),
        "predicted_positive_rate": float(np.mean(preds)),
    }


def main() -> None:
    OUTPUTS_DIR.joinpath("v25").mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    config = _load_best_config()
    items, labels = _build_dataset(config)
    if not items:
        raise RuntimeError("No training rows generated for tradeability model.")

    train_idx = [i for i, item in enumerate(items) if int(item.get("meta_year", 0)) <= 2024]
    valid_idx = [i for i, item in enumerate(items) if int(item.get("meta_year", 0)) > 2024]
    if len(train_idx) < 100:
        split = int(len(items) * 0.8)
        train_idx = list(range(split))
        valid_idx = list(range(split, len(items)))

    train_items = [items[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    valid_items = [items[i] for i in valid_idx]
    valid_labels = [labels[i] for i in valid_idx]

    model = TradeabilityModel(threshold=0.62)
    fit_summary = model.fit(train_items, train_labels)
    model.save(MODEL_PATH)

    train_eval = _metrics(model, train_items, train_labels)
    valid_eval = _metrics(model, valid_items, valid_labels)

    report = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "model_path": str(MODEL_PATH),
        "target_label": "trade_won_after_costs",
        "threshold": 0.62,
        "dataset": {
            "rows": int(len(items)),
            "train_rows": int(len(train_items)),
            "valid_rows": int(len(valid_items)),
            "positive_rate": float(np.mean(np.asarray(labels, dtype=np.float64))),
        },
        "training": fit_summary,
        "evaluation": valid_eval,
        "train_evaluation": train_eval,
        "targets": {
            "tradeability_precision_min": 0.65,
            "target_reached": bool(valid_eval.get("precision_at_threshold", 0.0) >= 0.65),
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
