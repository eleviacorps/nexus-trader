from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from config.project_config import FUSED_FEATURE_MATRIX_PATH, FUSED_TIMESTAMPS_PATH, GATE_CONTEXT_PATH, V19_SJD_DATASET_PATH, V21_FEATURES_PATH
from src.v24.world_state import build_world_state

DEFAULT_SEQUENCE_FEATURES = (
    "return_1",
    "return_3",
    "return_12",
    "atr_pct",
    "macro_realized_vol_20",
    "macro_vol_regime_class",
    "online_hmm_regime_confidence",
    "direction_sign",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if np.isnan(number) or np.isinf(number):
            return float(default)
        return float(number)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _direction_sign(value: Any) -> float:
    action = str(value or "HOLD").upper()
    if action == "BUY":
        return 1.0
    if action == "SELL":
        return -1.0
    return 0.0


@dataclass(frozen=True)
class TradeQualityDatasetBundle:
    static_features: np.ndarray
    sequence_features: np.ndarray
    targets: dict[str, np.ndarray]
    static_feature_names: tuple[str, ...]
    sequence_feature_names: tuple[str, ...]
    months: tuple[str, ...]
    timestamps: tuple[str, ...]
    metadata: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]

    def month_mask(self, selected_months: Sequence[str]) -> np.ndarray:
        selected = {str(item) for item in selected_months}
        return np.asarray([month in selected for month in self.months], dtype=bool)


def _default_month_debug_paths() -> list[Path]:
    root = Path("outputs") / "v22"
    return sorted(root.glob("month_debug_suite_*.json"))


def _load_feature_frame() -> pd.DataFrame:
    columns = [
        "return_1",
        "return_3",
        "return_12",
        "atr_pct",
        "macro_realized_vol_20",
        "macro_vol_regime_class",
    ]
    frame = pd.read_parquet(V21_FEATURES_PATH, columns=columns).copy()
    frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame = frame.loc[~frame.index.isna()].sort_index()
    return frame


def _load_optional_aligned_features() -> tuple[dict[pd.Timestamp, dict[str, float]], list[str]]:
    warnings: list[str] = []
    if not FUSED_FEATURE_MATRIX_PATH.exists() or not FUSED_TIMESTAMPS_PATH.exists():
        return {}, warnings
    timestamps = pd.to_datetime(np.load(FUSED_TIMESTAMPS_PATH), utc=True, errors="coerce")
    fused = np.load(FUSED_FEATURE_MATRIX_PATH)
    if len(timestamps) != len(fused):
        warnings.append(
            f"Skipped fused feature summaries because lengths differ: timestamps={len(timestamps)} fused={len(fused)}."
        )
        return {}, warnings
    gate = None
    if GATE_CONTEXT_PATH.exists():
        try:
            gate = np.load(GATE_CONTEXT_PATH)
            if len(gate) != len(fused):
                warnings.append(
                    f"Skipped gate summaries because lengths differ: gate={len(gate)} fused={len(fused)}."
                )
                gate = None
        except Exception as exc:
            warnings.append(f"Skipped gate summaries because loading failed: {exc}")
            gate = None
    mapping: dict[pd.Timestamp, dict[str, float]] = {}
    for idx, timestamp in enumerate(timestamps):
        if pd.isna(timestamp):
            continue
        fused_row = np.asarray(fused[idx], dtype=np.float32).reshape(-1)
        summary = {
            "aligned_fused_mean": float(fused_row.mean()) if fused_row.size else 0.0,
            "aligned_fused_std": float(fused_row.std(ddof=0)) if fused_row.size else 0.0,
            "aligned_fused_l2": float(np.linalg.norm(fused_row)) if fused_row.size else 0.0,
            "aligned_fused_absmax": float(np.max(np.abs(fused_row))) if fused_row.size else 0.0,
        }
        if gate is not None:
            gate_row = np.asarray(gate[idx], dtype=np.float32).reshape(-1)
            summary |= {
                "aligned_gate_mean": float(gate_row.mean()) if gate_row.size else 0.0,
                "aligned_gate_std": float(gate_row.std(ddof=0)) if gate_row.size else 0.0,
                "aligned_gate_absmax": float(np.max(np.abs(gate_row))) if gate_row.size else 0.0,
            }
        mapping[pd.Timestamp(timestamp)] = summary
    return mapping, warnings


def _load_sjd_priors() -> dict[str, float]:
    if not V19_SJD_DATASET_PATH.exists():
        return {
            "sjd_hold_prior": 0.0,
            "sjd_high_conf_prior": 0.0,
            "sjd_sell_prior": 0.0,
        }
    frame = pd.read_parquet(V19_SJD_DATASET_PATH)
    if frame.empty:
        return {
            "sjd_hold_prior": 0.0,
            "sjd_high_conf_prior": 0.0,
            "sjd_sell_prior": 0.0,
        }
    stance = frame["stance"].astype(str).str.upper()
    confidence = frame["confidence"].astype(str).str.upper()
    return {
        "sjd_hold_prior": float((stance == "HOLD").mean()),
        "sjd_high_conf_prior": float(confidence.isin({"HIGH", "VERY_HIGH"}).mean()),
        "sjd_sell_prior": float((stance == "SELL").mean()),
    }


def _sequence_window(
    feature_frame: pd.DataFrame,
    *,
    timestamp: pd.Timestamp,
    seq_len: int,
    hmm_confidence: float,
    direction_sign: float,
) -> np.ndarray:
    subset = feature_frame.loc[feature_frame.index <= timestamp].tail(max(1, int(seq_len)))
    if subset.empty:
        return np.zeros((int(seq_len), len(DEFAULT_SEQUENCE_FEATURES)), dtype=np.float32)
    sequence = np.column_stack(
        [
            pd.to_numeric(subset["return_1"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            pd.to_numeric(subset["return_3"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            pd.to_numeric(subset["return_12"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            pd.to_numeric(subset["atr_pct"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            pd.to_numeric(subset["macro_realized_vol_20"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            pd.to_numeric(subset["macro_vol_regime_class"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            np.full(len(subset), float(hmm_confidence), dtype=np.float32),
            np.full(len(subset), float(direction_sign), dtype=np.float32),
        ]
    ).astype(np.float32)
    if len(sequence) < int(seq_len):
        pad = np.zeros((int(seq_len) - len(sequence), sequence.shape[1]), dtype=np.float32)
        sequence = np.vstack([pad, sequence])
    return sequence


def _target_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    loss_magnitude = np.abs(np.minimum(pd.to_numeric(frame["pnl_proxy"], errors="coerce").fillna(0.0), 0.0))
    atr_pct = pd.to_numeric(frame["atr_pct"], errors="coerce").fillna(0.0)
    macro_vol = pd.to_numeric(frame["macro_vol_regime_class"], errors="coerce").fillna(0).astype(int)
    hmm_conf = pd.to_numeric(frame["hmm_confidence"], errors="coerce").fillna(0.5)
    agreement_rate = pd.to_numeric(frame["agreement_rate"], errors="coerce").fillna(0.5)
    disagree_prob = pd.to_numeric(frame["disagree_prob"], errors="coerce").fillna(0.5)
    confidence_rank = pd.to_numeric(frame["confidence_rank"], errors="coerce").fillna(0.0)
    return_3 = pd.to_numeric(frame["return_3"], errors="coerce").fillna(0.0)
    return_12 = pd.to_numeric(frame["return_12"], errors="coerce").fillna(0.0)
    rr_ratio = pd.to_numeric(frame["rr_ratio"], errors="coerce").fillna(0.0)
    pnl_pct = pd.to_numeric(frame["pnl_pct"], errors="coerce").fillna(0.0)
    win_probability = (pd.to_numeric(frame["pnl_proxy"], errors="coerce").fillna(0.0) > 0.0).astype(np.float32)

    regime_q95 = (
        pd.DataFrame({"macro_vol_regime_class": macro_vol, "loss_magnitude": loss_magnitude})
        .groupby("macro_vol_regime_class")["loss_magnitude"]
        .quantile(0.95)
        .to_dict()
    )
    atr_q80 = (
        pd.DataFrame({"macro_vol_regime_class": macro_vol, "atr_pct": atr_pct})
        .groupby("macro_vol_regime_class")["atr_pct"]
        .quantile(0.80)
        .to_dict()
    )

    danger_values: list[float] = []
    uncertainty_values: list[float] = []
    abstain_values: list[float] = []
    expected_drawdown_values: list[float] = []
    expected_value_values: list[float] = []
    realized_rr_values: list[float] = []
    for idx in range(len(frame)):
        group = int(macro_vol.iloc[idx])
        group_loss_q95 = max(float(regime_q95.get(group, loss_magnitude.max() if len(loss_magnitude) else 1.0)), 1e-6)
        group_atr_q80 = max(float(atr_q80.get(group, max(float(atr_pct.max()), 1e-6))), 1e-6)
        loss_component = float(loss_magnitude.iloc[idx] / group_loss_q95) if group_loss_q95 > 0.0 else 0.0
        atr_component = max(0.0, float(atr_pct.iloc[idx] / group_atr_q80) - 1.0)
        danger = float(
            np.clip(
                (0.55 * loss_component)
                + (0.15 * max(0, group - 1) / 2.0)
                + (0.15 * max(0.0, 0.58 - float(hmm_conf.iloc[idx])) * 2.5)
                + (0.15 * atr_component),
                0.0,
                1.0,
            )
        )
        directional_variance = abs(np.tanh(float(return_3.iloc[idx]) * 1000.0) - np.tanh(float(return_12.iloc[idx]) * 800.0))
        uncertainty = float(
            np.clip(
                (0.30 * (1.0 - float(agreement_rate.iloc[idx])))
                + (0.20 * float(disagree_prob.iloc[idx]))
                + (0.20 * max(0.0, 0.60 - float(hmm_conf.iloc[idx])) * 2.5)
                + (0.15 * max(0.0, 0.50 - (float(confidence_rank.iloc[idx]) / 4.0)) * 2.0)
                + (0.15 * directional_variance),
                0.0,
                1.0,
            )
        )
        expected_drawdown = float(
            np.clip(
                (abs(min(float(pnl_pct.iloc[idx]), 0.0)) * 40.0)
                + (float(atr_pct.iloc[idx]) * 6.0)
                + (0.02 * max(0, group - 1)),
                0.0,
                0.25,
            )
        )
        win = float(win_probability.iloc[idx])
        realized_rr = float(max(0.0, float(rr_ratio.iloc[idx])) * win)
        expected_value = float((win * max(0.0, float(rr_ratio.iloc[idx]))) - ((1.0 - win) * (1.0 + (0.5 * danger))))
        abstain = float(
            np.clip(
                (0.50 * (1.0 - win))
                + (0.25 * danger)
                + (0.15 * uncertainty)
                + (0.10 * float(float(rr_ratio.iloc[idx]) < 1.5)),
                0.0,
                1.0,
            )
        )
        danger_values.append(danger)
        uncertainty_values.append(uncertainty)
        abstain_values.append(abstain)
        expected_drawdown_values.append(expected_drawdown)
        expected_value_values.append(expected_value)
        realized_rr_values.append(realized_rr)

    frame["target_expected_value"] = np.asarray(expected_value_values, dtype=np.float32)
    frame["target_win_probability"] = win_probability.to_numpy(dtype=np.float32)
    frame["target_realized_rr"] = np.asarray(realized_rr_values, dtype=np.float32)
    frame["target_expected_drawdown"] = np.asarray(expected_drawdown_values, dtype=np.float32)
    frame["target_danger_score"] = np.asarray(danger_values, dtype=np.float32)
    frame["target_uncertainty"] = np.asarray(uncertainty_values, dtype=np.float32)
    frame["target_abstain_probability"] = np.asarray(abstain_values, dtype=np.float32)
    frame["target_quality_score"] = (
        frame["target_expected_value"] - (0.75 * frame["target_danger_score"]) - (0.50 * frame["target_uncertainty"])
    ).astype(np.float32)
    return frame


def build_trade_quality_dataset(
    report_paths: Sequence[Path] | None = None,
    *,
    sequence_length: int = 16,
    include_experiments: Sequence[str] | None = None,
) -> TradeQualityDatasetBundle:
    selected_paths = [Path(item) for item in (report_paths or _default_month_debug_paths()) if Path(item).exists()]
    if not selected_paths:
        raise FileNotFoundError("No month debug suite artifacts were found for Phase-2 target building.")

    feature_frame = _load_feature_frame()
    aligned_mapping, aligned_warnings = _load_optional_aligned_features()
    sjd_priors = _load_sjd_priors()
    experiment_filter = {str(item) for item in include_experiments} if include_experiments else None

    raw_records: list[dict[str, Any]] = []
    static_payloads: list[dict[str, float]] = []
    sequence_rows: list[np.ndarray] = []
    months: list[str] = []
    timestamps: list[str] = []
    metadata: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str, int]] = set()

    for path in selected_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for result in payload.get("results", []):
            experiment = str(result.get("experiment", ""))
            month = str(result.get("month", ""))
            if experiment_filter is not None and experiment not in experiment_filter:
                continue
            for trade in result.get("trades", []):
                signal_id = int(_safe_int(trade.get("signal_id"), len(raw_records)))
                unique_key = (month, experiment, str(trade.get("signal_time_utc", "")), signal_id)
                if unique_key in seen_keys:
                    continue
                seen_keys.add(unique_key)
                timestamp = pd.Timestamp(trade.get("signal_time_utc"))
                if timestamp.tzinfo is None:
                    timestamp = timestamp.tz_localize("UTC")
                else:
                    timestamp = timestamp.tz_convert("UTC")
                ensemble_state = dict(trade.get("v22_ensemble") or {})
                world_state = build_world_state(
                    trade,
                    live_performance=trade.get("live_performance"),
                    breaker_state=trade.get("v22_breaker"),
                    ensemble_state=ensemble_state,
                    symbol=str(trade.get("symbol", "XAUUSD")),
                )
                static_features = world_state.to_flat_features()
                static_features |= sjd_priors
                aligned_summary = aligned_mapping.get(timestamp, {})
                for key in (
                    "aligned_fused_mean",
                    "aligned_fused_std",
                    "aligned_fused_l2",
                    "aligned_fused_absmax",
                    "aligned_gate_mean",
                    "aligned_gate_std",
                    "aligned_gate_absmax",
                ):
                    static_features[key] = float(aligned_summary.get(key, 0.0))
                direction_sign = _direction_sign(trade.get("decision", trade.get("action")))
                sequence_rows.append(
                    _sequence_window(
                        feature_frame,
                        timestamp=timestamp,
                        seq_len=int(sequence_length),
                        hmm_confidence=_safe_float(trade.get("online_hmm_regime_confidence"), 0.0),
                        direction_sign=direction_sign,
                    )
                )
                static_payloads.append({str(key): float(value) for key, value in static_features.items()})
                raw_records.append(
                    {
                        "month": month,
                        "experiment": experiment,
                        "timestamp": str(timestamp),
                        "rr_ratio": _safe_float(trade.get("rr_ratio"), 0.0),
                        "pnl_proxy": _safe_float(trade.get("pnl_proxy"), 0.0),
                        "pnl_pct": _safe_float(trade.get("pnl_pct"), 0.0),
                        "macro_vol_regime_class": _safe_int(trade.get("macro_vol_regime_class"), 0),
                        "atr_pct": _safe_float(trade.get("atr_pct"), 0.0),
                        "hmm_confidence": _safe_float(trade.get("online_hmm_regime_confidence"), 0.0),
                        "agreement_rate": _safe_float(ensemble_state.get("agreement_rate"), 0.5),
                        "disagree_prob": _safe_float(ensemble_state.get("mean_disagree_prob"), 0.5),
                        "confidence_rank": _safe_int(trade.get("confidence_rank"), 0),
                        "return_3": _safe_float(trade.get("return_3"), 0.0),
                        "return_12": _safe_float(trade.get("return_12"), 0.0),
                    }
                )
                months.append(month)
                timestamps.append(str(timestamp))
                metadata.append(
                    {
                        "month": month,
                        "experiment": experiment,
                        "timestamp": str(timestamp),
                        "symbol": str(trade.get("symbol", "XAUUSD")),
                        "signal_id": signal_id,
                    }
                )

    target_frame = _target_frame(raw_records)
    static_feature_names = tuple(sorted({key for payload in static_payloads for key in payload.keys()}))
    static_matrix = np.asarray(
        [[_safe_float(payload.get(name), 0.0) for name in static_feature_names] for payload in static_payloads],
        dtype=np.float32,
    )
    sequence_matrix = np.asarray(sequence_rows, dtype=np.float32)
    targets = {
        "expected_value": target_frame["target_expected_value"].to_numpy(dtype=np.float32),
        "win_probability": target_frame["target_win_probability"].to_numpy(dtype=np.float32),
        "realized_rr": target_frame["target_realized_rr"].to_numpy(dtype=np.float32),
        "expected_drawdown": target_frame["target_expected_drawdown"].to_numpy(dtype=np.float32),
        "danger_score": target_frame["target_danger_score"].to_numpy(dtype=np.float32),
        "uncertainty": target_frame["target_uncertainty"].to_numpy(dtype=np.float32),
        "abstain_probability": target_frame["target_abstain_probability"].to_numpy(dtype=np.float32),
        "quality_score": target_frame["target_quality_score"].to_numpy(dtype=np.float32),
    }
    return TradeQualityDatasetBundle(
        static_features=static_matrix,
        sequence_features=sequence_matrix,
        targets=targets,
        static_feature_names=static_feature_names,
        sequence_feature_names=tuple(DEFAULT_SEQUENCE_FEATURES),
        months=tuple(months),
        timestamps=tuple(timestamps),
        metadata=tuple(metadata),
        warnings=tuple(aligned_warnings),
    )


__all__ = ["TradeQualityDatasetBundle", "build_trade_quality_dataset"]
