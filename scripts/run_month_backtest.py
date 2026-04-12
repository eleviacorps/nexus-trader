from __future__ import annotations

import argparse
import calendar
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import LEGACY_PRICE_FEATURES_PARQUET, OUTPUTS_EVAL_DIR, TFT_MODEL_DIR
from src.data.fused_dataset import DatasetSlice
from src.evaluation.walkforward import (
    _combined_gate_scores,
    _parse_horizon_minutes,
    apply_bucket_calibration,
    build_event_driven_backtest,
    directional_backtest,
    load_event_price_frame,
    load_model,
    resolve_gate_context_path,
    resolve_meta_gate_path,
    resolve_precision_gate_path,
    slice_gate_context,
    strategic_view_from_multihorizon,
    predict_multihorizon_for_slice,
)
from src.backtest.engine import capital_backtest_from_unit_pnl, fixed_risk_capital_backtest_from_unit_pnl
from src.training.meta_gate import load_meta_gate
from src.v22.backtest_metrics import attach_v22_month_metrics


def _tagged_eval_path(stem: str, run_tag: str, suffix: str = ".json") -> Path:
    return OUTPUTS_EVAL_DIR / f"{stem}_{run_tag}{suffix}"


def _latest_complete_month(timestamps: np.ndarray) -> str:
    values = np.asarray(timestamps, dtype="datetime64[ns]")
    latest = values[-1]
    latest_month = latest.astype("datetime64[M]")
    latest_day = int((latest - latest_month).astype("timedelta64[D]").astype(int)) + 1
    year = int(str(latest_month)[:4])
    month = int(str(latest_month)[5:7])
    last_day = calendar.monthrange(year, month)[1]
    if latest_day < last_day:
        if month == 1:
            return f"{year - 1}-12"
        return f"{year:04d}-{month - 1:02d}"
    return f"{year:04d}-{month:02d}"


def _month_slice(timestamps: np.ndarray, sequence_len: int, month: str) -> DatasetSlice:
    values = np.asarray(timestamps, dtype="datetime64[ns]")
    usable = len(values) - sequence_len
    if usable <= 0:
        raise ValueError("Not enough rows for sequence generation.")
    target_months = values[sequence_len - 1 : sequence_len - 1 + usable].astype("datetime64[M]")
    desired = np.datetime64(month, "M")
    positions = np.flatnonzero(target_months == desired)
    if positions.size == 0:
        raise ValueError(f"No rows found for month {month}.")
    return DatasetSlice(int(positions[0]), int(positions[-1] + 1))


def _safe_gate_context(path: Path | None, row_slice: DatasetSlice, sequence_len: int) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    try:
        context = slice_gate_context(path, row_slice, sequence_len)
        if context is None or len(context) != len(row_slice):
            return None
        return context
    except Exception:
        return None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a one-month local event-driven backtest using an existing tagged model.")
    parser.add_argument("--run-tag", default="mh12_recent_v8")
    parser.add_argument("--month", default="")
    parser.add_argument("--feature-root", default="cloud/data/features")
    parser.add_argument("--capital", type=float, default=100.0)
    parser.add_argument("--risk-fraction", type=float, default=0.02)
    parser.add_argument("--horizon", default="15m")
    args = parser.parse_args()

    feature_root = Path(args.feature_root)
    manifest_path = TFT_MODEL_DIR / f"model_manifest_{args.run_tag}.json"
    walkforward_path = OUTPUTS_EVAL_DIR / f"walkforward_report_{args.run_tag}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not walkforward_path.exists():
        raise FileNotFoundError(f"Missing walkforward report: {walkforward_path}")

    timestamps_path = feature_root / "timestamps.npy"
    feature_path = feature_root / "fused_features.npy"
    target_bundle_path = feature_root / "targets_multihorizon.npz"
    if not timestamps_path.exists() or not feature_path.exists() or not target_bundle_path.exists():
        raise FileNotFoundError("Full feature bundle is required under feature-root.")

    timestamps = np.load(timestamps_path, mmap_mode="r")
    month = args.month or _latest_complete_month(timestamps)

    model, manifest, device = load_model(manifest_path=manifest_path)
    horizon_labels = list(manifest.get("horizon_labels", ["5m", "10m", "15m", "30m"]))
    if args.horizon not in horizon_labels:
        raise ValueError(f"Horizon {args.horizon} not found in {horizon_labels}.")
    horizon_idx = horizon_labels.index(args.horizon)
    sequence_len = int(manifest.get("sequence_len", 120))
    row_slice = _month_slice(timestamps, sequence_len=sequence_len, month=month)
    target_keys = list(manifest.get("output_labels", []))

    targets, probabilities = predict_multihorizon_for_slice(
        model,
        device,
        row_slice,
        feature_path=feature_path,
        target_bundle_path=target_bundle_path,
        target_keys=target_keys,
        sequence_len=sequence_len,
        batch_size=1024,
        amp_enabled=bool(manifest.get("amp_enabled", False)),
        amp_dtype=str(manifest.get("amp_dtype", "bfloat16")),
    )

    direction_targets, hold_targets, confidence_targets = np.split(targets, 3, axis=1)
    direction_probabilities, hold_probabilities, confidence_probabilities = np.split(probabilities, 3, axis=1)

    walkforward = _load_json(walkforward_path)
    thresholds = walkforward.get("optimized_thresholds", {})
    calibration = walkforward.get("overall", {}).get("calibration", {})

    precision_gate_path = resolve_precision_gate_path(manifest)
    precision_gate = _load_json(precision_gate_path) if precision_gate_path is not None and precision_gate_path.exists() else None
    meta_gate_path = resolve_meta_gate_path(manifest)
    try:
        meta_gate = load_meta_gate(meta_gate_path) if meta_gate_path is not None else None
    except Exception:
        meta_gate = None
    gate_context = _safe_gate_context(resolve_gate_context_path(manifest), row_slice, sequence_len)
    _, _, _, gate_scores = _combined_gate_scores(
        probabilities,
        precision_gate,
        meta_gate,
        context_features=gate_context,
    )

    calibrated_horizon_probabilities = apply_bucket_calibration(direction_probabilities[:, horizon_idx], calibration)
    price_frame = load_event_price_frame()
    hold_bars = _parse_horizon_minutes(args.horizon)
    try:
        report = build_event_driven_backtest(
            direction_targets[:, horizon_idx],
            calibrated_horizon_probabilities,
            row_slice=row_slice,
            sequence_len=sequence_len,
            hold_bars=hold_bars,
            decision_threshold=float(thresholds.get("decision_threshold", 0.53)),
            confidence_floor=float(thresholds.get("confidence_floor", 0.06)),
            gate_scores=gate_scores,
            gate_threshold=float(thresholds.get("gate_threshold", 0.5)),
            hold_probabilities=hold_probabilities[:, horizon_idx],
            hold_threshold=float(thresholds.get("hold_threshold", 0.55)),
            confidence_probabilities=confidence_probabilities[:, horizon_idx],
            price_frame=price_frame,
        )
        report["backtest_engine"] = "event_driven"
    except ValueError as exc:
        report = directional_backtest(
            direction_targets[:, horizon_idx],
            calibrated_horizon_probabilities,
            decision_threshold=float(thresholds.get("decision_threshold", 0.53)),
            confidence_floor=float(thresholds.get("confidence_floor", 0.06)),
            gate_scores=gate_scores,
            gate_threshold=float(thresholds.get("gate_threshold", 0.5)),
            hold_probabilities=hold_probabilities[:, horizon_idx],
            hold_threshold=float(thresholds.get("hold_threshold", 0.55)),
            confidence_probabilities=confidence_probabilities[:, horizon_idx],
        )
        report["backtest_engine"] = "directional_fallback"
        report["event_backtest_error"] = str(exc)

    pnl = np.asarray([float(trade.get("net_unit_pnl", 0.0)) for trade in report.get("trades", [])], dtype=np.float32)
    report["custom_capital_backtests"] = {
        "usd_100": capital_backtest_from_unit_pnl(pnl, initial_capital=float(args.capital), risk_fraction=float(args.risk_fraction)),
        "usd_100_fixed_risk": fixed_risk_capital_backtest_from_unit_pnl(
            pnl,
            initial_capital=float(args.capital),
            risk_fraction=float(args.risk_fraction),
        ),
    }
    report["month"] = month
    report["run_tag"] = args.run_tag
    report["feature_root"] = str(feature_root)
    report["price_path"] = str(LEGACY_PRICE_FEATURES_PARQUET)
    report["selected_horizon"] = args.horizon
    report["row_slice"] = {"start": int(row_slice.start), "stop": int(row_slice.stop), "count": int(len(row_slice))}

    report = attach_v22_month_metrics(report, mode=args.horizon)

    out_path = _tagged_eval_path(f"month_backtest_{month}", args.run_tag)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(str(out_path))
    print(json.dumps(
        {
            "month": month,
            "run_tag": args.run_tag,
            "horizon": args.horizon,
            "trade_count": int(report.get("trade_count", 0)),
            "participation_rate": float(report.get("participation_rate", 0.0)),
            "win_rate": float(report.get("win_rate", 0.0)),
            "avg_unit_pnl": float(report.get("avg_unit_pnl", 0.0)),
            "cumulative_unit_pnl": float(report.get("cumulative_unit_pnl", 0.0)),
            "avg_realized_rr": float(((report.get("trade_health") or {}).get("avg_realized_rr", 0.0) or 0.0)),
            "longest_loss_streak": int(((report.get("trade_health") or {}).get("longest_loss_streak", 0) or 0)),
            "trade_frequency_target_met": bool(((report.get("v22_new_metrics") or {}).get("trade_frequency_target_met", False))),
            "usd_100_fixed_risk_final": float(report["custom_capital_backtests"]["usd_100_fixed_risk"].get("final_capital", 0.0)),
            "usd_100_fixed_risk_max_dd": float(report["custom_capital_backtests"]["usd_100_fixed_risk"].get("max_drawdown_pct", 0.0)),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
