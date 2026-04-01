from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from config.project_config import (
    FUSED_FEATURE_MATRIX_PATH,
    FUSED_TIMESTAMPS_PATH,
    MODEL_MANIFEST_PATH,
    PRECISION_GATE_PATH,
    TARGETS_PATH,
    TARGETS_MULTIHORIZON_PATH,
    TFT_CHECKPOINT_PATH,
    VAL_YEARS,
)
from src.data.fused_dataset import DatasetSlice, FusedMultiHorizonSequenceDataset, FusedSequenceDataset
from src.models.nexus_tft import NexusTFT, NexusTFTConfig, load_checkpoint_with_expansion
from src.training.train_tft import apply_precision_gate, build_calibration_report, collect_binary_metrics, collect_multihorizon_metrics

try:
    import torch  # type: ignore
    from torch.utils.data import DataLoader  # type: ignore
except ImportError:  # pragma: no cover
    torch = None
    DataLoader = None


@dataclass(frozen=True)
class FoldReport:
    year: int
    sample_count: int
    threshold: float
    metrics_raw: dict[str, float]
    metrics_calibrated: dict[str, float]
    calibration: dict[str, Any]
    backtest: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_checkpoint() -> Path:
    if TFT_CHECKPOINT_PATH.exists():
        return TFT_CHECKPOINT_PATH
    raise FileNotFoundError("No trained checkpoint available for walk-forward evaluation.")


def load_manifest(path: Path = MODEL_MANIFEST_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError("Model manifest is required for walk-forward evaluation.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_model(device: str | None = None, manifest_path: Path = MODEL_MANIFEST_PATH) -> tuple[Any, dict[str, Any], Any]:
    if torch is None:
        raise ImportError("PyTorch is required for walk-forward evaluation.")
    manifest = load_manifest(manifest_path)
    config_payload = manifest.get("model_config", {})
    feature_dim = int(manifest.get("feature_dim", config_payload.get("input_dim", 100)))
    model = NexusTFT(
        NexusTFTConfig(
            input_dim=feature_dim,
            hidden_dim=int(config_payload.get("hidden_dim", 128)),
            lstm_layers=int(config_payload.get("lstm_layers", 2)),
            dropout=float(config_payload.get("dropout", 0.1)),
            output_dim=int(config_payload.get("output_dim", 1)),
        )
    )
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = Path(manifest.get("checkpoint_path", str(_resolve_checkpoint())))
    load_checkpoint_with_expansion(model, checkpoint_path, new_input_dim=feature_dim)
    model = model.to(target_device)
    model.eval()
    return model, manifest, target_device


def yearly_slices(timestamps: Sequence[str], sequence_len: int, years: Sequence[int] | None = None) -> list[tuple[int, DatasetSlice]]:
    values = np.asarray(timestamps, dtype="datetime64[ns]")
    year_view = values.astype("datetime64[Y]").astype(int) + 1970
    usable = len(values) - sequence_len
    if usable <= 0:
        raise ValueError("Not enough rows for walk-forward evaluation.")
    target_years = year_view[sequence_len - 1 : sequence_len - 1 + usable]
    selected_years = [int(year) for year in (years or sorted(set(target_years.tolist())))]
    output: list[tuple[int, DatasetSlice]] = []
    for year in selected_years:
        positions = np.flatnonzero(target_years == year)
        if positions.size == 0:
            continue
        output.append((year, DatasetSlice(int(positions[0]), int(positions[-1] + 1))))
    return output


def predict_for_slice(
    model: Any,
    device: Any,
    row_slice: DatasetSlice,
    *,
    feature_path: Path = FUSED_FEATURE_MATRIX_PATH,
    target_path: Path = TARGETS_PATH,
    sequence_len: int,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    if torch is None or DataLoader is None:
        raise ImportError("PyTorch is required for walk-forward evaluation.")
    dataset = FusedSequenceDataset(feature_path, target_path, sequence_len=sequence_len, row_slice=row_slice)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            outputs = model(features).detach().cpu().numpy().astype(np.float32)
            probabilities.append(outputs)
            targets.append(labels.detach().cpu().numpy().astype(np.float32))
    probs = np.concatenate(probabilities) if probabilities else np.empty(0, dtype=np.float32)
    trgs = np.concatenate(targets) if targets else np.empty(0, dtype=np.float32)
    return trgs, probs


def predict_multihorizon_for_slice(
    model: Any,
    device: Any,
    row_slice: DatasetSlice,
    *,
    feature_path: Path = FUSED_FEATURE_MATRIX_PATH,
    target_bundle_path: Path = TARGETS_MULTIHORIZON_PATH,
    target_keys: Sequence[str],
    sequence_len: int,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    if torch is None or DataLoader is None:
        raise ImportError("PyTorch is required for walk-forward evaluation.")
    dataset = FusedMultiHorizonSequenceDataset(feature_path, target_bundle_path, target_keys=target_keys, sequence_len=sequence_len, row_slice=row_slice)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            outputs = model(features).detach().cpu().numpy().astype(np.float32)
            probabilities.append(outputs)
            targets.append(labels.detach().cpu().numpy().astype(np.float32))
    probs = np.concatenate(probabilities) if probabilities else np.empty((0, 0), dtype=np.float32)
    trgs = np.concatenate(targets) if targets else np.empty((0, 0), dtype=np.float32)
    return trgs, probs


def apply_bucket_calibration(probabilities: np.ndarray, calibration_report: dict[str, Any]) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    bins = list(calibration_report.get("bins", []))
    if probabilities.size == 0 or not bins:
        return probabilities.copy()
    calibrated = probabilities.copy()
    for bucket in bins:
        left = float(bucket.get("left", 0.0))
        right = float(bucket.get("right", 1.0))
        observed = float(bucket.get("observed_rate", bucket.get("predicted_mean", 0.5)))
        if right >= 1.0:
            mask = (probabilities >= left) & (probabilities <= right)
        else:
            mask = (probabilities >= left) & (probabilities < right)
        calibrated[mask] = observed
    return calibrated


def confidence_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float32)
    return np.clip(np.abs(values - 0.5) * 2.0, 0.0, 1.0)


def directional_backtest(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    decision_threshold: float = 0.5,
    confidence_floor: float = 0.12,
    gate_scores: np.ndarray | None = None,
    gate_threshold: float = 0.5,
) -> dict[str, Any]:
    targets = np.asarray(targets, dtype=np.float32)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    confidence = confidence_from_probabilities(probabilities)
    long_mask = (probabilities >= decision_threshold) & (confidence >= confidence_floor)
    short_mask = (probabilities <= (1.0 - decision_threshold)) & (confidence >= confidence_floor)
    if gate_scores is not None:
        gate_values = np.asarray(gate_scores, dtype=np.float32)
        gate_mask = gate_values >= float(gate_threshold)
        long_mask = long_mask & gate_mask
        short_mask = short_mask & gate_mask
    signals = np.zeros(len(probabilities), dtype=np.int8)
    signals[long_mask] = 1
    signals[short_mask] = -1
    realized = np.where(targets >= 0.5, 1, -1)
    pnl = np.where(signals == 0, 0.0, np.where(signals == realized, 1.0, -1.0))
    cumulative = np.cumsum(pnl)
    peak = np.maximum.accumulate(cumulative) if cumulative.size else np.empty(0, dtype=np.float32)
    drawdown = peak - cumulative if cumulative.size else np.empty(0, dtype=np.float32)
    trade_mask = signals != 0
    wins = (pnl > 0) & trade_mask
    losses = (pnl < 0) & trade_mask
    long_trades = signals == 1
    short_trades = signals == -1
    return {
        "trade_count": int(trade_mask.sum()),
        "hold_count": int((signals == 0).sum()),
        "participation_rate": round(float(trade_mask.mean()) if len(signals) else 0.0, 6),
        "win_rate": round(float(wins.sum() / max(1, trade_mask.sum())), 6),
        "loss_rate": round(float(losses.sum() / max(1, trade_mask.sum())), 6),
        "long_win_rate": round(float(((pnl > 0) & long_trades).sum() / max(1, long_trades.sum())), 6),
        "short_win_rate": round(float(((pnl > 0) & short_trades).sum() / max(1, short_trades.sum())), 6),
        "avg_unit_pnl": round(float(pnl.mean()) if len(pnl) else 0.0, 6),
        "cumulative_unit_pnl": round(float(cumulative[-1]) if len(cumulative) else 0.0, 6),
        "max_drawdown_units": round(float(drawdown.max()) if len(drawdown) else 0.0, 6),
        "decision_threshold": float(decision_threshold),
        "confidence_floor": float(confidence_floor),
        "gate_threshold": float(gate_threshold),
    }


def optimize_backtest_thresholds(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    best = {"decision_threshold": 0.5, "confidence_floor": 0.02, "score": 0.0}
    for threshold in np.linspace(0.5, 0.7, 9):
        for floor in np.linspace(0.01, 0.25, 13):
            report = directional_backtest(targets, probabilities, decision_threshold=float(threshold), confidence_floor=float(floor))
            participation = float(report["participation_rate"])
            if participation < 0.005:
                continue
            score = float(report["avg_unit_pnl"]) * 0.55 + float(report["win_rate"]) * 0.35 + participation * 0.10
            if score > best["score"]:
                best = {"decision_threshold": float(threshold), "confidence_floor": float(floor), "score": float(score)}
    return best


def run_walkforward_evaluation(
    *,
    years: Sequence[int],
    batch_size: int = 1024,
    max_windows_per_year: int = 0,
    max_calibration_windows: int = 12000,
    feature_path: Path = FUSED_FEATURE_MATRIX_PATH,
    target_path: Path = TARGETS_PATH,
    target_bundle_path: Path = TARGETS_MULTIHORIZON_PATH,
    timestamps_path: Path = FUSED_TIMESTAMPS_PATH,
    manifest_path: Path = MODEL_MANIFEST_PATH,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    sequence_len = int(manifest.get("sequence_len", 120))
    model, manifest, device = load_model(manifest_path=manifest_path)
    timestamps = np.load(timestamps_path, mmap_mode="r")
    folds = yearly_slices(timestamps, sequence_len=sequence_len, years=years)
    if not folds:
        raise ValueError("No walk-forward folds were produced for the requested years.")
    multi_horizon = bool(manifest.get("multi_horizon", False)) and target_bundle_path.exists()
    horizon_labels = list(manifest.get("horizon_labels", ["5m"]))
    horizon_keys = [f"target_{label}" for label in horizon_labels] if multi_horizon else []
    threshold = float(manifest.get("classification_threshold", 0.5))
    precision_gate = None
    precision_gate_path = Path(manifest.get("precision_gate_path", str(PRECISION_GATE_PATH)))
    if precision_gate_path.exists():
        precision_gate = json.loads(precision_gate_path.read_text(encoding="utf-8"))

    reports: list[FoldReport] = []
    all_targets: list[np.ndarray] = []
    all_probs_raw: list[np.ndarray] = []
    all_probs_calibrated: list[np.ndarray] = []
    all_probs_multi: list[np.ndarray] = []
    validation_years = manifest.get("split_report", {}).get("val_years", []) or list(VAL_YEARS)
    validation_folds = yearly_slices(timestamps, sequence_len=sequence_len, years=validation_years)
    calibration_targets: list[np.ndarray] = []
    calibration_probabilities: list[np.ndarray] = []
    for _, fold_slice in validation_folds:
        active_slice = fold_slice
        if max_calibration_windows > 0 and len(fold_slice) > max_calibration_windows:
            active_slice = DatasetSlice(fold_slice.start, fold_slice.start + max_calibration_windows)
        if multi_horizon:
            targets_val, probabilities_val = predict_multihorizon_for_slice(
                model,
                device,
                active_slice,
                feature_path=feature_path,
                target_bundle_path=target_bundle_path,
                target_keys=horizon_keys,
                sequence_len=sequence_len,
                batch_size=batch_size,
            )
            calibration_targets.append(targets_val[:, 0])
            calibration_probabilities.append(probabilities_val[:, 0])
            continue
        targets_val, probabilities_val = predict_for_slice(model, device, active_slice, feature_path=feature_path, target_path=target_path, sequence_len=sequence_len, batch_size=batch_size)
        calibration_targets.append(targets_val)
        calibration_probabilities.append(probabilities_val)
    calibration_targets_np = np.concatenate(calibration_targets) if calibration_targets else np.empty(0, dtype=np.float32)
    calibration_probabilities_np = np.concatenate(calibration_probabilities) if calibration_probabilities else np.empty(0, dtype=np.float32)
    global_calibration = build_calibration_report(calibration_targets_np, calibration_probabilities_np)
    optimized_thresholds = optimize_backtest_thresholds(calibration_targets_np, apply_bucket_calibration(calibration_probabilities_np, global_calibration))
    decision_threshold = float(optimized_thresholds["decision_threshold"])
    confidence_floor = float(optimized_thresholds["confidence_floor"])

    for year, fold_slice in folds:
        active_slice = fold_slice
        if max_windows_per_year > 0 and len(fold_slice) > max_windows_per_year:
            active_slice = DatasetSlice(fold_slice.start, fold_slice.start + max_windows_per_year)
        if multi_horizon:
            targets_year, probabilities_year = predict_multihorizon_for_slice(
                model,
                device,
                active_slice,
                feature_path=feature_path,
                target_bundle_path=target_bundle_path,
                target_keys=horizon_keys,
                sequence_len=sequence_len,
                batch_size=batch_size,
            )
            primary_targets_year = targets_year[:, 0]
            primary_probabilities_year = probabilities_year[:, 0]
        else:
            targets_year, probabilities_year = predict_for_slice(
                model,
                device,
                active_slice,
                feature_path=feature_path,
                target_path=target_path,
                sequence_len=sequence_len,
                batch_size=batch_size,
            )
            primary_targets_year = targets_year
            primary_probabilities_year = probabilities_year
        calibrated_probabilities = apply_bucket_calibration(primary_probabilities_year, global_calibration)
        if multi_horizon:
            raw_metrics = collect_multihorizon_metrics(targets_year, probabilities_year, thresholds=[threshold] + [0.5] * (len(horizon_labels) - 1), horizon_labels=horizon_labels)
            calibrated_metrics = {
                "primary": collect_binary_metrics(primary_targets_year, calibrated_probabilities, threshold=threshold),
                "horizons": raw_metrics["horizons"],
            }
        else:
            raw_metrics = collect_binary_metrics(primary_targets_year, primary_probabilities_year, threshold=threshold)
            calibrated_metrics = collect_binary_metrics(primary_targets_year, calibrated_probabilities, threshold=threshold)
        fold_calibration = build_calibration_report(primary_targets_year, calibrated_probabilities)
        gate_scores = None
        gate_threshold = 0.5
        if precision_gate is not None and multi_horizon:
            gate_scores = apply_precision_gate(probabilities_year, precision_gate)
            gate_threshold = float(precision_gate.get("threshold", 0.5))
        backtest = directional_backtest(
            primary_targets_year,
            calibrated_probabilities,
            decision_threshold=decision_threshold,
            confidence_floor=confidence_floor,
            gate_scores=gate_scores,
            gate_threshold=gate_threshold,
        )
        if precision_gate is not None and multi_horizon:
            backtest["precision_gate_positive_rate"] = round(float((gate_scores >= float(precision_gate.get("threshold", 0.5))).mean()), 6)
        report = FoldReport(
            year=int(year),
            sample_count=int(len(primary_targets_year)),
            threshold=threshold,
            metrics_raw=raw_metrics,
            metrics_calibrated=calibrated_metrics,
            calibration=fold_calibration,
            backtest=backtest,
        )
        reports.append(report)
        all_targets.append(primary_targets_year)
        all_probs_raw.append(primary_probabilities_year)
        all_probs_calibrated.append(calibrated_probabilities)
        if multi_horizon:
            all_probs_multi.append(probabilities_year)

    all_targets_np = np.concatenate(all_targets) if all_targets else np.empty(0, dtype=np.float32)
    all_probs_raw_np = np.concatenate(all_probs_raw) if all_probs_raw else np.empty(0, dtype=np.float32)
    all_probs_calibrated_np = np.concatenate(all_probs_calibrated) if all_probs_calibrated else np.empty(0, dtype=np.float32)
    overall = {
        "raw_metrics": collect_binary_metrics(all_targets_np, all_probs_raw_np, threshold=threshold),
        "calibrated_metrics": collect_binary_metrics(all_targets_np, all_probs_calibrated_np, threshold=threshold),
        "calibration": build_calibration_report(all_targets_np, all_probs_calibrated_np),
        "backtest": directional_backtest(
            all_targets_np,
            all_probs_calibrated_np,
            decision_threshold=decision_threshold,
            confidence_floor=confidence_floor,
            gate_scores=apply_precision_gate(np.concatenate(all_probs_multi), precision_gate) if precision_gate is not None and multi_horizon and all_probs_multi else None,
            gate_threshold=float(precision_gate.get("threshold", 0.5)) if precision_gate is not None and multi_horizon else 0.5,
        ),
    }
    if precision_gate is not None:
        overall["precision_gate"] = precision_gate
    return {
        "sequence_len": sequence_len,
        "feature_dim": int(manifest.get("feature_dim", 100)),
        "years": [int(year) for year, _ in folds],
        "calibration_source_years": [int(year) for year in validation_years],
        "optimized_thresholds": optimized_thresholds,
        "multi_horizon": multi_horizon,
        "horizon_labels": horizon_labels,
        "overall": overall,
        "folds": [report.to_dict() for report in reports],
    }
