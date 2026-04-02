from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from config.project_config import (
    FUSED_FEATURE_MATRIX_PATH,
    FUSED_TIMESTAMPS_PATH,
    GATE_CONTEXT_PATH,
    META_GATE_PATH,
    AMP_DTYPE,
    AMP_ENABLED,
    MODEL_MANIFEST_PATH,
    NUM_WORKERS,
    PIN_MEMORY,
    PREFETCH_FACTOR,
    PRECISION_GATE_PATH,
    PERSISTENT_WORKERS,
    TARGETS_PATH,
    TARGETS_MULTIHORIZON_PATH,
    TFT_CHECKPOINT_PATH,
    TFT_MODEL_DIR,
    VAL_YEARS,
)
from src.data.fused_dataset import DatasetSlice, FusedMultiHorizonSequenceDataset, FusedSequenceDataset
from src.models.nexus_tft import NexusTFT, NexusTFTConfig, load_checkpoint_with_expansion
from src.training.meta_gate import apply_meta_gate, combine_gate_scores, load_meta_gate
from src.training.train_tft import (
    apply_precision_gate,
    autocast_context,
    build_calibration_report,
    collect_binary_metrics,
    collect_multihorizon_metrics,
    split_multihorizon_heads_numpy,
)

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


def resolve_checkpoint_path(manifest: dict[str, Any], manifest_path: Path) -> Path:
    configured_path = Path(manifest.get("checkpoint_path", str(_resolve_checkpoint())))
    if configured_path.exists():
        return configured_path
    local_by_name = TFT_MODEL_DIR / configured_path.name
    if local_by_name.exists():
        return local_by_name
    local_by_manifest_tag = TFT_MODEL_DIR / manifest_path.with_suffix(".ckpt").name.replace("model_manifest", "final_tft")
    if local_by_manifest_tag.exists():
        return local_by_manifest_tag
    fallback = _resolve_checkpoint()
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Unable to resolve checkpoint for manifest {manifest_path}. "
        f"Tried {configured_path}, {local_by_name}, and {local_by_manifest_tag}."
    )


def resolve_precision_gate_path(manifest: dict[str, Any]) -> Path | None:
    configured = Path(manifest.get("precision_gate_path", str(PRECISION_GATE_PATH)))
    if configured.exists():
        return configured
    local_by_name = TFT_MODEL_DIR / configured.name
    if local_by_name.exists():
        return local_by_name
    run_tag = str(manifest.get("run_tag", "")).strip()
    if run_tag:
        tagged_local = TFT_MODEL_DIR / f"{PRECISION_GATE_PATH.stem}_{run_tag}{PRECISION_GATE_PATH.suffix}"
        if tagged_local.exists():
            return tagged_local
    if PRECISION_GATE_PATH.exists():
        return PRECISION_GATE_PATH
    return None


def resolve_meta_gate_path(manifest: dict[str, Any]) -> Path | None:
    configured = Path(manifest.get("meta_gate_path", str(META_GATE_PATH)))
    if configured.exists():
        return configured
    local_by_name = TFT_MODEL_DIR / configured.name
    if local_by_name.exists():
        return local_by_name
    run_tag = str(manifest.get("run_tag", "")).strip()
    if run_tag:
        tagged_local = TFT_MODEL_DIR / f"{META_GATE_PATH.stem}_{run_tag}{META_GATE_PATH.suffix}"
        if tagged_local.exists():
            return tagged_local
    if META_GATE_PATH.exists():
        return META_GATE_PATH
    return None


def resolve_gate_context_path(manifest: dict[str, Any]) -> Path | None:
    configured = Path(manifest.get("gate_context_path", str(GATE_CONTEXT_PATH)))
    if configured.exists():
        return configured
    local_by_name = GATE_CONTEXT_PATH.parent / configured.name
    if local_by_name.exists():
        return local_by_name
    if GATE_CONTEXT_PATH.exists():
        return GATE_CONTEXT_PATH
    return None


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
            regime_count=int(config_payload.get("regime_count", 4)),
            router_hidden_dim=int(config_payload.get("router_hidden_dim", 64)),
            router_temperature=float(config_payload.get("router_temperature", 1.0)),
        )
    )
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = resolve_checkpoint_path(manifest, manifest_path)
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


def slice_gate_context(path: Path | None, row_slice: DatasetSlice, sequence_len: int) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    context = np.load(path, mmap_mode="r")
    start = row_slice.start + sequence_len - 1
    stop = row_slice.stop + sequence_len - 1
    return np.asarray(context[start:stop], dtype=np.float32)


def predict_for_slice(
    model: Any,
    device: Any,
    row_slice: DatasetSlice,
    *,
    feature_path: Path = FUSED_FEATURE_MATRIX_PATH,
    target_path: Path = TARGETS_PATH,
    sequence_len: int,
    batch_size: int = 1024,
    amp_enabled: bool = AMP_ENABLED,
    amp_dtype: str = AMP_DTYPE,
) -> tuple[np.ndarray, np.ndarray]:
    if torch is None or DataLoader is None:
        raise ImportError("PyTorch is required for walk-forward evaluation.")
    dataset = FusedSequenceDataset(feature_path, target_path, sequence_len=sequence_len, row_slice=row_slice)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, **_loader_kwargs())
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device, non_blocking=True)
            with autocast_context(device, amp_enabled, amp_dtype):
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
    amp_enabled: bool = AMP_ENABLED,
    amp_dtype: str = AMP_DTYPE,
) -> tuple[np.ndarray, np.ndarray]:
    if torch is None or DataLoader is None:
        raise ImportError("PyTorch is required for walk-forward evaluation.")
    dataset = FusedMultiHorizonSequenceDataset(feature_path, target_bundle_path, target_keys=target_keys, sequence_len=sequence_len, row_slice=row_slice)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, **_loader_kwargs())
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device, non_blocking=True)
            with autocast_context(device, amp_enabled, amp_dtype):
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


def strategic_view_from_multihorizon(targets: np.ndarray, probabilities: np.ndarray, horizon_labels: Sequence[str]) -> dict[str, np.ndarray]:
    direction_targets, hold_targets, confidence_targets = split_multihorizon_heads_numpy(targets, len(horizon_labels))
    direction_probabilities, hold_probabilities, confidence_probabilities = split_multihorizon_heads_numpy(probabilities, len(horizon_labels))
    strategic_target = (direction_targets[:, -2:].mean(axis=1) >= 0.5).astype(np.float32)
    strategic_probability = direction_probabilities[:, -2:].mean(axis=1).astype(np.float32)
    strategic_hold_target = (hold_targets[:, -2:].mean(axis=1) >= 0.5).astype(np.float32)
    strategic_hold_probability = hold_probabilities[:, -2:].mean(axis=1).astype(np.float32)
    strategic_confidence_target = confidence_targets[:, -2:].mean(axis=1).astype(np.float32)
    strategic_confidence_probability = confidence_probabilities[:, -2:].mean(axis=1).astype(np.float32)
    return {
        "strategic_target": strategic_target,
        "strategic_probability": strategic_probability,
        "strategic_hold_target": strategic_hold_target,
        "strategic_hold_probability": strategic_hold_probability,
        "strategic_confidence_target": strategic_confidence_target,
        "strategic_confidence_probability": strategic_confidence_probability,
    }


def directional_backtest(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    decision_threshold: float = 0.5,
    confidence_floor: float = 0.12,
    gate_scores: np.ndarray | None = None,
    gate_threshold: float = 0.5,
    hold_probabilities: np.ndarray | None = None,
    hold_threshold: float = 0.5,
    confidence_probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    targets = np.asarray(targets, dtype=np.float32)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    confidence = confidence_from_probabilities(probabilities)
    long_mask = (probabilities >= decision_threshold) & (confidence >= confidence_floor)
    short_mask = (probabilities <= (1.0 - decision_threshold)) & (confidence >= confidence_floor)
    if hold_probabilities is not None:
        predicted_hold = np.asarray(hold_probabilities, dtype=np.float32) >= float(hold_threshold)
        long_mask = long_mask & (~predicted_hold)
        short_mask = short_mask & (~predicted_hold)
    if confidence_probabilities is not None:
        explicit_conf = np.asarray(confidence_probabilities, dtype=np.float32)
        long_mask = long_mask & (explicit_conf >= confidence_floor)
        short_mask = short_mask & (explicit_conf >= confidence_floor)
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
        "hold_threshold": float(hold_threshold),
        "capital_backtests": {
            "usd_10": capital_backtest_from_unit_pnl(pnl, initial_capital=10.0, risk_fraction=0.02),
            "usd_1000": capital_backtest_from_unit_pnl(pnl, initial_capital=1000.0, risk_fraction=0.02),
            "usd_10_fixed_risk": fixed_risk_capital_backtest_from_unit_pnl(pnl, initial_capital=10.0, risk_fraction=0.02),
            "usd_1000_fixed_risk": fixed_risk_capital_backtest_from_unit_pnl(pnl, initial_capital=1000.0, risk_fraction=0.02),
        },
    }


def capital_backtest_from_unit_pnl(
    pnl: np.ndarray,
    *,
    initial_capital: float,
    risk_fraction: float = 0.02,
) -> dict[str, Any]:
    pnl = np.asarray(pnl, dtype=np.float32)
    equity = float(initial_capital)
    peak = equity
    max_drawdown_pct = 0.0
    winning_trades = 0
    losing_trades = 0
    trade_count = 0
    overflowed = False
    log10_equity = float(np.log10(max(initial_capital, 1e-12)))
    for unit in pnl:
        if float(unit) == 0.0:
            peak = max(peak, equity)
            drawdown_pct = 0.0 if peak <= 0 else (peak - equity) / peak
            max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
            continue
        trade_count += 1
        growth_factor = max(1e-12, 1.0 + float(risk_fraction) * float(unit))
        log10_equity += float(np.log10(growth_factor))
        if not overflowed:
            equity *= growth_factor
            if not np.isfinite(equity) or equity > 1e308:
                overflowed = True
        if unit > 0:
            winning_trades += 1
        else:
            losing_trades += 1
        if not overflowed:
            peak = max(peak, equity)
            drawdown_pct = 0.0 if peak <= 0 else (peak - equity) / peak
            max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
    if overflowed:
        return {
            "initial_capital": round(float(initial_capital), 6),
            "mode": "compounding_r_multiple",
            "risk_fraction": float(risk_fraction),
            "final_capital": None,
            "net_profit": None,
            "return_pct": None,
            "max_drawdown_pct": round(float(max_drawdown_pct * 100.0), 6),
            "trade_count": int(trade_count),
            "winning_trades": int(winning_trades),
            "losing_trades": int(losing_trades),
            "overflowed": True,
            "log10_final_capital": round(float(log10_equity), 6),
        }
    return {
        "initial_capital": round(float(initial_capital), 6),
        "mode": "compounding_r_multiple",
        "risk_fraction": float(risk_fraction),
        "final_capital": round(float(equity), 6),
        "net_profit": round(float(equity - initial_capital), 6),
        "return_pct": round(float((equity / initial_capital - 1.0) * 100.0) if initial_capital > 0 else 0.0, 6),
        "max_drawdown_pct": round(float(max_drawdown_pct * 100.0), 6),
        "trade_count": int(trade_count),
        "winning_trades": int(winning_trades),
        "losing_trades": int(losing_trades),
        "overflowed": False,
        "log10_final_capital": round(float(np.log10(max(equity, 1e-12))), 6),
    }


def _loader_kwargs() -> dict[str, Any]:
    workers = max(0, int(NUM_WORKERS))
    kwargs: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": bool(PIN_MEMORY),
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(PERSISTENT_WORKERS)
        kwargs["prefetch_factor"] = max(2, int(PREFETCH_FACTOR))
    return kwargs


def fixed_risk_capital_backtest_from_unit_pnl(
    pnl: np.ndarray,
    *,
    initial_capital: float,
    risk_fraction: float = 0.02,
) -> dict[str, Any]:
    pnl = np.asarray(pnl, dtype=np.float32)
    stake = float(initial_capital) * float(risk_fraction)
    equity = float(initial_capital)
    equity_curve = []
    winning_trades = 0
    losing_trades = 0
    trade_count = 0
    for unit in pnl:
        if float(unit) == 0.0:
            equity_curve.append(equity)
            continue
        trade_count += 1
        equity = max(0.0, equity + stake * float(unit))
        if unit > 0:
            winning_trades += 1
        else:
            losing_trades += 1
        equity_curve.append(equity)
    equity_array = np.asarray(equity_curve, dtype=np.float32) if equity_curve else np.asarray([initial_capital], dtype=np.float32)
    final_capital = float(equity_array[-1]) if equity_array.size else float(initial_capital)
    peak = np.maximum.accumulate(equity_array) if equity_array.size else np.asarray([initial_capital], dtype=np.float32)
    drawdown = peak - equity_array if equity_array.size else np.asarray([0.0], dtype=np.float32)
    drawdown_pct = np.divide(drawdown, np.maximum(peak, 1e-6)) if drawdown.size else np.asarray([0.0], dtype=np.float32)
    return {
        "initial_capital": round(float(initial_capital), 6),
        "mode": "fixed_r_multiple",
        "risk_fraction": float(risk_fraction),
        "risk_amount": round(float(stake), 6),
        "final_capital": round(final_capital, 6),
        "net_profit": round(final_capital - float(initial_capital), 6),
        "return_pct": round(float((final_capital / initial_capital - 1.0) * 100.0) if initial_capital > 0 else 0.0, 6),
        "max_drawdown_pct": round(float(drawdown_pct.max() * 100.0), 6),
        "trade_count": int(trade_count),
        "winning_trades": int(winning_trades),
        "losing_trades": int(losing_trades),
    }


def optimize_backtest_thresholds(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    hold_probabilities: np.ndarray | None = None,
    confidence_probabilities: np.ndarray | None = None,
    gate_scores: np.ndarray | None = None,
    gate_threshold: float | None = None,
) -> dict[str, float]:
    best = {
        "decision_threshold": 0.56,
        "confidence_floor": 0.12,
        "hold_threshold": 0.62,
        "gate_threshold": float(gate_threshold) if gate_threshold is not None else 0.5,
        "score": 0.0,
    }
    target_participation = 0.14
    hold_candidates = np.linspace(0.55, 0.85, 7) if hold_probabilities is not None else [0.5]
    gate_candidates: list[float]
    if gate_scores is not None:
        gate_values = np.asarray(gate_scores, dtype=np.float32)
        quantile_points = [0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99]
        gate_candidates = sorted({float(np.quantile(gate_values, q)) for q in quantile_points})
        if gate_threshold is not None:
            gate_candidates.append(float(gate_threshold))
            gate_candidates = sorted(set(gate_candidates))
    else:
        gate_candidates = [0.5]
    best_backup = None
    best_backup_score = float("-inf")
    for threshold in np.linspace(0.53, 0.75, 12):
        for floor in np.linspace(0.06, 0.30, 13):
            for hold_threshold_value in hold_candidates:
                for gate_threshold_value in gate_candidates:
                    report = directional_backtest(
                        targets,
                        probabilities,
                        decision_threshold=float(threshold),
                        confidence_floor=float(floor),
                        gate_scores=gate_scores,
                        gate_threshold=float(gate_threshold_value),
                        hold_probabilities=hold_probabilities,
                        hold_threshold=float(hold_threshold_value),
                        confidence_probabilities=confidence_probabilities,
                    )
                    participation = float(report["participation_rate"])
                    participation_reward = max(0.0, 1.0 - abs(participation - target_participation) / target_participation)
                    score = (
                        float(report["win_rate"]) * 0.55
                        + float(report["avg_unit_pnl"]) * 0.25
                        + participation_reward * 0.15
                        + (1.0 - min(1.0, participation / 0.35)) * 0.05
                    )
                    if 0.01 <= participation <= 0.35 and score > best["score"]:
                        best = {
                            "decision_threshold": float(threshold),
                            "confidence_floor": float(floor),
                            "hold_threshold": float(hold_threshold_value),
                            "gate_threshold": float(gate_threshold_value),
                            "score": float(score),
                        }
                    if participation > 0.0 and score > best_backup_score:
                        best_backup = {
                            "decision_threshold": float(threshold),
                            "confidence_floor": float(floor),
                            "hold_threshold": float(hold_threshold_value),
                            "gate_threshold": float(gate_threshold_value),
                            "score": float(score),
                        }
                        best_backup_score = float(score)
    if best["score"] <= 0.0 and best_backup is not None:
        best = best_backup
    return best


def _serialize_meta_gate(meta_gate: dict[str, Any] | None) -> dict[str, Any] | None:
    if not meta_gate:
        return None
    return {key: value for key, value in meta_gate.items() if key != "model"}


def _combined_gate_scores(
    probabilities: np.ndarray,
    precision_gate: dict[str, Any] | None,
    meta_gate: dict[str, Any] | None,
    *,
    context_features: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    precision_scores = (
        apply_precision_gate(probabilities, precision_gate, context_features=context_features)
        if precision_gate is not None
        else None
    )
    meta_scores = apply_meta_gate(probabilities, meta_gate, context_features=context_features) if meta_gate is not None else None
    combined_scores = combine_gate_scores(precision_scores, meta_scores)
    return precision_scores, meta_scores, combined_scores


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
    horizon_keys = ([f"target_{label}" for label in horizon_labels] + [f"hold_{label}" for label in horizon_labels] + [f"confidence_{label}" for label in horizon_labels]) if multi_horizon else []
    threshold = float(manifest.get("classification_threshold", 0.5))
    precision_gate_path = resolve_precision_gate_path(manifest)
    precision_gate = json.loads(precision_gate_path.read_text(encoding="utf-8")) if precision_gate_path is not None and precision_gate_path.exists() else None
    meta_gate_path = resolve_meta_gate_path(manifest)
    meta_gate = load_meta_gate(meta_gate_path) if meta_gate_path is not None else None
    gate_context_path = resolve_gate_context_path(manifest)

    reports: list[FoldReport] = []
    all_targets: list[np.ndarray] = []
    all_probs_raw: list[np.ndarray] = []
    all_probs_calibrated: list[np.ndarray] = []
    all_probs_multi: list[np.ndarray] = []
    all_hold_probs: list[np.ndarray] = []
    all_conf_probs: list[np.ndarray] = []
    all_gate_scores: list[np.ndarray] = []
    validation_years = manifest.get("split_report", {}).get("val_years", []) or list(VAL_YEARS)
    validation_folds = yearly_slices(timestamps, sequence_len=sequence_len, years=validation_years)
    if not validation_folds:
        validation_folds = folds
        validation_years = [int(year) for year, _ in folds]
    calibration_targets: list[np.ndarray] = []
    calibration_probabilities: list[np.ndarray] = []
    calibration_hold_probabilities: list[np.ndarray] = []
    calibration_conf_probabilities: list[np.ndarray] = []
    calibration_gate_scores_collection: list[np.ndarray] = []
    for _, fold_slice in validation_folds:
        active_slice = fold_slice
        if max_calibration_windows > 0 and len(fold_slice) > max_calibration_windows:
            active_slice = DatasetSlice(fold_slice.start, fold_slice.start + max_calibration_windows)
        gate_context_val = slice_gate_context(gate_context_path, active_slice, sequence_len)
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
            strategic_val = strategic_view_from_multihorizon(targets_val, probabilities_val, horizon_labels)
            calibration_targets.append(strategic_val["strategic_target"])
            calibration_probabilities.append(strategic_val["strategic_probability"])
            calibration_hold_probabilities.append(strategic_val["strategic_hold_probability"])
            calibration_conf_probabilities.append(strategic_val["strategic_confidence_probability"])
            _, _, combined_gate = _combined_gate_scores(
                probabilities_val,
                precision_gate,
                meta_gate,
                context_features=gate_context_val,
            )
            if combined_gate is not None:
                calibration_gate_scores_collection.append(combined_gate)
            continue
        targets_val, probabilities_val = predict_for_slice(model, device, active_slice, feature_path=feature_path, target_path=target_path, sequence_len=sequence_len, batch_size=batch_size)
        calibration_targets.append(targets_val)
        calibration_probabilities.append(probabilities_val)
    calibration_targets_np = np.concatenate(calibration_targets) if calibration_targets else np.empty(0, dtype=np.float32)
    calibration_probabilities_np = np.concatenate(calibration_probabilities) if calibration_probabilities else np.empty(0, dtype=np.float32)
    global_calibration = build_calibration_report(calibration_targets_np, calibration_probabilities_np)
    calibration_gate_scores = np.concatenate(calibration_gate_scores_collection) if calibration_gate_scores_collection else None
    optimized_thresholds = optimize_backtest_thresholds(
        calibration_targets_np,
        apply_bucket_calibration(calibration_probabilities_np, global_calibration),
        hold_probabilities=np.concatenate(calibration_hold_probabilities) if calibration_hold_probabilities else None,
        confidence_probabilities=np.concatenate(calibration_conf_probabilities) if calibration_conf_probabilities else None,
        gate_scores=calibration_gate_scores,
        gate_threshold=float(
            (meta_gate or {}).get(
                "threshold",
                (precision_gate or {}).get("threshold", 0.5),
            )
        )
        if multi_horizon
        else None,
    )
    decision_threshold = float(optimized_thresholds["decision_threshold"])
    confidence_floor = float(optimized_thresholds["confidence_floor"])
    optimized_hold_threshold = float(optimized_thresholds.get("hold_threshold", 0.5))
    optimized_gate_threshold = float(optimized_thresholds.get("gate_threshold", 0.5))

    for year, fold_slice in folds:
        active_slice = fold_slice
        if max_windows_per_year > 0 and len(fold_slice) > max_windows_per_year:
            active_slice = DatasetSlice(fold_slice.start, fold_slice.start + max_windows_per_year)
        gate_context_year = slice_gate_context(gate_context_path, active_slice, sequence_len)
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
            strategic_view = strategic_view_from_multihorizon(targets_year, probabilities_year, horizon_labels)
            primary_targets_year = strategic_view["strategic_target"]
            primary_probabilities_year = strategic_view["strategic_probability"]
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
            raw_metrics = collect_multihorizon_metrics(targets_year, probabilities_year, thresholds=[0.5, 0.5, threshold, threshold], horizon_labels=horizon_labels)
            calibrated_metrics = {
                "primary": collect_binary_metrics(primary_targets_year, calibrated_probabilities, threshold=threshold),
                "horizons": raw_metrics["horizons"],
                "strategic": raw_metrics.get("strategic", {}),
                "hold_horizons": raw_metrics.get("hold_horizons", {}),
                "confidence_horizons": raw_metrics.get("confidence_horizons", {}),
            }
        else:
            raw_metrics = collect_binary_metrics(primary_targets_year, primary_probabilities_year, threshold=threshold)
            calibrated_metrics = collect_binary_metrics(primary_targets_year, calibrated_probabilities, threshold=threshold)
        fold_calibration = build_calibration_report(primary_targets_year, calibrated_probabilities)
        gate_scores = None
        gate_threshold = 0.5
        if multi_horizon:
            _, _, gate_scores = _combined_gate_scores(
                probabilities_year,
                precision_gate,
                meta_gate,
                context_features=gate_context_year,
            )
            gate_threshold = optimized_gate_threshold if gate_scores is not None else 0.5
            strategic_hold_probability = strategic_view["strategic_hold_probability"]
            strategic_confidence_probability = strategic_view["strategic_confidence_probability"]
        else:
            strategic_hold_probability = None
            strategic_confidence_probability = None
        backtest = directional_backtest(
            primary_targets_year,
            calibrated_probabilities,
            decision_threshold=decision_threshold,
            confidence_floor=confidence_floor,
            gate_scores=gate_scores,
            gate_threshold=gate_threshold,
            hold_probabilities=strategic_hold_probability,
            hold_threshold=optimized_hold_threshold,
            confidence_probabilities=strategic_confidence_probability,
        )
        if gate_scores is not None:
            backtest["gate_positive_rate"] = round(float((gate_scores >= optimized_gate_threshold).mean()), 6)
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
            all_hold_probs.append(strategic_view["strategic_hold_probability"])
            all_conf_probs.append(strategic_view["strategic_confidence_probability"])
        if gate_scores is not None:
            all_gate_scores.append(gate_scores)

    all_targets_np = np.concatenate(all_targets) if all_targets else np.empty(0, dtype=np.float32)
    all_probs_raw_np = np.concatenate(all_probs_raw) if all_probs_raw else np.empty(0, dtype=np.float32)
    all_probs_calibrated_np = np.concatenate(all_probs_calibrated) if all_probs_calibrated else np.empty(0, dtype=np.float32)
    overall_gate_scores = np.concatenate(all_gate_scores) if all_gate_scores else None
    overall = {
        "raw_metrics": collect_binary_metrics(all_targets_np, all_probs_raw_np, threshold=threshold),
        "calibrated_metrics": collect_binary_metrics(all_targets_np, all_probs_calibrated_np, threshold=threshold),
        "calibration": build_calibration_report(all_targets_np, all_probs_calibrated_np),
        "backtest": directional_backtest(
            all_targets_np,
            all_probs_calibrated_np,
            decision_threshold=decision_threshold,
            confidence_floor=confidence_floor,
            gate_scores=overall_gate_scores,
            gate_threshold=optimized_gate_threshold if overall_gate_scores is not None else 0.5,
            hold_probabilities=np.concatenate(all_hold_probs) if multi_horizon and all_hold_probs else None,
            hold_threshold=optimized_hold_threshold,
            confidence_probabilities=np.concatenate(all_conf_probs) if multi_horizon and all_conf_probs else None,
        ),
    }
    if precision_gate is not None:
        overall["precision_gate"] = precision_gate
    if meta_gate is not None:
        overall["meta_gate"] = _serialize_meta_gate(meta_gate)
    if overall_gate_scores is not None:
        overall["gate_positive_rate"] = round(float((overall_gate_scores >= optimized_gate_threshold).mean()), 6)
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
