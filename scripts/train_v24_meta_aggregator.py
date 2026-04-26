from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import OUTPUTS_V24_DIR, V24_META_AGGREGATOR_CONFIG, V24_META_AGGREGATOR_PATH, V24_META_AGGREGATOR_REPORT
from src.v22.month_debugger import V22DebugConfig, run_v22_month_suite
from src.v24.models import MetaAggregatorModel, MetaAggregatorModelConfig
from src.v24.training import build_trade_quality_dataset


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=np.float64).reshape(-1)
    right = np.asarray(b, dtype=np.float64).reshape(-1)
    if left.size == 0 or right.size == 0:
        return 0.0
    if np.allclose(left.std(ddof=0), 0.0) or np.allclose(right.std(ddof=0), 0.0):
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _normalize_features(train_values: np.ndarray, full_values: np.ndarray, *, axis: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_values.mean(axis=axis, keepdims=True).astype(np.float32)
    std = train_values.std(axis=axis, ddof=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    normalized = (full_values - mean) / std
    return normalized.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def _tensor_dataset(
    static_features: np.ndarray,
    sequence_features: np.ndarray,
    targets: Mapping[str, np.ndarray],
    indices: np.ndarray,
) -> TensorDataset:
    ordered_indices = np.asarray(indices, dtype=np.int64)
    payload = [
        torch.from_numpy(sequence_features[ordered_indices]),
        torch.from_numpy(static_features[ordered_indices]),
        torch.from_numpy(targets["expected_value"][ordered_indices]),
        torch.from_numpy(targets["win_probability"][ordered_indices]),
        torch.from_numpy(targets["realized_rr"][ordered_indices]),
        torch.from_numpy(targets["expected_drawdown"][ordered_indices]),
        torch.from_numpy(targets["danger_score"][ordered_indices]),
        torch.from_numpy(targets["uncertainty"][ordered_indices]),
        torch.from_numpy(targets["abstain_probability"][ordered_indices]),
    ]
    return TensorDataset(*payload)


def _compute_loss(outputs: Mapping[str, torch.Tensor], batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
    _, _, target_expected_value, target_win, target_rr, target_drawdown, target_danger, target_uncertainty, target_abstain = batch
    loss_expected_value = nn.functional.smooth_l1_loss(outputs["expected_value"], target_expected_value)
    loss_win = nn.functional.binary_cross_entropy(outputs["win_probability"], target_win)
    loss_rr = nn.functional.smooth_l1_loss(outputs["realized_rr"], target_rr)
    loss_drawdown = nn.functional.smooth_l1_loss(outputs["expected_drawdown"], target_drawdown)
    loss_danger = nn.functional.smooth_l1_loss(outputs["danger_score"], target_danger)
    loss_uncertainty = nn.functional.smooth_l1_loss(outputs["uncertainty"], target_uncertainty)
    loss_abstain = nn.functional.smooth_l1_loss(outputs["abstain_probability"], target_abstain)
    return (
        (1.00 * loss_expected_value)
        + (0.60 * loss_win)
        + (0.30 * loss_rr)
        + (0.20 * loss_drawdown)
        + (0.35 * loss_danger)
        + (0.25 * loss_uncertainty)
        + (0.25 * loss_abstain)
    )


def _evaluate_model(
    model: MetaAggregatorModel,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    expected_true: list[float] = []
    expected_pred: list[float] = []
    win_true: list[float] = []
    win_pred: list[float] = []
    rr_true: list[float] = []
    rr_pred: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            sequence, static, expected_value, win, rr, drawdown, danger, uncertainty, abstain = [item.to(device, dtype=torch.float32) for item in batch]
            outputs = model(sequence, static)
            loss = _compute_loss(outputs, (sequence, static, expected_value, win, rr, drawdown, danger, uncertainty, abstain))
            losses.append(float(loss.item()))
            expected_true.extend(expected_value.detach().cpu().tolist())
            expected_pred.extend(outputs["expected_value"].detach().cpu().tolist())
            win_true.extend(win.detach().cpu().tolist())
            win_pred.extend(outputs["win_probability"].detach().cpu().tolist())
            rr_true.extend(rr.detach().cpu().tolist())
            rr_pred.extend(outputs["realized_rr"].detach().cpu().tolist())
    expected_true_np = np.asarray(expected_true, dtype=np.float32)
    expected_pred_np = np.asarray(expected_pred, dtype=np.float32)
    win_true_np = np.asarray(win_true, dtype=np.float32)
    win_pred_np = np.asarray(win_pred, dtype=np.float32)
    rr_true_np = np.asarray(rr_true, dtype=np.float32)
    rr_pred_np = np.asarray(rr_pred, dtype=np.float32)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "expected_value_correlation": _corrcoef(expected_true_np, expected_pred_np),
        "expected_value_mae": float(np.mean(np.abs(expected_true_np - expected_pred_np))) if expected_true_np.size else 0.0,
        "win_probability_brier": float(np.mean((win_true_np - win_pred_np) ** 2)) if win_true_np.size else 0.0,
        "rr_mae": float(np.mean(np.abs(rr_true_np - rr_pred_np))) if rr_true_np.size else 0.0,
        "predicted_win_rate": float((win_pred_np >= 0.5).mean()) if win_pred_np.size else 0.0,
    }


def _build_split_indices(months: Sequence[str], timestamps: Sequence[str], train_months: set[str], eval_months: set[str]) -> dict[str, np.ndarray]:
    month_array = np.asarray([str(item) for item in months], dtype=object)
    timestamp_array = pd.to_datetime(list(timestamps), utc=True, errors="coerce").view("int64")
    train_candidates = np.where(np.isin(month_array, list(train_months)))[0]
    eval_indices = np.where(np.isin(month_array, list(eval_months)))[0]
    if train_candidates.size == 0:
        raise ValueError(f"No train samples found for months: {sorted(train_months)}")
    if eval_indices.size == 0:
        raise ValueError(f"No eval samples found for months: {sorted(eval_months)}")
    ordered_train = train_candidates[np.argsort(timestamp_array[train_candidates])]
    split = max(1, int(math.floor(len(ordered_train) * 0.80)))
    split = min(split, len(ordered_train) - 1) if len(ordered_train) > 1 else len(ordered_train)
    train_indices = ordered_train[:split]
    valid_indices = ordered_train[split:] if split < len(ordered_train) else ordered_train[-max(1, min(64, len(ordered_train))):]
    return {
        "train": train_indices.astype(np.int64),
        "valid": valid_indices.astype(np.int64),
        "eval": eval_indices.astype(np.int64),
    }


def _bridge_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(report.get("results", []))
    if not rows:
        return {"months": [], "weighted_win_rate": 0.0, "weighted_trade_count": 0, "trade_frequency_ok": False}
    total_trades = max(1, sum(int(item.get("trade_count", 0)) for item in rows))
    weighted_win_rate = sum(float(item.get("win_rate", 0.0)) * int(item.get("trade_count", 0)) for item in rows) / total_trades
    weighted_cumulative_return = sum(float(item.get("cumulative_return", 0.0)) for item in rows)
    compact_months = [{key: value for key, value in item.items() if key != "trades"} for item in rows]
    return {
        "months": compact_months,
        "weighted_trade_count": int(total_trades),
        "weighted_win_rate": float(weighted_win_rate),
        "weighted_cumulative_return": float(weighted_cumulative_return),
        "trade_frequency_ok": bool(all(bool(item.get("target_trade_band_met", False)) for item in rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the Phase-2 V24 learned meta-aggregator.")
    parser.add_argument("--reports", default="")
    parser.add_argument("--train-months", default="2023-12")
    parser.add_argument("--eval-months", default="2024-12")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--heuristic-prior-weight", type=float, default=0.25)
    parser.add_argument("--checkpoint", default=str(V24_META_AGGREGATOR_PATH))
    parser.add_argument("--config", default=str(V24_META_AGGREGATOR_CONFIG))
    parser.add_argument("--report", default=str(V24_META_AGGREGATOR_REPORT))
    args = parser.parse_args()

    _set_seed(int(args.seed))
    report_paths = [Path(item.strip()) for item in str(args.reports).split(",") if item.strip()]
    dataset = build_trade_quality_dataset(report_paths or None)
    train_months = {item.strip() for item in str(args.train_months).split(",") if item.strip()}
    eval_months = {item.strip() for item in str(args.eval_months).split(",") if item.strip()}
    splits = _build_split_indices(dataset.months, dataset.timestamps, train_months, eval_months)

    static_norm, static_mean, static_std = _normalize_features(
        dataset.static_features[splits["train"]],
        dataset.static_features,
        axis=(0,),
    )
    sequence_norm, sequence_mean, sequence_std = _normalize_features(
        dataset.sequence_features[splits["train"]],
        dataset.sequence_features,
        axis=(0, 1),
    )

    train_loader = DataLoader(_tensor_dataset(static_norm, sequence_norm, dataset.targets, splits["train"]), batch_size=int(args.batch_size), shuffle=True)
    valid_loader = DataLoader(_tensor_dataset(static_norm, sequence_norm, dataset.targets, splits["valid"]), batch_size=int(args.batch_size), shuffle=False)
    eval_loader = DataLoader(_tensor_dataset(static_norm, sequence_norm, dataset.targets, splits["eval"]), batch_size=int(args.batch_size), shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = MetaAggregatorModelConfig(
        static_dim=int(static_norm.shape[1]),
        sequence_dim=int(sequence_norm.shape[2]),
        seq_len=int(sequence_norm.shape[1]),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
    )
    model = MetaAggregatorModel(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)

    best_state: dict[str, Any] | None = None
    best_valid_loss = float("inf")
    best_epoch = -1
    patience = 10
    stale_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            sequence, static, expected_value, win, rr, drawdown, danger, uncertainty, abstain = [item.to(device, dtype=torch.float32) for item in batch]
            optimizer.zero_grad(set_to_none=True)
            outputs = model(sequence, static)
            loss = _compute_loss(outputs, (sequence, static, expected_value, win, rr, drawdown, danger, uncertainty, abstain))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))
        valid_metrics = _evaluate_model(model, valid_loader, device=device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
                **valid_metrics,
            }
        )
        if valid_metrics["loss"] < best_valid_loss:
            best_valid_loss = float(valid_metrics["loss"])
            best_epoch = int(epoch)
            best_state = {
                "model_state": model.state_dict(),
                "model_config": model_config.to_dict(),
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint state.")

    model.load_state_dict(best_state["model_state"])
    valid_metrics = _evaluate_model(model, valid_loader, device=device)
    eval_metrics = _evaluate_model(model, eval_loader, device=device)

    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config)
    report_path = Path(args.report)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    OUTPUTS_V24_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "model_state": best_state["model_state"],
        "model_config": best_state["model_config"],
        "static_feature_names": list(dataset.static_feature_names),
        "sequence_feature_names": list(dataset.sequence_feature_names),
        "static_mean": static_mean.reshape(-1).tolist(),
        "static_std": static_std.reshape(-1).tolist(),
        "sequence_mean": sequence_mean.reshape(-1).tolist(),
        "sequence_std": sequence_std.reshape(-1).tolist(),
        "best_epoch": best_epoch,
        "valid_metrics": valid_metrics,
        "eval_metrics": eval_metrics,
    }
    torch.save(checkpoint_payload, checkpoint_path)

    runtime_config = {
        "enabled": True,
        "heuristic_prior_weight": float(args.heuristic_prior_weight),
        "model_config": best_state["model_config"],
        "train_months": sorted(train_months),
        "eval_months": sorted(eval_months),
    }
    config_path.write_text(json.dumps(runtime_config, indent=2), encoding="utf-8")

    heuristic_report = run_v22_month_suite(
        sorted(eval_months),
        configs=[
            V22DebugConfig(
                name="v24_trade_quality_bridge",
                mode="v24_bridge",
                meta_aggregator_preference="heuristic",
                cooldown_bars=5,
                ensemble_risk_threshold=0.75,
                ensemble_disagreement_threshold=1.10,
            )
        ],
    )
    learned_report = run_v22_month_suite(
        sorted(eval_months),
        configs=[
            V22DebugConfig(
                name="v24_trade_quality_bridge",
                mode="v24_bridge",
                meta_aggregator_preference="learned",
                cooldown_bars=5,
                ensemble_risk_threshold=0.75,
                ensemble_disagreement_threshold=1.10,
            )
        ],
    )

    training_report = {
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path),
        "train_months": sorted(train_months),
        "eval_months": sorted(eval_months),
        "dataset": {
            "row_count": int(len(dataset.months)),
            "train_rows": int(len(splits["train"])),
            "valid_rows": int(len(splits["valid"])),
            "eval_rows": int(len(splits["eval"])),
            "static_dim": int(dataset.static_features.shape[1]),
            "sequence_shape": [int(dataset.sequence_features.shape[1]), int(dataset.sequence_features.shape[2])],
            "warnings": list(dataset.warnings),
        },
        "best_epoch": int(best_epoch),
        "history_tail": history[-10:],
        "validation_metrics": valid_metrics,
        "eval_metrics": eval_metrics,
        "heuristic_bridge": _bridge_summary(heuristic_report),
        "learned_bridge": _bridge_summary(learned_report),
    }
    report_path.write_text(json.dumps(training_report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(training_report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
