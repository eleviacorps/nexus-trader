from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import OUTPUTS_V21_DIR, V21_FEATURES_PATH, V21_XLSTM_MODEL_PATH
from src.v21.xlstm_backbone import NexusXLSTM


def _vol_bucket(values: pd.Series) -> pd.Series:
    quantiles = values.quantile([0.33, 0.66]).to_list() if len(values) else [0.0, 0.0]
    low, high = float(quantiles[0]), float(quantiles[1])
    return pd.Series(np.where(values <= low, 0, np.where(values <= high, 1, 2)), index=values.index, dtype=np.int64)


def _range_bucket(values: pd.Series) -> pd.Series:
    quantiles = values.quantile([0.33, 0.66]).to_list() if len(values) else [0.0, 0.0]
    low, high = float(quantiles[0]), float(quantiles[1])
    return pd.Series(np.where(values <= low, 0, np.where(values <= high, 1, 2)), index=values.index, dtype=np.int64)


def build_sequences(frame: pd.DataFrame, *, sequence_len: int, max_rows: int) -> tuple[list[str], TensorDataset]:
    numeric = frame.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    feature_columns = [
        column
        for column in numeric.columns
        if column
        not in {
            "target_up_15m",
            "target_up_30m",
            "future_return_15m",
            "future_return_30m",
            "range_forward_15m",
        }
    ]
    if max_rows > 0:
        numeric = numeric.tail(max_rows).copy()
    numeric = numeric.dropna().reset_index(drop=True)
    regime_ids = pd.to_numeric(numeric.get("hmm_state"), errors="coerce").fillna(0).clip(lower=0, upper=5).astype(np.int64)
    dir_15 = pd.to_numeric(numeric.get("target_up_15m"), errors="coerce").fillna(0.0).astype(np.float32)
    dir_30 = pd.to_numeric(numeric.get("target_up_30m"), errors="coerce").fillna(0.0).astype(np.float32)
    vol_proxy = _vol_bucket(pd.to_numeric(numeric.get("future_return_30m"), errors="coerce").fillna(0.0).abs())
    range_bucket = _range_bucket(pd.to_numeric(numeric.get("range_forward_15m"), errors="coerce").fillna(0.0))

    values = numeric[feature_columns].to_numpy(dtype=np.float32)
    regime_values = regime_ids.to_numpy(dtype=np.int64)
    dir15_values = dir_15.to_numpy(dtype=np.float32)
    dir30_values = dir_30.to_numpy(dtype=np.float32)
    vol_values = vol_proxy.to_numpy(dtype=np.int64)
    range_values = range_bucket.to_numpy(dtype=np.int64)

    x_rows: list[np.ndarray] = []
    regime_rows: list[np.ndarray] = []
    y_dir15: list[float] = []
    y_dir30: list[float] = []
    y_vol: list[int] = []
    y_regime: list[int] = []
    y_range: list[int] = []
    for index in range(sequence_len, len(numeric)):
        x_rows.append(values[index - sequence_len : index])
        regime_rows.append(regime_values[index - sequence_len : index])
        y_dir15.append(float(dir15_values[index]))
        y_dir30.append(float(dir30_values[index]))
        y_vol.append(int(vol_values[index]))
        y_regime.append(int(regime_values[index]))
        y_range.append(int(range_values[index]))

    dataset = TensorDataset(
        torch.tensor(np.asarray(x_rows), dtype=torch.float32),
        torch.tensor(np.asarray(regime_rows), dtype=torch.long),
        torch.tensor(np.asarray(y_dir15), dtype=torch.float32),
        torch.tensor(np.asarray(y_dir30), dtype=torch.float32),
        torch.tensor(np.asarray(y_vol), dtype=torch.long),
        torch.tensor(np.asarray(y_regime), dtype=torch.long),
        torch.tensor(np.asarray(y_range), dtype=torch.long),
    )
    return feature_columns, dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the V21 VSN+xLSTM backbone.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sequence-len", type=int, default=240)
    parser.add_argument("--max-rows", type=int, default=50000)
    args = parser.parse_args()

    if not V21_FEATURES_PATH.exists():
        raise SystemExit(f"Missing V21 feature frame at {V21_FEATURES_PATH}.")

    frame = pd.read_parquet(V21_FEATURES_PATH)
    feature_columns, dataset = build_sequences(frame, sequence_len=int(args.sequence_len), max_rows=int(args.max_rows))
    if len(dataset) < 32:
        raise SystemExit("Not enough sequence rows to train the V21 xLSTM backbone.")

    split_index = max(int(len(dataset) * 0.9), 1)
    train_dataset = torch.utils.data.Subset(dataset, range(0, split_index))
    valid_dataset = torch.utils.data.Subset(dataset, range(split_index, len(dataset)))
    if len(valid_dataset) == 0:
        valid_dataset = torch.utils.data.Subset(dataset, range(max(len(dataset) - min(256, len(dataset)), 0), len(dataset)))

    train_loader = DataLoader(train_dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    valid_loader = DataLoader(valid_dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NexusXLSTM(n_features=len(feature_columns), d_model=512, n_layers=4, n_regimes=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    best_metric = math.inf
    logs: list[dict[str, float]] = []

    for epoch in range(int(args.epochs)):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            x, regime_ids, y15, y30, y_vol, y_regime, y_range = [tensor.to(device, non_blocking=True) for tensor in batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(x, regime_ids)
                loss = (
                    1.0 * bce(outputs["dir_15m"], y15)
                    + 1.0 * bce(outputs["dir_30m"], y30)
                    + 0.5 * ce(outputs["vol_env"], y_vol)
                    + 0.5 * ce(outputs["regime"], y_regime)
                    + 0.3 * ce(outputs["range"], y_range)
                )
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu())

        model.eval()
        valid_loss = 0.0
        regime_correct = 0
        regime_total = 0
        with torch.no_grad():
            for batch in valid_loader:
                x, regime_ids, y15, y30, y_vol, y_regime, y_range = [tensor.to(device, non_blocking=True) for tensor in batch]
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    outputs = model(x, regime_ids)
                    loss = (
                        1.0 * bce(outputs["dir_15m"], y15)
                        + 1.0 * bce(outputs["dir_30m"], y30)
                        + 0.5 * ce(outputs["vol_env"], y_vol)
                        + 0.5 * ce(outputs["regime"], y_regime)
                        + 0.3 * ce(outputs["range"], y_range)
                    )
                valid_loss += float(loss.detach().cpu())
                regime_pred = outputs["regime"].argmax(dim=-1)
                regime_correct += int((regime_pred == y_regime).sum().item())
                regime_total += int(y_regime.numel())

        mean_valid = valid_loss / max(len(valid_loader), 1)
        regime_accuracy = float(regime_correct / regime_total) if regime_total else 0.0
        epoch_log = {
            "epoch": float(epoch + 1),
            "train_loss": running_loss / max(len(train_loader), 1),
            "valid_loss": mean_valid,
            "regime_accuracy": regime_accuracy,
        }
        logs.append(epoch_log)
        if mean_valid <= best_metric:
            best_metric = mean_valid
            V21_XLSTM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_columns": feature_columns,
                    "sequence_len": int(args.sequence_len),
                    "logs": logs,
                },
                V21_XLSTM_MODEL_PATH,
            )

    report = {
        "feature_count": int(len(feature_columns)),
        "dataset_sequences": int(len(dataset)),
        "train_sequences": int(len(train_dataset)),
        "valid_sequences": int(len(valid_dataset)),
        "best_valid_loss": round(float(best_metric), 6),
        "latest_regime_accuracy": round(float(logs[-1]["regime_accuracy"]) if logs else 0.0, 6),
        "logs": logs,
        "note": "This script implements the V21 VSN+xLSTM training path and supports remote full-archive training once Phase 1 finishes.",
    }
    report_path = OUTPUTS_V21_DIR / "xlstm_training_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(V21_XLSTM_MODEL_PATH), **report}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
