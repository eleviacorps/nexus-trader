from __future__ import annotations

import argparse
import json
import math
import os
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

from config.project_config import OUTPUTS_V21_DIR, V21_BIMAMBA_MODEL_PATH, V21_FEATURES_PATH
from src.v20.mamba_backbone import NexusBiMamba


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
    if len(numeric) <= sequence_len:
        raise SystemExit("Not enough V21 feature rows available for BiMamba training.")

    values = numeric[feature_columns].to_numpy(dtype=np.float32)
    targets = pd.to_numeric(numeric.get("target_up_15m"), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    for index in range(sequence_len, len(numeric)):
        x_rows.append(values[index - sequence_len : index])
        y_rows.append(float(targets[index]))

    dataset = TensorDataset(
        torch.tensor(np.asarray(x_rows), dtype=torch.float32),
        torch.tensor(np.asarray(y_rows), dtype=torch.float32),
    )
    return feature_columns, dataset


def _loader(dataset: torch.utils.data.Dataset, *, batch_size: int, shuffle: bool) -> DataLoader:
    num_workers = min(8, max(1, ((os.cpu_count() or 2) // 2)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the V21 BiMamba ablation on the V21 feature frame.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sequence-len", type=int, default=240)
    parser.add_argument("--max-rows", type=int, default=50000)
    args = parser.parse_args()

    if not V21_FEATURES_PATH.exists():
        raise SystemExit(f"Missing V21 feature frame at {V21_FEATURES_PATH}.")

    frame = pd.read_parquet(V21_FEATURES_PATH)
    feature_columns, dataset = build_sequences(frame, sequence_len=int(args.sequence_len), max_rows=int(args.max_rows))
    split_index = max(int(len(dataset) * 0.9), 1)
    train_dataset = torch.utils.data.Subset(dataset, range(0, split_index))
    valid_dataset = torch.utils.data.Subset(dataset, range(split_index, len(dataset)))
    if len(valid_dataset) == 0:
        valid_dataset = torch.utils.data.Subset(dataset, range(max(len(dataset) - min(256, len(dataset)), 0), len(dataset)))

    train_loader = _loader(train_dataset, batch_size=int(args.batch_size), shuffle=True)
    valid_loader = _loader(valid_dataset, batch_size=int(args.batch_size), shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NexusBiMamba(n_features=len(feature_columns), sequence_len=int(args.sequence_len)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    best_metric = math.inf
    logs: list[dict[str, float]] = []

    for epoch in range(int(args.epochs)):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(batch_x)["dir_15m"]
                loss = loss_fn(logits, batch_y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.detach().cpu())

        model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    logits = model(batch_x)["dir_15m"]
                    loss = loss_fn(logits, batch_y)
                valid_loss += float(loss.detach().cpu())
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += int((preds == batch_y).sum().item())
                total += int(batch_y.numel())

        mean_valid = valid_loss / max(len(valid_loader), 1)
        accuracy = float(correct / total) if total else 0.0
        epoch_log = {
            "epoch": float(epoch + 1),
            "train_loss": train_loss / max(len(train_loader), 1),
            "valid_loss": mean_valid,
            "direction_accuracy": accuracy,
        }
        logs.append(epoch_log)
        if mean_valid <= best_metric:
            best_metric = mean_valid
            V21_BIMAMBA_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_columns": feature_columns,
                    "sequence_len": int(args.sequence_len),
                    "logs": logs,
                },
                V21_BIMAMBA_MODEL_PATH,
            )

    report = {
        "feature_count": int(len(feature_columns)),
        "dataset_sequences": int(len(dataset)),
        "train_sequences": int(len(train_dataset)),
        "valid_sequences": int(len(valid_dataset)),
        "best_valid_loss": round(float(best_metric), 6),
        "latest_direction_accuracy": round(float(logs[-1]["direction_accuracy"]) if logs else 0.0, 6),
        "logs": logs,
        "note": "BiMamba is the V21 ablation against the primary VSN+xLSTM backbone.",
    }
    report_path = OUTPUTS_V21_DIR / "bimamba_training_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(V21_BIMAMBA_MODEL_PATH), **report}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
