from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import V20_FEATURES_PATH, V20_MAMBA_MODEL_PATH, V20_MAMBA_TRAINING_LOG_PATH
from src.v20.mamba_backbone import NexusBiMamba


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the V20 BiMamba fallback backbone on the local feature frame.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence-len", type=int, default=32)
    args = parser.parse_args()

    frame = pd.read_parquet(V20_FEATURES_PATH).select_dtypes(include=["number"]).dropna().tail(8000)
    feature_columns = [col for col in frame.columns if not str(col).startswith("target_") and not str(col).startswith("future_return_") and col != "range_forward_15m"]
    if len(frame) <= args.sequence_len + 1:
        raise SystemExit("Not enough V20 feature rows available for backbone training.")
    X_rows, y_rows = [], []
    values = frame[feature_columns].to_numpy(dtype="float32")
    targets = frame["target_up_15m"].to_numpy(dtype="float32")
    for index in range(args.sequence_len, len(frame)):
        X_rows.append(values[index - args.sequence_len : index])
        y_rows.append(targets[index])
    dataset = TensorDataset(torch.tensor(X_rows), torch.tensor(y_rows))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NexusBiMamba(n_features=len(feature_columns), sequence_len=args.sequence_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    V20_MAMBA_TRAINING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logs: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        running = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(batch_x)["dir_15m"]
                loss = loss_fn(logits, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach().cpu())
        logs.append({"epoch": float(epoch + 1), "loss": running / max(len(loader), 1)})
    torch.save({"model_state": model.state_dict(), "feature_columns": feature_columns, "sequence_len": args.sequence_len}, V20_MAMBA_MODEL_PATH)
    V20_MAMBA_TRAINING_LOG_PATH.write_text("\n".join(json.dumps(item) for item in logs) + "\n", encoding="utf-8")
    print(f"saved={V20_MAMBA_MODEL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
