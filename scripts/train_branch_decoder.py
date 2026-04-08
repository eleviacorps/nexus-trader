from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import V20_BRANCH_DECODER_PATH, V20_FEATURES_PATH
from src.v20.branch_decoder import BranchDecoder


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the V20 branch decoder on local V20 features.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    frame = pd.read_parquet(V20_FEATURES_PATH).select_dtypes(include=["number"]).dropna().tail(5000)
    feature_columns = [col for col in frame.columns if not str(col).startswith("future_return_")][:128]
    if len(frame) < 64:
        raise SystemExit("Not enough V20 rows to train the branch decoder.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BranchDecoder(d_mamba=len(feature_columns), horizon=3, n_branches=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    features = torch.tensor(frame[feature_columns].to_numpy(dtype="float32"))
    future = torch.tensor(frame[["open", "high", "low", "close", "volume"]].shift(-1).ffill().to_numpy(dtype="float32")).view(len(frame), 1, 5)
    dataset = TensorDataset(features, future)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    for _ in range(args.epochs):
        for batch_features, batch_future in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_future = batch_future.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model.generate_branches(batch_features, n_branches=1)
                loss = loss_fn(outputs[:, 0, :1, :], batch_future)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    torch.save({"model_state": model.state_dict(), "feature_columns": feature_columns}, V20_BRANCH_DECODER_PATH)
    print(f"saved={V20_BRANCH_DECODER_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
