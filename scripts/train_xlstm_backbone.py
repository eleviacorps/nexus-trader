from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import OUTPUTS_V21_DIR, V21_FEATURES_PATH, V21_XLSTM_MODEL_PATH
from src.v21.xlstm_backbone import NexusXLSTM
from src.v21.training_data import build_v21_sequence_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the V21 VSN+xLSTM backbone.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sequence-len", type=int, default=240)
    parser.add_argument("--max-rows", type=int, default=120000)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()

    if not V21_FEATURES_PATH.exists():
        raise SystemExit(f"Missing V21 feature frame at {V21_FEATURES_PATH}.")

    frame = pd.read_parquet(V21_FEATURES_PATH)
    sequence_bundle = build_v21_sequence_bundle(frame, sequence_len=int(args.sequence_len), max_rows=int(args.max_rows))
    feature_columns = sequence_bundle.feature_columns
    dataset = sequence_bundle.dataset
    if len(dataset) < 32:
        raise SystemExit("Not enough sequence rows to train the V21 xLSTM backbone.")

    split_index = max(int(len(dataset) * 0.9), 1)
    train_dataset = torch.utils.data.Subset(dataset, range(0, split_index))
    valid_dataset = torch.utils.data.Subset(dataset, range(split_index, len(dataset)))
    if len(valid_dataset) == 0:
        valid_dataset = torch.utils.data.Subset(dataset, range(max(len(dataset) - min(256, len(dataset)), 0), len(dataset)))

    num_workers = min(4, max(1, (os.cpu_count() or 4) // 2))
    train_loader = DataLoader(train_dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    valid_loader = DataLoader(valid_dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NexusXLSTM(n_features=len(feature_columns), d_model=int(args.d_model), n_layers=int(args.n_layers), n_regimes=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate))
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    best_metric = math.inf
    logs: list[dict[str, float]] = []

    for epoch in range(int(args.epochs)):
        model.train()
        running_loss = 0.0
        train_steps = 0
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
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu())
            train_steps += 1

        model.eval()
        valid_loss = 0.0
        valid_steps = 0
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
                if not torch.isfinite(loss):
                    continue
                valid_loss += float(loss.detach().cpu())
                valid_steps += 1
                regime_pred = outputs["regime"].argmax(dim=-1)
                regime_correct += int((regime_pred == y_regime).sum().item())
                regime_total += int(y_regime.numel())

        mean_valid = valid_loss / max(valid_steps, 1)
        regime_accuracy = float(regime_correct / regime_total) if regime_total else 0.0
        epoch_log = {
            "epoch": float(epoch + 1),
            "train_loss": running_loss / max(train_steps, 1),
            "valid_loss": mean_valid,
            "regime_accuracy": regime_accuracy,
            "train_steps": float(train_steps),
            "valid_steps": float(valid_steps),
        }
        logs.append(epoch_log)
        if valid_steps > 0 and math.isfinite(mean_valid) and mean_valid <= best_metric:
            best_metric = mean_valid
            V21_XLSTM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_columns": feature_columns,
                    "sequence_len": int(args.sequence_len),
                    "feature_mean": sequence_bundle.feature_mean.tolist(),
                    "feature_std": sequence_bundle.feature_std.tolist(),
                    "d_model": int(args.d_model),
                    "n_layers": int(args.n_layers),
                    "logs": logs,
                },
                V21_XLSTM_MODEL_PATH,
            )

    report = {
        "feature_count": int(len(feature_columns)),
        "dataset_sequences": int(len(dataset)),
        "train_sequences": int(len(train_dataset)),
        "valid_sequences": int(len(valid_dataset)),
        "best_valid_loss": round(float(best_metric), 6) if math.isfinite(best_metric) else None,
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
