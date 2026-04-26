from __future__ import annotations

import argparse
import json
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

from config.project_config import OUTPUTS_V20_DIR, V20_SJD_DATASET_PATH, V20_SJD_MODEL_PATH
from src.v20.macro_features import MACRO_FEATURE_COLS
from src.v20.sjd_v20 import CONFIDENCE_LABELS, STANCE_LABELS, SJD_V20


def _label_index(values: pd.Series, labels: list[str]) -> np.ndarray:
    mapping = {label: index for index, label in enumerate(labels)}
    return values.astype(str).map(mapping).fillna(0).astype(np.int64).to_numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the V20 SJD fallback model.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    if not V20_SJD_DATASET_PATH.exists():
        raise SystemExit(f"Missing V20 SJD dataset at {V20_SJD_DATASET_PATH}. Run generate_sjd_dataset_v20.py first.")

    frame = pd.read_parquet(V20_SJD_DATASET_PATH).replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    macro_columns = [col for col in MACRO_FEATURE_COLS if col in frame.columns]
    feature_columns = [
        col
        for col in frame.select_dtypes(include=["number"]).columns
        if col not in {"tp_offset", "sl_offset", "kelly", "source_unique_rows", "macro_feature_count"} and col not in macro_columns
    ]
    if not feature_columns or not macro_columns:
        raise SystemExit("V20 SJD dataset does not contain enough feature or macro columns for training.")

    split_index = max(int(len(frame) * 0.9), 1)
    train = frame.iloc[:split_index].copy()
    valid = frame.iloc[split_index:].copy() if split_index < len(frame) else frame.iloc[-max(min(5000, len(frame)), 1) :].copy()

    train_dataset = TensorDataset(
        torch.tensor(train[feature_columns].to_numpy(dtype=np.float32)),
        torch.tensor(train[macro_columns].to_numpy(dtype=np.float32)),
        torch.tensor(_label_index(train["stance"], STANCE_LABELS)),
        torch.tensor(_label_index(train["confidence"], CONFIDENCE_LABELS)),
        torch.tensor(train["tp_offset"].to_numpy(dtype=np.float32)),
        torch.tensor(train["sl_offset"].to_numpy(dtype=np.float32)),
        torch.tensor(train["kelly"].to_numpy(dtype=np.float32)),
    )
    valid_dataset = TensorDataset(
        torch.tensor(valid[feature_columns].to_numpy(dtype=np.float32)),
        torch.tensor(valid[macro_columns].to_numpy(dtype=np.float32)),
        torch.tensor(_label_index(valid["stance"], STANCE_LABELS)),
        torch.tensor(_label_index(valid["confidence"], CONFIDENCE_LABELS)),
        torch.tensor(valid["tp_offset"].to_numpy(dtype=np.float32)),
        torch.tensor(valid["sl_offset"].to_numpy(dtype=np.float32)),
        torch.tensor(valid["kelly"].to_numpy(dtype=np.float32)),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SJD_V20(d_features=len(feature_columns), d_macro=len(macro_columns)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    stance_loss = nn.CrossEntropyLoss()
    confidence_loss = nn.CrossEntropyLoss()
    reg_loss = nn.SmoothL1Loss()
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_state: dict[str, object] | None = None
    best_accuracy = -1.0
    logs: list[dict[str, float]] = []

    for epoch in range(int(args.epochs)):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            features, macro, stance_target, confidence_target, tp_target, sl_target, kelly_target = [tensor.to(device, non_blocking=True) for tensor in batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(features, macro)
                loss = (
                    stance_loss(outputs["stance"], stance_target)
                    + 0.5 * confidence_loss(outputs["confidence"], confidence_target)
                    + 0.10 * reg_loss(outputs["tp_offset"], tp_target)
                    + 0.10 * reg_loss(outputs["sl_offset"], sl_target)
                    + 0.05 * reg_loss(outputs["kelly"], kelly_target)
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu())

        model.eval()
        correct = 0
        total = 0
        tp_errors: list[float] = []
        sl_errors: list[float] = []
        with torch.no_grad():
            for batch in valid_loader:
                features, macro, stance_target, _, tp_target, sl_target, _ = [tensor.to(device, non_blocking=True) for tensor in batch]
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    outputs = model(features, macro)
                predictions = outputs["stance"].argmax(dim=-1)
                correct += int((predictions == stance_target).sum().item())
                total += int(stance_target.numel())
                tp_errors.extend(torch.abs(outputs["tp_offset"] - tp_target).detach().cpu().tolist())
                sl_errors.extend(torch.abs(outputs["sl_offset"] - sl_target).detach().cpu().tolist())

        accuracy = float(correct / total) if total else 0.0
        epoch_log = {
            "epoch": float(epoch + 1),
            "train_loss": running_loss / max(len(train_loader), 1),
            "valid_stance_accuracy": accuracy,
            "valid_tp_mae": float(np.mean(tp_errors)) if tp_errors else 0.0,
            "valid_sl_mae": float(np.mean(sl_errors)) if sl_errors else 0.0,
        }
        logs.append(epoch_log)
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_state = {
                "model_state": model.state_dict(),
                "feature_columns": feature_columns,
                "macro_columns": macro_columns,
                "logs": logs,
            }

    if best_state is None:
        raise SystemExit("SJD V20 training did not produce a checkpoint.")

    V20_SJD_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, V20_SJD_MODEL_PATH)
    report = {
        "dataset_rows": int(len(frame)),
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "feature_count": int(len(feature_columns)),
        "macro_feature_count": int(len(macro_columns)),
        "best_valid_stance_accuracy": round(float(best_accuracy), 6),
        "target_accuracy": 0.82,
        "target_met": bool(best_accuracy >= 0.82),
        "logs": logs,
        "note": "This is a local fallback SJD V20 pass with a balanced/resampled dataset, not the full GPU-scale 50k unique-row training target.",
    }
    report_path = OUTPUTS_V20_DIR / "sjd_v20_training_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(V20_SJD_MODEL_PATH), **report}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
