from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import OUTPUTS_V21_DIR, V21_FEATURES_PATH, V21_RAAM_INDEX_PATH, V21_RAAM_OUTCOMES_PATH, V21_XLSTM_MODEL_PATH
from src.v21.raam import RetrievalAugmentedAnalogMemory
from src.v21.xlstm_backbone import NexusXLSTM


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    numeric = frame.select_dtypes(include=["number"])
    return [
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


def _farthest_point_sampling(vectors: np.ndarray, max_points: int) -> np.ndarray:
    if len(vectors) <= max_points:
        return np.arange(len(vectors), dtype=np.int64)
    chosen = [0]
    min_dist = np.full(len(vectors), np.inf, dtype=np.float32)
    for _ in range(1, int(max_points)):
        latest = vectors[chosen[-1]]
        dist = np.sum((vectors - latest) ** 2, axis=1)
        min_dist = np.minimum(min_dist, dist)
        min_dist[chosen] = -1.0
        next_idx = int(np.argmax(min_dist))
        chosen.append(next_idx)
    return np.asarray(chosen, dtype=np.int64)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the V21 RAAM index from xLSTM hidden states.")
    parser.add_argument("--sequence-len", type=int, default=240)
    parser.add_argument("--max-windows", type=int, default=100000)
    parser.add_argument("--sample-stride", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    if not V21_FEATURES_PATH.exists():
        raise SystemExit(f"Missing V21 features at {V21_FEATURES_PATH}.")
    if not V21_XLSTM_MODEL_PATH.exists():
        raise SystemExit(f"Missing xLSTM checkpoint at {V21_XLSTM_MODEL_PATH}.")

    frame = pd.read_parquet(V21_FEATURES_PATH).select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
    feature_columns = _feature_columns(frame)
    checkpoint = torch.load(V21_XLSTM_MODEL_PATH, map_location="cpu")
    model = NexusXLSTM(
        n_features=len(feature_columns),
        d_model=512,
        n_layers=4,
        n_regimes=6,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    values = frame[feature_columns].to_numpy(dtype=np.float32)
    regime_ids = pd.to_numeric(frame.get("hmm_state"), errors="coerce").fillna(0).clip(lower=0, upper=5).astype(np.int64).to_numpy()
    future_15 = pd.to_numeric(frame.get("future_return_15m"), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    future_30 = pd.to_numeric(frame.get("future_return_30m"), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    range_15 = pd.to_numeric(frame.get("range_forward_15m"), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    end_indices = np.arange(int(args.sequence_len), len(frame), max(1, int(args.sample_stride)), dtype=np.int64)
    embeddings: list[np.ndarray] = []
    outcomes: list[dict[str, float]] = []

    with torch.no_grad():
        for start in range(0, len(end_indices), int(args.batch_size)):
            batch_indices = end_indices[start : start + int(args.batch_size)]
            x_batch = np.stack([values[idx - int(args.sequence_len) : idx] for idx in batch_indices], axis=0)
            regime_batch = np.stack([regime_ids[idx - int(args.sequence_len) : idx] for idx in batch_indices], axis=0)
            outputs = model(
                torch.tensor(x_batch, dtype=torch.float32),
                torch.tensor(regime_batch, dtype=torch.long),
            )
            embeddings.append(outputs["hidden"].cpu().numpy().astype(np.float32))
            for idx in batch_indices.tolist():
                outcomes.append(
                    {
                        "direction_15m": float(np.sign(future_15[idx])),
                        "vol_next": float(abs(future_30[idx])),
                        "regime_match": 1.0,
                        "max_drawdown": float(abs(range_15[idx])),
                        "regime_transition": float(regime_ids[min(idx + 1, len(regime_ids) - 1)] != regime_ids[idx]),
                    }
                )

    all_embeddings = np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, 512), dtype=np.float32)
    if len(all_embeddings) == 0:
        raise SystemExit("No RAAM embeddings were produced.")

    selected = _farthest_point_sampling(all_embeddings, int(args.max_windows))
    selected_embeddings = all_embeddings[selected]
    selected_outcomes = [outcomes[int(index)] for index in selected.tolist()]

    memory = RetrievalAugmentedAnalogMemory(embedding_dim=selected_embeddings.shape[1], n_neighbors=10)
    memory.build(selected_embeddings, selected_outcomes)
    memory.save(index_path=V21_RAAM_INDEX_PATH, outcomes_path=V21_RAAM_OUTCOMES_PATH)

    report = {
        "candidate_windows": int(len(all_embeddings)),
        "selected_windows": int(len(selected_embeddings)),
        "embedding_dim": int(selected_embeddings.shape[1]),
        "sequence_len": int(args.sequence_len),
        "sample_stride": int(args.sample_stride),
        "index_path": str(V21_RAAM_INDEX_PATH),
        "outcomes_path": str(V21_RAAM_OUTCOMES_PATH),
    }
    report_path = OUTPUTS_V21_DIR / "raam_build_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
