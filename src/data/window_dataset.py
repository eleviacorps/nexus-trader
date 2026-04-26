from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

try:
    import torch  # type: ignore
    from torch.utils.data import Dataset  # type: ignore
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object  # type: ignore


@dataclass
class WindowSample:
    features: List[List[float]]
    target: float
    sim_target: float | None = None
    sim_confidence: float | None = None


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_sliding_windows(
    rows: Sequence[Dict[str, str]],
    feature_columns: Sequence[str],
    target_column: str,
    sequence_len: int,
    sim_targets: Sequence[float] | None = None,
    sim_confidence: Sequence[float] | None = None,
) -> List[WindowSample]:
    samples: List[WindowSample] = []
    for idx in range(len(rows) - sequence_len):
        window = rows[idx : idx + sequence_len]
        sample_features = [
            [float(row.get(column, 0.0) or 0.0) for column in feature_columns]
            for row in window
        ]
        target_row = rows[idx + sequence_len]
        sample = WindowSample(
            features=sample_features,
            target=float(target_row.get(target_column, 0.0) or 0.0),
        )
        if sim_targets is not None and idx < len(sim_targets):
            sample.sim_target = float(sim_targets[idx])
        if sim_confidence is not None and idx < len(sim_confidence):
            sample.sim_confidence = float(sim_confidence[idx])
        samples.append(sample)
    return samples


class MarketWindowDataset(Dataset):
    def __init__(self, samples: Sequence[WindowSample]):
        if torch is None:
            raise ImportError("PyTorch is required to instantiate MarketWindowDataset.")
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        features = torch.tensor(sample.features, dtype=torch.float32)
        target = torch.tensor(sample.target, dtype=torch.float32)
        sim_target = torch.tensor(sample.sim_target if sample.sim_target is not None else 0.0, dtype=torch.float32)
        sim_confidence = torch.tensor(sample.sim_confidence if sample.sim_confidence is not None else 0.0, dtype=torch.float32)
        return features, target, sim_target, sim_confidence
