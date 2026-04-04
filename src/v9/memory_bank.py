from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class MemoryBankQueryResult:
    analog_confidence: float
    bullish_probability: float
    mean_distance: float
    top_k_indices: list[int]


@dataclass(frozen=True)
class MemoryBankReport:
    device: str
    window_size: int
    sample_count: int
    embedding_dim: int
    epochs: int
    train_loss: float
    validation_accuracy: float


class MemoryBankEncoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 64, hidden_dim: int = 256, class_count: int = 2) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, class_count)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        embedding = self.backbone(inputs)
        return torch.nn.functional.normalize(embedding, p=2.0, dim=-1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encode(inputs)
        logits = self.classifier(embedding)
        return embedding, logits


def _resolve_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_memory_bank_windows(
    feature_matrix: np.ndarray,
    targets: np.ndarray,
    *,
    window_size: int = 60,
    sample_stride: int = 15,
    max_samples: int = 25000,
) -> tuple[np.ndarray, np.ndarray]:
    usable = max(0, len(feature_matrix) - window_size + 1)
    starts = np.arange(0, usable, max(1, sample_stride), dtype=np.int64)
    if max_samples > 0 and len(starts) > max_samples:
        keep = np.linspace(0, len(starts) - 1, max_samples, dtype=np.int64)
        starts = starts[keep]
    windows = np.stack([feature_matrix[start : start + window_size].reshape(-1) for start in starts]).astype(np.float32)
    labels = targets[starts + window_size - 1].astype(np.int64)
    return windows, labels


def train_memory_bank_encoder(
    windows: np.ndarray,
    labels: np.ndarray,
    *,
    embedding_dim: int = 64,
    epochs: int = 8,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    device: str | None = None,
) -> tuple[MemoryBankEncoder, MemoryBankReport]:
    torch_device = _resolve_device(device)
    model = MemoryBankEncoder(input_dim=windows.shape[1], embedding_dim=embedding_dim).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    split = max(1, int(len(windows) * 0.8))
    train_x = torch.from_numpy(windows[:split]).to(torch_device)
    train_y = torch.from_numpy(labels[:split]).to(torch_device)
    valid_x = torch.from_numpy(windows[split:]).to(torch_device) if split < len(windows) else train_x[:1]
    valid_y = torch.from_numpy(labels[split:]).to(torch_device) if split < len(labels) else train_y[:1]
    loss_fn = nn.CrossEntropyLoss()
    final_loss = 0.0
    for _ in range(max(1, int(epochs))):
        model.train()
        order = torch.randperm(train_x.shape[0], device=torch_device)
        losses = []
        for start in range(0, train_x.shape[0], max(1, int(batch_size))):
            index = order[start : start + batch_size]
            batch_x = train_x[index]
            batch_y = train_y[index]
            embedding, logits = model(batch_x)
            class_loss = loss_fn(logits, batch_y)
            similarity = embedding @ embedding.T
            positive_mask = (batch_y[:, None] == batch_y[None, :]).float()
            contrastive = torch.logsumexp(similarity, dim=1) - ((similarity * positive_mask).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1.0))
            loss = class_loss + 0.10 * contrastive.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(losses)) if losses else 0.0
    model.eval()
    with torch.no_grad():
        _, logits = model(valid_x)
        predictions = torch.argmax(logits, dim=1)
        validation_accuracy = float((predictions == valid_y).float().mean().detach().cpu())
    report = MemoryBankReport(
        device=str(torch_device),
        window_size=int(windows.shape[1] // 100) if windows.shape[1] % 100 == 0 else 0,
        sample_count=int(len(windows)),
        embedding_dim=int(embedding_dim),
        epochs=max(1, int(epochs)),
        train_loss=final_loss,
        validation_accuracy=validation_accuracy,
    )
    return model, report


def build_memory_bank_index(model: MemoryBankEncoder, windows: np.ndarray, *, device: str | None = None) -> np.ndarray:
    torch_device = _resolve_device(device)
    model = model.to(torch_device)
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(windows), 2048):
            batch = torch.from_numpy(windows[start : start + 2048]).to(torch_device)
            outputs.append(model.encode(batch).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 64), dtype=np.float32)


def save_memory_bank(
    encoder_path: Path,
    index_path: Path,
    report_path: Path,
    model: MemoryBankEncoder,
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    report: MemoryBankReport,
    window_size: int,
    sample_stride: int,
) -> None:
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "input_dim": model.backbone[0].in_features}, encoder_path)
    np.savez_compressed(index_path, embeddings=embeddings.astype(np.float32), labels=labels.astype(np.int64), window_size=int(window_size), sample_stride=int(sample_stride))
    report_path.write_text(json.dumps(report.__dict__, indent=2), encoding="utf-8")


def load_memory_bank(encoder_path: Path, index_path: Path, *, device: str | None = None) -> tuple[MemoryBankEncoder | None, dict[str, np.ndarray] | None]:
    if not encoder_path.exists() or not index_path.exists():
        return None, None
    payload = torch.load(encoder_path, map_location=_resolve_device(device))
    model = MemoryBankEncoder(input_dim=int(payload.get("input_dim", 6000)))
    model.load_state_dict(payload["state_dict"])
    bundle = np.load(index_path)
    return model, {key: np.asarray(bundle[key]) for key in bundle.files}


def query_memory_bank(
    model: MemoryBankEncoder,
    bank: Mapping[str, np.ndarray],
    window: np.ndarray,
    *,
    top_k: int = 20,
    device: str | None = None,
) -> MemoryBankQueryResult:
    torch_device = _resolve_device(device)
    model = model.to(torch_device)
    vector = torch.from_numpy(window.reshape(1, -1).astype(np.float32)).to(torch_device)
    model.eval()
    with torch.no_grad():
        embedding = model.encode(vector).detach().cpu().numpy().astype(np.float32)
    bank_embeddings = np.asarray(bank["embeddings"], dtype=np.float32)
    if bank_embeddings.size == 0:
        return MemoryBankQueryResult(analog_confidence=0.0, bullish_probability=0.5, mean_distance=0.0, top_k_indices=[])
    distances = np.linalg.norm(bank_embeddings - embedding, axis=1)
    ranking = np.argsort(distances)[: max(1, int(top_k))]
    weights = 1.0 / np.maximum(distances[ranking], 1e-6)
    labels = np.asarray(bank["labels"], dtype=np.int64)[ranking]
    bullish_probability = float(np.average(labels.astype(np.float32), weights=weights))
    analog_confidence = float(np.clip(abs(bullish_probability - 0.5) * 2.0, 0.0, 1.0))
    return MemoryBankQueryResult(
        analog_confidence=analog_confidence,
        bullish_probability=bullish_probability,
        mean_distance=float(np.mean(distances[ranking])),
        top_k_indices=[int(value) for value in ranking.tolist()],
    )
