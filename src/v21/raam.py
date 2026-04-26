from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from config.project_config import V21_RAAM_INDEX_PATH, V21_RAAM_OUTCOMES_PATH

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - fallback is validated in tests
    faiss = None


@dataclass(frozen=True)
class RAAMRetrieval:
    indices: list[int]
    distances: list[float]
    weights: list[float]
    outcomes: list[dict[str, Any]]


class RetrievalAugmentedAnalogMemory:
    def __init__(self, embedding_dim: int = 128, n_neighbors: int = 10) -> None:
        self.embedding_dim = int(embedding_dim)
        self.n_neighbors = int(n_neighbors)
        self.index = faiss.IndexFlatL2(self.embedding_dim) if faiss is not None else None
        self.embeddings = np.zeros((0, self.embedding_dim), dtype=np.float32)
        self.outcomes: list[dict[str, Any]] = []

    def build(self, embeddings: np.ndarray, outcomes: Sequence[dict[str, Any]]) -> None:
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected embeddings of shape (N, {self.embedding_dim}).")
        if len(vectors) != len(outcomes):
            raise ValueError("Embeddings and outcomes must be aligned.")
        self.embeddings = vectors
        self.outcomes = [dict(item) for item in outcomes]
        if self.index is not None:
            self.index.reset()
            if len(vectors):
                self.index.add(vectors)

    def retrieve(self, query_embedding: np.ndarray, k: int | None = None) -> RAAMRetrieval:
        if len(self.embeddings) == 0:
            return RAAMRetrieval(indices=[], distances=[], weights=[], outcomes=[])
        k = min(int(k or self.n_neighbors), len(self.embeddings))
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected query of width {self.embedding_dim}.")

        if self.index is not None:
            distances, indices = self.index.search(query, k)
            index_values = indices[0].astype(int).tolist()
            distance_values = distances[0].astype(float).tolist()
        else:
            deltas = self.embeddings - query
            all_distances = np.sum(deltas * deltas, axis=1)
            idx = np.argpartition(all_distances, k - 1)[:k]
            idx = idx[np.argsort(all_distances[idx])]
            index_values = idx.astype(int).tolist()
            distance_values = all_distances[idx].astype(float).tolist()

        weights = 1.0 / (np.asarray(distance_values, dtype=np.float64) + 1e-6)
        weights = weights / max(float(weights.sum()), 1e-12)
        outcome_values = [self.outcomes[index] for index in index_values]
        return RAAMRetrieval(
            indices=index_values,
            distances=[round(float(value), 6) for value in distance_values],
            weights=[round(float(value), 6) for value in weights.tolist()],
            outcomes=outcome_values,
        )

    def get_analog_prior(self, query_embedding: np.ndarray, k: int | None = None) -> dict[str, float]:
        retrieved = self.retrieve(query_embedding, k=k)
        if not retrieved.outcomes:
            return {
                "analog_direction_15m": 0.0,
                "analog_vol_next": 0.0,
                "analog_regime_match_rate": 0.0,
                "analog_max_drawdown": 0.0,
            }
        weights = np.asarray(retrieved.weights, dtype=np.float64)
        return {
            "analog_direction_15m": round(
                float(sum(weight * float(item.get("direction_15m", 0.0)) for weight, item in zip(weights, retrieved.outcomes, strict=False))),
                6,
            ),
            "analog_vol_next": round(
                float(sum(weight * float(item.get("vol_next", 0.0)) for weight, item in zip(weights, retrieved.outcomes, strict=False))),
                6,
            ),
            "analog_regime_match_rate": round(
                float(sum(weight * float(item.get("regime_match", 0.0)) for weight, item in zip(weights, retrieved.outcomes, strict=False))),
                6,
            ),
            "analog_max_drawdown": round(
                float(sum(weight * float(item.get("max_drawdown", 0.0)) for weight, item in zip(weights, retrieved.outcomes, strict=False))),
                6,
            ),
        }

    def save(self, index_path: Path = V21_RAAM_INDEX_PATH, outcomes_path: Path = V21_RAAM_OUTCOMES_PATH) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        outcomes_path.parent.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(index_path))
        else:
            np.save(index_path.with_suffix(".npy"), self.embeddings)
        outcomes_path.write_text(json.dumps(self.outcomes, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        index_path: Path = V21_RAAM_INDEX_PATH,
        outcomes_path: Path = V21_RAAM_OUTCOMES_PATH,
        *,
        embedding_dim: int = 128,
        n_neighbors: int = 10,
    ) -> "RetrievalAugmentedAnalogMemory":
        memory = cls(embedding_dim=embedding_dim, n_neighbors=n_neighbors)
        if outcomes_path.exists():
            memory.outcomes = json.loads(outcomes_path.read_text(encoding="utf-8"))
        if faiss is not None and index_path.exists():
            memory.index = faiss.read_index(str(index_path))
            memory.embeddings = np.zeros((memory.index.ntotal, memory.embedding_dim), dtype=np.float32)
        else:
            npy_path = index_path.with_suffix(".npy")
            if npy_path.exists():
                memory.embeddings = np.load(npy_path).astype(np.float32)
        return memory


__all__ = ["RAAMRetrieval", "RetrievalAugmentedAnalogMemory"]
