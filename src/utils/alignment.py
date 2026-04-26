from __future__ import annotations

from bisect import bisect_right
from datetime import datetime, timedelta
from typing import Iterable, List, Sequence


def _coerce_vector(values: Sequence[float], dims: int) -> List[float]:
    vector = list(values[:dims])
    if len(vector) < dims:
        vector.extend([0.0] * (dims - len(vector)))
    return [float(value) for value in vector]


def zero_embedding_matrix(length: int, dims: int) -> List[List[float]]:
    return [[0.0 for _ in range(dims)] for _ in range(length)]


def forward_fill_embeddings(
    bar_times: Sequence[datetime],
    event_times: Sequence[datetime],
    event_vectors: Sequence[Sequence[float]],
    dims: int,
    fill_limit_minutes: int,
) -> List[List[float]]:
    if len(event_times) != len(event_vectors):
        raise ValueError("event_times and event_vectors must have matching lengths.")
    if not bar_times:
        return []
    if not event_times:
        return zero_embedding_matrix(len(bar_times), dims)

    ordered = sorted(zip(event_times, event_vectors), key=lambda item: item[0])
    times = [item[0] for item in ordered]
    vectors = [_coerce_vector(item[1], dims) for item in ordered]
    limit = timedelta(minutes=fill_limit_minutes)

    aligned: List[List[float]] = []
    for bar_time in bar_times:
        index = bisect_right(times, bar_time) - 1
        if index < 0:
            aligned.append([0.0] * dims)
            continue
        if bar_time - times[index] > limit:
            aligned.append([0.0] * dims)
            continue
        aligned.append(list(vectors[index]))
    return aligned


def concatenate_feature_blocks(
    price_rows: Sequence[Sequence[float]],
    news_rows: Sequence[Sequence[float]],
    crowd_rows: Sequence[Sequence[float]],
) -> List[List[float]]:
    if not (len(price_rows) == len(news_rows) == len(crowd_rows)):
        raise ValueError("Price, news, and crowd blocks must have the same row count.")
    fused_rows: List[List[float]] = []
    for price_row, news_row, crowd_row in zip(price_rows, news_rows, crowd_rows):
        fused_rows.append(list(price_row) + list(news_row) + list(crowd_row))
    return fused_rows


def infer_alignment_coverage(aligned_rows: Iterable[Sequence[float]]) -> float:
    rows = list(aligned_rows)
    if not rows:
        return 0.0
    populated = sum(1 for row in rows if any(float(value) != 0.0 for value in row))
    return populated / len(rows)
