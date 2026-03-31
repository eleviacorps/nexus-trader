from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.data.fused_dataset import DatasetSlice


@dataclass(frozen=True)
class SplitConfig:
    train: DatasetSlice
    val: DatasetSlice
    test: DatasetSlice
    mode: str


def split_by_years(
    timestamps: Sequence[str],
    sequence_len: int,
    train_years: Sequence[int],
    val_years: Sequence[int],
    test_years: Sequence[int],
) -> SplitConfig:
    values = np.asarray(timestamps, dtype='datetime64[ns]')
    years = values.astype('datetime64[Y]').astype(int) + 1970
    usable = len(values) - sequence_len
    if usable <= 0:
        raise ValueError('Not enough rows for sequence generation.')
    target_years_view = years[sequence_len - 1 : sequence_len - 1 + usable]

    def build_slice(target_years: Sequence[int]) -> DatasetSlice:
        if not target_years:
            raise ValueError('Missing year configuration for one of the splits.')
        mask = np.isin(target_years_view, [int(year) for year in target_years])
        valid_positions = np.flatnonzero(mask)
        if valid_positions.size == 0:
            raise ValueError('Requested year split produced no usable sequence windows.')
        return DatasetSlice(int(valid_positions[0]), int(valid_positions[-1] + 1))

    return SplitConfig(
        train=build_slice(train_years),
        val=build_slice(val_years),
        test=build_slice(test_years),
        mode='year',
    )
