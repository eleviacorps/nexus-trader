from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.v12.sarv import run_sarv_validation, write_sarv_report
from src.v12.tctl import load_tctl_model, prepare_tctl_candidates, replay_candidates_with_online_bcfe


def _latest_ninety_day_slice(frame: pd.DataFrame) -> pd.DataFrame:
    timestamps = pd.to_datetime(frame['timestamp'], utc=True, errors='coerce')
    end = timestamps.max()
    start = end - pd.Timedelta(days=90)
    return frame.loc[timestamps >= start].copy().reset_index(drop=True)


def main() -> int:
    frame = pd.read_parquet(PROJECT_ROOT / 'outputs' / 'v10' / 'branch_features_v10_full.parquet')
    train_candidates, valid_candidates, _ = prepare_tctl_candidates(frame, validation_fraction=0.2)
    valid_window = _latest_ninety_day_slice(valid_candidates)
    bar_replay_candidates = replay_candidates_with_online_bcfe(valid_window)
    model, feature_names = load_tctl_model()
    report = run_sarv_validation(
        model=model,
        train_candidates=train_candidates,
        archive_candidates=valid_window,
        bar_replay_candidates=bar_replay_candidates,
        feature_names=feature_names,
    )
    report['bar_replay_feature_mode'] = 'online_bcfe'
    report['archive_candidate_count'] = int(len(valid_window))
    report['bar_replay_candidate_count'] = int(len(bar_replay_candidates))
    report['replay_window'] = {
        'start': str(pd.to_datetime(valid_window['timestamp'], utc=True).min()),
        'end': str(pd.to_datetime(valid_window['timestamp'], utc=True).max()),
        'days': 90,
    }
    path = write_sarv_report(report)
    print(str(path), flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
