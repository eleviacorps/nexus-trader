from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config.project_config import V13_PAPER_TRADE_LOG_PATH, V13_RCPC_CALIBRATOR_PATH
from src.v12.bar_consistent_features import load_default_raw_bars
from src.v12.tctl import replay_candidates_with_online_bcfe
from src.v13.cabr import augment_cabr_context, load_cabr_model, load_v13_candidate_frames, score_cabr_model
from src.v13.policy_utils import attach_execution_prices, derive_deployable_regimes, enrich_v13_policy_frame, fit_uts_selector, generate_v13_decisions
from src.v13.rcpc import RegimeConditionalPriorCalibrator
from src.v13.s3pta import PaperTradeAccumulator


def _latest_ninety_days(frame: pd.DataFrame) -> pd.DataFrame:
    timestamps = pd.to_datetime(frame['timestamp'], utc=True, errors='coerce')
    end = timestamps.max()
    start = end - pd.Timedelta(days=90)
    return frame.loc[timestamps >= start].copy().reset_index(drop=True)


def main() -> int:
    archive = pd.read_parquet(PROJECT_ROOT / 'outputs' / 'v10' / 'branch_features_v10_full.parquet')
    train_frame, valid_frame, _, _ = load_v13_candidate_frames(archive)
    valid_window = _latest_ninety_days(valid_frame)
    replay_window = augment_cabr_context(replay_candidates_with_online_bcfe(valid_window))
    model, branch_cols, context_cols, _ = load_cabr_model(map_location='cpu')

    train_frame = train_frame.copy()
    replay_window = replay_window.copy()
    train_frame['cabr_score'] = score_cabr_model(model, train_frame, branch_feature_names=branch_cols, context_feature_names=context_cols, device='cpu')
    replay_window['cabr_score'] = score_cabr_model(model, replay_window, branch_feature_names=branch_cols, context_feature_names=context_cols, device='cpu')

    calibrator = RegimeConditionalPriorCalibrator()
    train_enriched = enrich_v13_policy_frame(train_frame, cabr_score_column='cabr_score', calibrator=calibrator)
    replay_enriched = enrich_v13_policy_frame(replay_window, cabr_score_column='cabr_score', calibrator=calibrator)
    threshold_selector = fit_uts_selector(train_enriched)
    deployable_regimes = derive_deployable_regimes(train_enriched)

    raw_bars = load_default_raw_bars(
        start=pd.to_datetime(replay_enriched['timestamp'], utc=True).min() - pd.Timedelta(minutes=30),
        end=pd.to_datetime(replay_enriched['timestamp'], utc=True).max() + pd.Timedelta(minutes=30),
    )
    replay_enriched = attach_execution_prices(replay_enriched, raw_bars)
    executed, skipped = generate_v13_decisions(replay_enriched, threshold_selector=threshold_selector, deployable_regimes=deployable_regimes)

    if V13_PAPER_TRADE_LOG_PATH.exists():
        V13_PAPER_TRADE_LOG_PATH.unlink()
    accumulator = PaperTradeAccumulator(V13_PAPER_TRADE_LOG_PATH)
    completed = []
    for row in executed.itertuples(index=False):
        direction = 'BUY' if int(round(float(getattr(row, 'setl_trade_direction', getattr(row, 'branch_direction', 1.0))))) >= 0 else 'SELL'
        payload = accumulator.log_completed_trade(
            symbol='XAUUSD',
            direction=direction,
            uts_score=float(row.uts_score),
            cabr_score=float(row.cabr_raw_score),
            regime=str(row.regime_class),
            entry_price=float(row.entry_price),
            exit_price=float(row.exit_price),
            entry_time=str(row.decision_ts),
            exit_time=str(row.exit_ts),
        )
        completed.append(payload)
        calibrator.record_outcome(float(row.cabr_raw_score), payload['outcome'] == 'win')
    calibrator.save(V13_RCPC_CALIBRATOR_PATH)

    skip_breakdown: dict[str, int] = {}
    for item in skipped:
        reason = str(item.get('reason', 'unknown'))
        skip_breakdown[reason] = skip_breakdown.get(reason, 0) + 1

    summary = accumulator.summary()
    report = {
        'paper_trade_summary': summary,
        'executed_count': int(len(completed)),
        'skipped_count': int(len(skipped)),
        'skip_reason_breakdown': skip_breakdown,
        'deployable_regimes': sorted(deployable_regimes),
        'uses_learned_calibration': bool(calibrator.uses_learned_calibration),
        'calibrator_summary': calibrator.summary(),
        'window': {
            'start': str(pd.to_datetime(replay_enriched['timestamp'], utc=True).min()),
            'end': str(pd.to_datetime(replay_enriched['timestamp'], utc=True).max()),
        },
    }
    out_path = PROJECT_ROOT / 'outputs' / 'v13' / 's3pta_summary.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(str(out_path), flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
