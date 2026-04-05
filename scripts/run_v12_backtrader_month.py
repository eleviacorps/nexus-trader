from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BACKTRADER_ROOT = PROJECT_ROOT / 'SimilarExistingSolutions' / 'backtrader-master'
if str(BACKTRADER_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKTRADER_ROOT))

import backtrader as bt  # type: ignore
import numpy as np
import pandas as pd

from config.project_config import V13_BACKTRADER_MONTH_REPORT_PATH, V13_PAPER_TRADE_LOG_PATH, V13_RCPC_CALIBRATOR_PATH
from src.v12.backtrader_strategy import NexusV12Strategy, V12SignalData
from src.v12.bar_consistent_features import load_default_raw_bars
from src.v12.sarv import run_scored_sarv_validation
from src.v12.tctl import replay_candidates_with_online_bcfe
from src.v13.cabr import augment_cabr_context, load_cabr_model, load_v13_candidate_frames, score_cabr_model
from src.v13.daps import daps_lot_size
from src.v13.policy_utils import attach_execution_prices, derive_deployable_regimes, enrich_v13_policy_frame, fit_uts_selector, generate_v13_decisions
from src.v13.rcpc import RegimeConditionalPriorCalibrator
from src.v13.s3pta import PaperTradeAccumulator


def _available_months(frame: pd.DataFrame) -> list[str]:
    ts = pd.to_datetime(frame['timestamp'], utc=True, errors='coerce')
    return sorted(pd.Series(ts.dt.to_period('M').astype(str)).dropna().unique().tolist())


def _trade_cost_pips(*, spread_pips: float, slippage_pips: float, commission_usd: float, contract_size_oz: float, pip_size: float) -> float:
    commission_pips = float(commission_usd) / max(float(contract_size_oz) * float(pip_size), 1e-6)
    return float(spread_pips + slippage_pips + commission_pips)


def _v13_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    archive = pd.read_parquet(PROJECT_ROOT / 'outputs' / 'v10' / 'branch_features_v10_full.parquet')
    train_frame, valid_frame, branch_cols, context_cols = load_v13_candidate_frames(archive)
    replay_frame = augment_cabr_context(replay_candidates_with_online_bcfe(valid_frame))
    return train_frame, valid_frame, replay_frame, branch_cols, context_cols


def _score_v13_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, set[str], RegimeConditionalPriorCalibrator]:
    train_frame, valid_frame, replay_frame, branch_cols, context_cols = _v13_frames()
    model, _, _, payload = load_cabr_model(map_location='cpu')
    train_frame = train_frame.copy()
    valid_frame = valid_frame.copy()
    replay_frame = replay_frame.copy()
    train_frame['cabr_score'] = score_cabr_model(model, train_frame, branch_feature_names=branch_cols, context_feature_names=context_cols, device='cpu')
    valid_frame['cabr_score'] = score_cabr_model(model, valid_frame, branch_feature_names=branch_cols, context_feature_names=context_cols, device='cpu')
    replay_frame['cabr_score'] = score_cabr_model(model, replay_frame, branch_feature_names=branch_cols, context_feature_names=context_cols, device='cpu')
    calibrator = RegimeConditionalPriorCalibrator.load(V13_RCPC_CALIBRATOR_PATH)
    train_enriched = enrich_v13_policy_frame(train_frame, cabr_score_column='cabr_score', calibrator=calibrator)
    valid_enriched = enrich_v13_policy_frame(valid_frame, cabr_score_column='cabr_score', calibrator=calibrator)
    replay_enriched = enrich_v13_policy_frame(replay_frame, cabr_score_column='cabr_score', calibrator=calibrator)
    selector = fit_uts_selector(train_enriched)
    deployable_regimes = derive_deployable_regimes(train_enriched)
    return train_enriched, valid_enriched, replay_enriched, selector.thresholds, deployable_regimes, calibrator


def _apply_thresholds(frame: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    working = frame.copy()
    working['threshold_for_regime'] = [float(thresholds.get(str(regime), np.median(list(thresholds.values())) if thresholds else 0.5)) for regime in working['regime_class'].tolist()]
    return working


def _summarize_regimes(trades: list[dict]) -> dict[str, dict[str, float | int]]:
    if not trades:
        return {}
    frame = pd.DataFrame(trades)
    summary = {}
    for regime, subset in frame.groupby('dominant_regime', sort=True):
        net = subset['net_pnl_usd'].to_numpy(dtype=np.float64)
        summary[str(regime)] = {
            'trade_count': int(len(subset)),
            'win_rate': round(float(np.mean(net > 0.0)) if len(subset) else 0.0, 6),
            'net_pnl_usd': round(float(np.sum(net)), 6),
        }
    return summary


def _compute_preplanned_lots(frame: pd.DataFrame, *, capital: float, contract_size_oz: float, pip_size: float, cost_pips: float) -> pd.DataFrame:
    working = frame.sort_values('decision_ts').copy().reset_index(drop=True)
    equity = float(capital)
    recent_outcomes: list[int] = []
    planned_lots: list[float] = []
    for row in working.itertuples(index=False):
        recent_win_rate = 0.55 if not recent_outcomes else float(np.mean(recent_outcomes[-20:]))
        lot = daps_lot_size(
            base_capital=float(capital),
            current_equity=float(equity),
            recent_win_rate=float(recent_win_rate),
            regime=str(getattr(row, 'regime_class', 'unknown')),
            uts_score=float(getattr(row, 'uts_score', 0.5)),
        )
        lot = round(max(0.01, min(float(lot) * float(getattr(row, 'size_multiplier', 1.0)), 1.0)), 2)
        direction = 1.0 if int(np.sign(float(getattr(row, 'setl_trade_direction', getattr(row, 'branch_direction', 1.0))) or 1)) >= 0 else -1.0
        gross_pips = ((float(getattr(row, 'exit_price', getattr(row, 'entry_price', 0.0))) - float(getattr(row, 'entry_price', 0.0))) * direction) / max(float(pip_size), 1e-6)
        net_pips = gross_pips - float(cost_pips)
        pip_value = float(contract_size_oz) * float(pip_size) * float(lot)
        net_pnl = net_pips * pip_value
        equity += net_pnl
        recent_outcomes.append(1 if net_pnl > 0.0 else 0)
        planned_lots.append(float(lot))
    working['planned_lot'] = planned_lots
    return working


def run_v13(args: argparse.Namespace) -> int:
    train_frame, valid_frame, replay_frame, thresholds, deployable_regimes, calibrator = _score_v13_frames()
    threshold_median = float(np.median(list(thresholds.values()))) if thresholds else 0.5
    selector_wrapper = type('SelectorWrapper', (), {
        'thresholds': thresholds,
        'should_trade': lambda self, score, regime: float(score) >= float(self.thresholds.get(str(regime), threshold_median)),
    })()

    sarv_report = run_scored_sarv_validation(
        model_name='cabr_v13',
        train_scores=train_frame['uts_score'].to_numpy(dtype=np.float32),
        train_outcomes=train_frame['setl_target_net_unit_pnl'].to_numpy(dtype=np.float32),
        archive_candidates=_apply_thresholds(valid_frame, thresholds).assign(uts_score=valid_frame['uts_score']),
        archive_score_column='uts_score',
        bar_replay_candidates=_apply_thresholds(replay_frame, thresholds).assign(uts_score=replay_frame['uts_score']),
        replay_score_column='uts_score',
        paper_trade_log_path=V13_PAPER_TRADE_LOG_PATH,
    )

    available_months = _available_months(replay_frame)
    month_start = pd.Timestamp(f'{args.month}-01 00:00:00+00:00')
    month_end = month_start + pd.offsets.MonthBegin(1)
    month_frame = replay_frame.loc[
        (pd.to_datetime(replay_frame['timestamp'], utc=True) >= month_start)
        & (pd.to_datetime(replay_frame['timestamp'], utc=True) < month_end)
    ].copy()
    if month_frame.empty:
        raise SystemExit(f'No V13 replay candidates for month {args.month}. Available months: {available_months}')

    raw_bars = load_default_raw_bars(start=month_start - pd.Timedelta(minutes=240), end=month_end + pd.Timedelta(days=1))
    month_frame = attach_execution_prices(month_frame, raw_bars)
    executed, skipped = generate_v13_decisions(month_frame, threshold_selector=selector_wrapper, deployable_regimes=deployable_regimes)

    cost_pips = _trade_cost_pips(
        spread_pips=float(args.spread_pips),
        slippage_pips=float(args.slippage_pips),
        commission_usd=float(args.commission_usd),
        contract_size_oz=float(args.contract_size_oz),
        pip_size=float(args.pip_size),
    )
    executed = _compute_preplanned_lots(executed, capital=float(args.capital), contract_size_oz=float(args.contract_size_oz), pip_size=float(args.pip_size), cost_pips=cost_pips)

    plan_map = {}
    for row in executed.itertuples(index=False):
        direction_sign = int(np.sign(float(getattr(row, 'setl_trade_direction', getattr(row, 'branch_direction', 1.0))) or 1))
        decision_ts = pd.Timestamp(getattr(row, 'decision_ts')).tz_localize(None) if pd.Timestamp(getattr(row, 'decision_ts')).tzinfo is not None else pd.Timestamp(getattr(row, 'decision_ts'))
        exit_ts = pd.Timestamp(getattr(row, 'exit_ts')).tz_localize(None) if pd.Timestamp(getattr(row, 'exit_ts')).tzinfo is not None else pd.Timestamp(getattr(row, 'exit_ts'))
        plan_map[decision_ts] = {
            'sample_id': int(getattr(row, 'sample_id')),
            'decision_ts': decision_ts,
            'exit_signal_ts': exit_ts,
            'direction': direction_sign,
            'planned_lot': float(getattr(row, 'planned_lot', 0.01)),
            'confidence': float(getattr(row, 'calibrated_win_prob', 0.5)),
            'dominant_regime': str(getattr(row, 'regime_class', 'unknown')),
            'cabr_score': float(getattr(row, 'cabr_raw_score', 0.5)),
            'uts_score': float(getattr(row, 'uts_score', 0.0)),
        }

    data = V12SignalData(dataname=raw_bars.tz_convert(None))
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(float(args.capital))
    cerebro.broker.setcommission(commission=0.0, margin=0.0, mult=1.0, leverage=100.0, stocklike=False)
    cerebro.broker.set_slippage_perc(float(args.slippage_pips) * float(args.pip_size) / max(float(raw_bars['close'].iloc[-1]), 1e-6), slip_open=True, slip_limit=True, slip_match=True, slip_out=False)
    cerebro.adddata(data)
    cerebro.addstrategy(
        NexusV12Strategy,
        plan_map=plan_map,
        planned_skips=skipped,
        start_lot=float(args.initial_lot),
        max_lot=float(args.max_lot),
        start_equity=float(args.capital),
        end_equity=float(args.lot_growth_target_equity),
        contract_size_oz=float(args.contract_size_oz),
    )
    results = cerebro.run()
    strategy = results[0]

    trades = []
    equity = float(args.capital)
    peak = equity
    max_drawdown = 0.0
    for trade in sorted(strategy.trades_log, key=lambda item: pd.Timestamp(item.get('exit_dt', item.get('decision_ts')))):
        lot = float(trade.get('planned_lot', args.initial_lot))
        direction = int(np.sign(float(trade.get('direction', 1))) or 1)
        entry_price = float(trade.get('entry_fill_price', 0.0))
        exit_price = float(trade.get('exit_fill_price', entry_price))
        gross_pips = ((exit_price - entry_price) * direction) / max(float(args.pip_size), 1e-6)
        net_pips = gross_pips - cost_pips
        pip_value = float(args.contract_size_oz) * float(args.pip_size) * lot
        gross_pnl = gross_pips * pip_value
        net_pnl = net_pips * pip_value
        equity += net_pnl
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, 0.0 if peak <= 0 else (peak - equity) / peak)
        record = dict(trade)
        record['gross_pips'] = round(float(gross_pips), 6)
        record['net_pips'] = round(float(net_pips), 6)
        record['gross_pnl_usd'] = round(float(gross_pnl), 6)
        record['net_pnl_usd'] = round(float(net_pnl), 6)
        record['equity_after_trade'] = round(float(equity), 6)
        record['dominant_regime'] = str(record.get('dominant_regime', 'unknown'))
        trades.append(record)

    skip_breakdown: dict[str, int] = {}
    for item in skipped:
        reason = str(item.get('reason', 'unknown'))
        skip_breakdown[reason] = skip_breakdown.get(reason, 0) + 1

    paper_summary = PaperTradeAccumulator(V13_PAPER_TRADE_LOG_PATH).summary()
    lot_values = [float(item.get('planned_lot', 0.0)) for item in trades]
    net_pnls = np.asarray([item['net_pnl_usd'] for item in trades], dtype=np.float64) if trades else np.asarray([], dtype=np.float64)
    net_pips = np.asarray([item['net_pips'] for item in trades], dtype=np.float64) if trades else np.asarray([], dtype=np.float64)
    final_capital = float(args.capital) + float(np.sum(net_pnls))

    report = {
        'version': 'v13',
        'month': args.month,
        'start_capital': round(float(args.capital), 6),
        'final_capital': round(final_capital, 6),
        'net_profit': round(final_capital - float(args.capital), 6),
        'return_pct': round(((final_capital / float(args.capital)) - 1.0) * 100.0, 6),
        'trades_executed': int(len(trades)),
        'win_rate': round(float(np.mean(net_pnls > 0.0)) if len(trades) else 0.0, 6),
        'max_drawdown_pct': round(float(max_drawdown * 100.0), 6),
        'avg_win_pips': round(float(np.mean(net_pips[net_pips > 0.0])) if np.any(net_pips > 0.0) else 0.0, 6),
        'avg_loss_pips': round(float(np.mean(net_pips[net_pips < 0.0])) if np.any(net_pips < 0.0) else 0.0, 6),
        'profit_factor': None if not np.any(net_pnls < 0.0) else round(float(np.sum(net_pnls[net_pnls > 0.0]) / abs(np.sum(net_pnls[net_pnls < 0.0]))), 6),
        'skip_reason_breakdown': skip_breakdown,
        'stage_1_vs_stage_2_gap': round(float(sarv_report.get('stage_2', {}).get('gap_vs_stage1_winrate') or 0.0), 6),
        'paper_trade_summary': paper_summary,
        'deployable_regimes': sorted(deployable_regimes),
        'uts_thresholds': {k: round(float(v), 6) for k, v in thresholds.items()},
        'daps_lot_summary': {
            'min_lot': round(float(min(lot_values)) if lot_values else 0.0, 6),
            'max_lot': round(float(max(lot_values)) if lot_values else 0.0, 6),
            'avg_lot': round(float(np.mean(lot_values)) if lot_values else 0.0, 6),
        },
        'mbeg_veto_rate': round(float(skip_breakdown.get('minority_veto', 0) / max(len(executed) + len(skipped), 1)), 6),
        'lrtd_suppression_rate': round(float(skip_breakdown.get('lrtd_suppressed', 0) / max(len(executed) + len(skipped), 1)), 6),
        'regime_breakdown': _summarize_regimes(trades),
        'calibrator_summary': calibrator.summary(),
        'trade_log': trades,
        'skipped_trades': skipped,
    }
    out_path = (
        V13_BACKTRADER_MONTH_REPORT_PATH
        if args.month == '2023-12'
        else (PROJECT_ROOT / 'outputs' / 'v13' / f'backtrader_month_{args.month.replace("-", "_")}_v13.json')
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    print(str(out_path), flush=True)
    print(json.dumps({k: v for k, v in report.items() if k not in {'trade_log', 'skipped_trades'}}, indent=2), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description='Run a Backtrader month replay for Nexus Trader.')
    parser.add_argument('--month', default='2023-12')
    parser.add_argument('--version', default='v12')
    parser.add_argument('--capital', type=float, default=1000.0)
    parser.add_argument('--initial-lot', type=float, default=0.10)
    parser.add_argument('--max-lot', type=float, default=1.0)
    parser.add_argument('--lot-growth-target-equity', type=float, default=2500.0)
    parser.add_argument('--spread-pips', type=float, default=0.5)
    parser.add_argument('--commission-usd', type=float, default=7.0)
    parser.add_argument('--slippage-pips', type=float, default=0.2)
    parser.add_argument('--contract-size-oz', type=float, default=100.0)
    parser.add_argument('--pip-size', type=float, default=0.1)
    args = parser.parse_args()
    if str(args.version).strip().lower() != 'v13':
        raise SystemExit('This script currently supports --version v13 for the V13 prompt path.')
    return run_v13(args)


if __name__ == '__main__':
    raise SystemExit(main())



