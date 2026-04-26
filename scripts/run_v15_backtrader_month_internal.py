from __future__ import annotations

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

from config.project_config import (
    V14_BRANCH_FEATURES_PATH,
    V14_CABR_TEMPORAL_MODEL_PATH,
    V15_BACKTRADER_MONTH_REPORT_PATH,
    V15_ECI_CALENDAR_PATH,
    V15_PAPER_TRADE_LOG_PATH,
    V15_RSC_BOOTSTRAPPED_PATH,
    V15_RSC_RUNTIME_PATH,
)
from src.v12.backtrader_strategy import NexusV12Strategy, V12SignalData
from src.v12.bar_consistent_features import load_default_raw_bars
from src.v12.sarv import run_scored_sarv_validation
from src.v12.tctl import replay_candidates_with_online_bcfe
from src.v13.cabr import augment_cabr_context, load_cabr_model, load_v13_candidate_frames, score_cabr_model
from src.v13.daps import daps_lot_size
from src.v13.s3pta import PaperTradeAccumulator
from src.v14.rsc import RegimeStratifiedCalibrator
from src.v15.cpm import ConditionalPredictabilityMapper
from src.v15.eci import EconomicCalendarIntegration
from src.v15.pce import PredictabilityConditionedExecution
from src.v15.policy_utils import attach_execution_prices, enrich_v15_policy_frame, generate_v15_decisions


def _available_months(frame: pd.DataFrame) -> list[str]:
    ts = pd.to_datetime(frame['timestamp'], utc=True, errors='coerce')
    return sorted(pd.Series(ts.dt.to_period('M').astype(str)).dropna().unique().tolist())


def _trade_cost_pips(*, spread_pips: float, slippage_pips: float, commission_usd: float, contract_size_oz: float, pip_size: float) -> float:
    commission_pips = float(commission_usd) / max(float(contract_size_oz) * float(pip_size), 1e-6)
    return float(spread_pips + slippage_pips + commission_pips)


def _v15_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    archive = pd.read_parquet(V14_BRANCH_FEATURES_PATH)
    train_frame, valid_frame, _, _ = load_v13_candidate_frames(
        archive,
        use_temporal_context=True,
        n_context_bars=12,
    )
    replay_frame = augment_cabr_context(replay_candidates_with_online_bcfe(valid_frame))
    return train_frame, valid_frame, replay_frame


def _load_v15_calibrator() -> RegimeStratifiedCalibrator:
    for path in (V15_RSC_RUNTIME_PATH, V15_RSC_BOOTSTRAPPED_PATH):
        if path.exists():
            return RegimeStratifiedCalibrator.load(path)
    return RegimeStratifiedCalibrator()


def _score_v15_frames(
    *,
    pce_target_rate: float,
    eci: EconomicCalendarIntegration,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, RegimeStratifiedCalibrator, PredictabilityConditionedExecution]:
    train_frame, valid_frame, replay_frame = _v15_frames()
    model, branch_cols, context_cols, _ = load_cabr_model(V14_CABR_TEMPORAL_MODEL_PATH, map_location='cpu')

    train_frame = train_frame.copy()
    valid_frame = valid_frame.copy()
    replay_frame = replay_frame.copy()
    train_frame['cabr_score'] = score_cabr_model(model, train_frame, branch_feature_names=branch_cols, context_feature_names=context_cols, device='cpu')
    valid_frame['cabr_score'] = score_cabr_model(model, valid_frame, branch_feature_names=branch_cols, context_feature_names=context_cols, device='cpu')
    replay_frame['cabr_score'] = score_cabr_model(model, replay_frame, branch_feature_names=branch_cols, context_feature_names=context_cols, device='cpu')

    calibrator = _load_v15_calibrator()
    mapper = ConditionalPredictabilityMapper()
    train_enriched = enrich_v15_policy_frame(train_frame, cabr_score_column='cabr_score', calibrator=calibrator, cpm_mapper=mapper, eci=eci)
    valid_enriched = enrich_v15_policy_frame(valid_frame, cabr_score_column='cabr_score', calibrator=calibrator, cpm_mapper=mapper, eci=eci)
    replay_enriched = enrich_v15_policy_frame(replay_frame, cabr_score_column='cabr_score', calibrator=calibrator, cpm_mapper=mapper, eci=eci)

    pce_seed = PredictabilityConditionedExecution()
    tuned_threshold = pce_seed.tune_threshold_for_participation(
        train_enriched['cpm_live_score'].to_numpy(dtype=np.float32),
        target_rate=float(pce_target_rate),
    )
    pce = PredictabilityConditionedExecution(cpm_threshold=tuned_threshold)
    return train_enriched, valid_enriched, replay_enriched, calibrator, pce


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


def _compute_preplanned_lots_v15(
    frame: pd.DataFrame,
    *,
    capital: float,
    contract_size_oz: float,
    pip_size: float,
    cost_pips: float,
    max_account_leverage: float | None,
    max_lot: float,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    working = frame.sort_values('decision_ts').copy().reset_index(drop=True)
    equity = float(capital)
    recent_outcomes: list[int] = []
    planned_rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []

    for row in working.to_dict(orient='records'):
        recent_win_rate = 0.55 if not recent_outcomes else float(np.mean(recent_outcomes[-20:]))
        lot = daps_lot_size(
            base_capital=float(capital),
            current_equity=float(equity),
            recent_win_rate=float(recent_win_rate),
            regime=str(row.get('regime_class', 'unknown')),
            uts_score=float(row.get('uts_score', 0.5) or 0.5),
            min_lot=0.05,
            max_lot=float(max_lot),
            max_account_leverage=max_account_leverage,
            price_per_ounce=float(row.get('entry_price', 0.0) or 0.0),
            contract_size_oz=float(contract_size_oz),
        )
        if float(lot) < 0.05:
            skipped.append(
                {
                    'sample_id': int(row.get('sample_id', -1)),
                    'decision_ts': str(row.get('decision_ts')),
                    'reason': 'daps_minimum_lot',
                    'regime': str(row.get('regime_class', 'unknown')),
                    'uts_score': float(row.get('uts_score', 0.0) or 0.0),
                    'cabr_score': float(row.get('cabr_raw_score', 0.0) or 0.0),
                }
            )
            continue

        row['planned_lot'] = float(lot)
        planned_rows.append(row)
        direction = 1.0 if int(np.sign(float(row.get('setl_trade_direction', row.get('branch_direction', 1.0))) or 1)) >= 0 else -1.0
        gross_pips = ((float(row.get('exit_price', row.get('entry_price', 0.0))) - float(row.get('entry_price', 0.0))) * direction) / max(float(pip_size), 1e-6)
        net_pips = gross_pips - float(cost_pips)
        pip_value = float(contract_size_oz) * float(pip_size) * float(lot)
        net_pnl = net_pips * pip_value
        equity += net_pnl
        recent_outcomes.append(1 if net_pnl > 0.0 else 0)

    if not planned_rows:
        empty = working.iloc[0:0].copy()
        empty['planned_lot'] = pd.Series(dtype=float)
        return empty, skipped
    return pd.DataFrame(planned_rows), skipped


def run_v15(args) -> int:
    paper_mode = str(getattr(args, 'paper_mode', 'append')).strip().lower()
    if paper_mode == 'reset':
        if V15_PAPER_TRADE_LOG_PATH.exists():
            V15_PAPER_TRADE_LOG_PATH.unlink()
        if V15_RSC_RUNTIME_PATH.exists():
            V15_RSC_RUNTIME_PATH.unlink()

    calendar = EconomicCalendarIntegration.from_csv(V15_ECI_CALENDAR_PATH)
    train_frame, valid_frame, replay_frame, calibrator, pce = _score_v15_frames(
        pce_target_rate=float(getattr(args, 'pce_target_rate', 0.20)),
        eci=calendar,
    )

    sarv_report = run_scored_sarv_validation(
        model_name='cabr_v15',
        train_scores=train_frame['uts_score'].to_numpy(dtype=np.float32),
        train_outcomes=train_frame['setl_target_net_unit_pnl'].to_numpy(dtype=np.float32),
        archive_candidates=valid_frame.assign(uts_score=valid_frame['uts_score']),
        archive_score_column='uts_score',
        bar_replay_candidates=replay_frame.assign(uts_score=replay_frame['uts_score']),
        replay_score_column='uts_score',
        paper_trade_log_path=V15_PAPER_TRADE_LOG_PATH,
    )

    available_months = _available_months(replay_frame)
    month_start = pd.Timestamp(f'{args.month}-01 00:00:00+00:00')
    month_end = month_start + pd.offsets.MonthBegin(1)
    month_frame = replay_frame.loc[
        (pd.to_datetime(replay_frame['timestamp'], utc=True) >= month_start)
        & (pd.to_datetime(replay_frame['timestamp'], utc=True) < month_end)
    ].copy()
    if month_frame.empty:
        raise SystemExit(f'No V15 replay candidates for month {args.month}. Available months: {available_months}')

    raw_bars = load_default_raw_bars(start=month_start - pd.Timedelta(minutes=240), end=month_end + pd.Timedelta(days=1))
    month_frame = attach_execution_prices(month_frame, raw_bars)
    executed, skipped = generate_v15_decisions(
        month_frame,
        pce=pce,
        cabr_minimum=float(getattr(args, 'pce_cabr_minimum', 0.52)),
    )

    cost_pips = _trade_cost_pips(
        spread_pips=float(args.spread_pips),
        slippage_pips=float(args.slippage_pips),
        commission_usd=float(args.commission_usd),
        contract_size_oz=float(args.contract_size_oz),
        pip_size=float(args.pip_size),
    )
    executed, daps_skipped = _compute_preplanned_lots_v15(
        executed,
        capital=float(args.capital),
        contract_size_oz=float(args.contract_size_oz),
        pip_size=float(args.pip_size),
        cost_pips=cost_pips,
        max_account_leverage=float(args.max_account_leverage) if getattr(args, 'max_account_leverage', None) is not None else None,
        max_lot=float(args.max_lot),
    )
    skipped.extend(daps_skipped)

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
            'planned_lot': float(getattr(row, 'planned_lot', 0.05)),
            'confidence': float(getattr(row, 'calibrated_win_prob', 0.5)),
            'dominant_regime': str(getattr(row, 'regime_class', 'unknown')),
            'cabr_score': float(getattr(row, 'cabr_raw_score', 0.5)),
            'uts_score': float(getattr(row, 'uts_score', 0.0)),
            'cpm_live_score': float(getattr(row, 'cpm_live_score', 0.5)),
            'adjusted_cpm_score': float(getattr(row, 'adjusted_cpm_score', 0.5)),
            'cpm_agreement': float(getattr(row, 'cpm_agreement', 0.0)),
            'eci_reaction_window': bool(getattr(row, 'eci_reaction_window', False)),
            'eci_post_settling': bool(getattr(row, 'eci_post_settling', False)),
        }

    data = V12SignalData(dataname=raw_bars.tz_convert(None))
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(float(args.capital))
    broker_leverage = 1.0 if getattr(args, 'broker_leverage', None) is None else float(args.broker_leverage)
    cerebro.broker.setcommission(commission=0.0, margin=0.0, mult=1.0, leverage=broker_leverage, stocklike=False)
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
        max_account_leverage=float(args.max_account_leverage) if getattr(args, 'max_account_leverage', None) is not None else None,
    )
    results = cerebro.run()
    strategy = results[0]

    trades = []
    equity = float(args.capital)
    peak = equity
    max_drawdown = 0.0
    accumulator = PaperTradeAccumulator(V15_PAPER_TRADE_LOG_PATH)
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
        if paper_mode != 'none':
            completed = accumulator.log_completed_trade(
                symbol='XAUUSD',
                direction='BUY' if direction > 0 else 'SELL',
                uts_score=float(record.get('uts_score', 0.0) or 0.0),
                cabr_score=float(record.get('cabr_score', 0.5) or 0.5),
                regime=str(record.get('dominant_regime', 'unknown')),
                entry_price=entry_price,
                exit_price=exit_price,
                entry_time=str(record.get('decision_ts')),
                exit_time=str(record.get('exit_dt')),
            )
            calibrator.record_outcome(float(record.get('cabr_score', 0.5) or 0.5), str(record.get('dominant_regime', 'unknown')), completed.get('outcome') == 'win')
    if paper_mode != 'none':
        calibrator.save(V15_RSC_RUNTIME_PATH)

    skip_breakdown: dict[str, int] = {}
    for item in skipped:
        reason = str(item.get('reason', 'unknown'))
        skip_breakdown[reason] = skip_breakdown.get(reason, 0) + 1

    paper_summary = PaperTradeAccumulator(V15_PAPER_TRADE_LOG_PATH).summary()
    lot_values = [float(item.get('planned_lot', 0.0)) for item in trades]
    net_pnls = np.asarray([item['net_pnl_usd'] for item in trades], dtype=np.float64) if trades else np.asarray([], dtype=np.float64)
    net_pips = np.asarray([item['net_pips'] for item in trades], dtype=np.float64) if trades else np.asarray([], dtype=np.float64)
    final_capital = float(args.capital) + float(np.sum(net_pnls))
    total_candidates = max(len(trades) + len(skipped), 1)
    pce_skipped = sum(1 for item in skipped if str(item.get('reason')) == 'pce_not_predictable')

    report = {
        'version': 'v15',
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
        'pce_threshold': round(float(pce.cpm_threshold), 6),
        'pce_target_rate': round(float(getattr(args, 'pce_target_rate', 0.20)), 6),
        'pce_rejection_rate': round(float(pce_skipped / total_candidates), 6),
        'cpm_month_summary': {
            'mean_live_score': round(float(month_frame.get('cpm_live_score', pd.Series([0.5])).mean()), 6),
            'mean_adjusted_score': round(float(month_frame.get('adjusted_cpm_score', pd.Series([0.5])).mean()), 6),
            'mean_agreement': round(float(month_frame.get('cpm_agreement', pd.Series([0.0])).mean()), 6),
        },
        'daps_lot_summary': {
            'min_lot': round(float(min(lot_values)) if lot_values else 0.0, 6),
            'max_lot': round(float(max(lot_values)) if lot_values else 0.0, 6),
            'avg_lot': round(float(np.mean(lot_values)) if lot_values else 0.0, 6),
        },
        'leverage_settings': {
            'broker_leverage': float(broker_leverage),
            'max_account_leverage': None if getattr(args, 'max_account_leverage', None) is None else float(args.max_account_leverage),
        },
        'regime_breakdown': _summarize_regimes(trades),
        'rsc_summary': calibrator.summary(),
        'trade_log': trades,
        'skipped_trades': skipped,
    }
    out_path = (
        V15_BACKTRADER_MONTH_REPORT_PATH
        if args.month == '2023-12'
        else (PROJECT_ROOT / 'outputs' / 'v15' / f'backtrader_month_{args.month.replace("-", "_")}_v15.json')
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    print(str(out_path), flush=True)
    print(json.dumps({k: v for k, v in report.items() if k not in {'trade_log', 'skipped_trades'}}, indent=2), flush=True)
    return 0
