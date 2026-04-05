from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config.project_config import V14_BACKTRADER_WALKFORWARD_REPORT_PATH, V14_BRANCH_FEATURES_PATH, V14_PAPER_TRADE_LOG_PATH, V14_RSC_CALIBRATOR_PATH
from src.v12.tctl import replay_candidates_with_online_bcfe
from src.v13.cabr import augment_cabr_context, load_v13_candidate_frames
from src.v13.s3pta import PaperTradeAccumulator
from src.v14.rsc import RegimeStratifiedCalibrator


def _available_months() -> list[str]:
    archive = pd.read_parquet(V14_BRANCH_FEATURES_PATH)
    _, valid_frame, _, _ = load_v13_candidate_frames(archive, use_temporal_context=True, n_context_bars=12)
    replay = augment_cabr_context(replay_candidates_with_online_bcfe(valid_frame))
    ts = pd.to_datetime(replay['timestamp'], utc=True, errors='coerce')
    return sorted(pd.Series(ts.dt.to_period('M').astype(str)).dropna().unique().tolist())


def _report_path_for_month(month: str) -> Path:
    if month == '2023-12':
        return PROJECT_ROOT / 'outputs' / 'v14' / 'backtrader_month_2023_12_v14.json'
    return PROJECT_ROOT / 'outputs' / 'v14' / f'backtrader_month_{month.replace("-", "_")}_v14.json'


def _month_pass(report: dict) -> bool:
    trades = int(report.get('trades_executed', 0))
    win_rate = float(report.get('win_rate', 0.0))
    profit_factor = float(report.get('profit_factor') or 0.0)
    max_dd = float(report.get('max_drawdown_pct', 0.0))
    gap = float(report.get('stage_1_vs_stage_2_gap', 1.0))
    return 15 <= trades <= 60 and win_rate > 0.56 and profit_factor > 2.0 and max_dd < 12.0 and gap < 0.05


def _failure_reasons(report: dict) -> list[str]:
    reasons = []
    trades = int(report.get('trades_executed', 0))
    if not (15 <= trades <= 60):
        reasons.append(f'trades={trades}')
    if not (float(report.get('win_rate', 0.0)) > 0.56):
        reasons.append(f"win_rate={float(report.get('win_rate', 0.0)):.6f}")
    if not (float(report.get('profit_factor') or 0.0) > 2.0):
        reasons.append(f"profit_factor={float(report.get('profit_factor') or 0.0):.6f}")
    if not (float(report.get('max_drawdown_pct', 0.0)) < 12.0):
        reasons.append(f"max_dd={float(report.get('max_drawdown_pct', 0.0)):.6f}")
    if not (float(report.get('stage_1_vs_stage_2_gap', 1.0)) < 0.05):
        reasons.append(f"stage_gap={float(report.get('stage_1_vs_stage_2_gap', 1.0)):.6f}")
    return reasons


def main() -> int:
    if V14_PAPER_TRADE_LOG_PATH.exists():
        V14_PAPER_TRADE_LOG_PATH.unlink()
    if V14_RSC_CALIBRATOR_PATH.exists():
        V14_RSC_CALIBRATOR_PATH.unlink()

    script = PROJECT_ROOT / 'scripts' / 'run_v12_backtrader_month.py'
    months = _available_months()
    reports = []
    for idx, month in enumerate(months):
        mode = 'reset' if idx == 0 else 'append'
        completed = subprocess.run(
            [sys.executable, str(script), '--month', str(month), '--version', 'v14', '--paper-mode', mode],
            cwd=str(PROJECT_ROOT),
            check=False,
        )
        if completed.returncode != 0:
            return int(completed.returncode)
        report_path = _report_path_for_month(month)
        payload = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
        payload['month'] = month
        reports.append(payload)

    valid_reports = [report for report in reports if report]
    aggregate_trades = sum(int(report.get('trades_executed', 0)) for report in valid_reports)
    aggregate_wins = sum(int(round(float(report.get('win_rate', 0.0)) * float(report.get('trades_executed', 0)))) for report in valid_reports)
    aggregate_return = sum(float(report.get('return_pct', 0.0)) for report in valid_reports)
    max_single_month_drawdown = max((float(report.get('max_drawdown_pct', 0.0)) for report in valid_reports), default=0.0)
    profitable_months = sum(1 for report in valid_reports if float(report.get('net_profit', 0.0)) > 0.0)
    objective_pass_months = sum(1 for report in valid_reports if _month_pass(report))
    aggregate_skip_candidates = sum(int(report.get('trades_executed', 0)) + sum(int(v) for v in report.get('skip_reason_breakdown', {}).values()) for report in valid_reports)
    all_trade_pnls = np.asarray(
        [
            float(item.get('net_pnl_usd', 0.0))
            for report in valid_reports
            for item in report.get('trade_log', [])
        ],
        dtype=np.float64,
    )
    profits = all_trade_pnls[all_trade_pnls > 0.0].sum() if all_trade_pnls.size else 0.0
    losses = np.abs(all_trade_pnls[all_trade_pnls < 0.0].sum()) if all_trade_pnls.size else 0.0
    aggregate_profit_factor = None if losses <= 0.0 else round(float(profits / losses), 6)
    paper_summary = PaperTradeAccumulator(V14_PAPER_TRADE_LOG_PATH).summary()
    rsc_summary = RegimeStratifiedCalibrator.load(V14_RSC_CALIBRATOR_PATH).summary()
    final_report = {
        'version': 'v14',
        'month_count': int(len(valid_reports)),
        'months': valid_reports,
        'aggregate_trades': int(aggregate_trades),
        'aggregate_win_rate': round(float(aggregate_wins / max(aggregate_trades, 1)), 6),
        'aggregate_return_pct_sum': round(float(aggregate_return), 6),
        'aggregate_profit_factor': aggregate_profit_factor,
        'profitable_months': int(profitable_months),
        'objective_pass_months': int(objective_pass_months),
        'max_single_month_drawdown_pct': round(float(max_single_month_drawdown), 6),
        'avg_stage_1_vs_stage_2_gap': round(float(np.mean([float(report.get('stage_1_vs_stage_2_gap', 0.0)) for report in valid_reports])) if valid_reports else 0.0, 6),
        'avg_ssc_rejection_rate': round(float(np.mean([float(report.get('ssc_rejection_rate', 0.0)) for report in valid_reports])) if valid_reports else 0.0, 6),
        'avg_bst_survival_score': round(float(np.mean([float(report.get('avg_bst_survival_score', 0.0)) for report in valid_reports])) if valid_reports else 0.0, 6),
        'bst_score_distribution': {
            'mean': round(float(np.mean([float(report.get('bst_score_summary', {}).get('mean', 0.0)) for report in valid_reports])) if valid_reports else 0.0, 6),
            'min': round(float(min((float(report.get('bst_score_summary', {}).get('min', 0.0)) for report in valid_reports), default=0.0)), 6),
            'max': round(float(max((float(report.get('bst_score_summary', {}).get('max', 0.0)) for report in valid_reports), default=0.0)), 6),
        },
        'fear_index_by_month': {report.get('month', 'unknown'): float(report.get('avg_fear_index_retail', 0.0)) for report in valid_reports},
        'paper_trade_summary': paper_summary,
        'rsc_summary': rsc_summary,
        'failed_objective_months': [
            {'month': report.get('month', 'unknown'), 'reasons': _failure_reasons(report)}
            for report in valid_reports
            if not _month_pass(report)
        ],
    }
    V14_BACKTRADER_WALKFORWARD_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    V14_BACKTRADER_WALKFORWARD_REPORT_PATH.write_text(json.dumps(final_report, indent=2), encoding='utf-8')
    print(str(V14_BACKTRADER_WALKFORWARD_REPORT_PATH), flush=True)
    print(json.dumps({k: v for k, v in final_report.items() if k != 'months'}, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
