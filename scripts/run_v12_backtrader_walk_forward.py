from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config.project_config import V13_BACKTRADER_WALKFORWARD_REPORT_PATH
from src.v12.tctl import replay_candidates_with_online_bcfe
from src.v13.cabr import augment_cabr_context, load_v13_candidate_frames


def _available_months() -> list[str]:
    archive = pd.read_parquet(PROJECT_ROOT / 'outputs' / 'v10' / 'branch_features_v10_full.parquet')
    _, valid_frame, _, _ = load_v13_candidate_frames(archive)
    replay = augment_cabr_context(replay_candidates_with_online_bcfe(valid_frame))
    ts = pd.to_datetime(replay['timestamp'], utc=True, errors='coerce')
    months = sorted(pd.Series(ts.dt.to_period('M').astype(str)).dropna().unique().tolist())
    return months


def _report_path_for_month(month: str) -> Path:
    if month == '2023-12':
        return PROJECT_ROOT / 'outputs' / 'v13' / 'backtrader_month_2023_12_v13.json'
    return PROJECT_ROOT / 'outputs' / 'v13' / f'backtrader_month_{month.replace("-", "_")}_v13.json'


def main() -> int:
    parser = argparse.ArgumentParser(description='Run V13 month replay across all available replay months.')
    parser.add_argument('--version', default='v13')
    args = parser.parse_args()
    if str(args.version).strip().lower() != 'v13':
        raise SystemExit('This script currently supports --version v13 only.')

    script = PROJECT_ROOT / 'scripts' / 'run_v12_backtrader_month.py'
    months = _available_months()
    reports = []
    for month in months:
        report_path = _report_path_for_month(month)
        if not report_path.exists():
            completed = subprocess.run([sys.executable, str(script), '--month', str(month), '--version', 'v13'], cwd=str(PROJECT_ROOT), check=False)
            if completed.returncode != 0:
                return int(completed.returncode)
        payload = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
        payload['month'] = month
        reports.append(payload)

    valid_reports = [report for report in reports if report]
    aggregate_trades = sum(int(report.get('trades_executed', 0)) for report in valid_reports)
    aggregate_wins = sum(int(round(float(report.get('win_rate', 0.0)) * float(report.get('trades_executed', 0)))) for report in valid_reports)
    aggregate_return = sum(float(report.get('return_pct', 0.0)) for report in valid_reports)
    max_single_month_drawdown = max((float(report.get('max_drawdown_pct', 0.0)) for report in valid_reports), default=0.0)
    profitable_months = sum(1 for report in valid_reports if float(report.get('net_profit', 0.0)) > 0.0)
    objective_pass_months = sum(
        1
        for report in valid_reports
        if int(report.get('trades_executed', 0)) >= 15
        and int(report.get('trades_executed', 0)) <= 50
        and float(report.get('win_rate', 0.0)) > 0.54
        and float(report.get('max_drawdown_pct', 0.0)) < 15.0
        and float(report.get('stage_1_vs_stage_2_gap', 1.0)) < 0.05
    )
    wfri_breakdown = {report.get('month', 'unknown'): report.get('deployable_regimes', []) for report in valid_reports}
    final_report = {
        'version': 'v13',
        'month_count': int(len(valid_reports)),
        'months': valid_reports,
        'aggregate_trades': int(aggregate_trades),
        'aggregate_win_rate': round(float(aggregate_wins / max(aggregate_trades, 1)), 6),
        'aggregate_return_pct_sum': round(float(aggregate_return), 6),
        'profitable_months': int(profitable_months),
        'objective_pass_months': int(objective_pass_months),
        'max_single_month_drawdown_pct': round(float(max_single_month_drawdown), 6),
        'wfri_regime_breakdown': wfri_breakdown,
    }
    V13_BACKTRADER_WALKFORWARD_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    V13_BACKTRADER_WALKFORWARD_REPORT_PATH.write_text(json.dumps(final_report, indent=2), encoding='utf-8')
    print(str(V13_BACKTRADER_WALKFORWARD_REPORT_PATH), flush=True)
    print(json.dumps({k: v for k, v in final_report.items() if k != 'months'}, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
