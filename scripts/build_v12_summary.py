from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.project_config import V12_SUMMARY_JSON_PATH, V12_SUMMARY_MD_PATH


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def main() -> int:
    audit = _load(PROJECT_ROOT / 'outputs' / 'v12' / 'feature_consistency_report.json')
    tctl = _load(PROJECT_ROOT / 'outputs' / 'v12' / 'tctl_evaluation_report.json')
    sarv = _load(PROJECT_ROOT / 'outputs' / 'v12' / 'sarv_report.json')
    backtest = _load(PROJECT_ROOT / 'outputs' / 'v12' / 'backtrader_month_2023_12.json')

    pass_features = audit.get('legacy_archive_vs_live', {}).get('pass_features', [])
    fail_features = audit.get('legacy_archive_vs_live', {}).get('fail_features', [])
    stage2_gap = sarv.get('stage_2', {}).get('gap_vs_stage1_winrate')
    within_gap_target = bool(stage2_gap is not None and float(stage2_gap) <= 0.05)
    calibration_error = backtest.get('confidence_calibration_error_replay', backtest.get('confidence_calibration_error_valid'))
    deployable = backtest.get('wfri', {}).get('deployable_regimes', [])

    honest_lines = []
    if within_gap_target:
        honest_lines.append('BCFE plus TCTL appears to have materially reduced the archive-vs-replay gap.')
    else:
        honest_lines.append('The primary V12 success criterion is still not met: Stage 2 remains too far from Stage 1.')
    if float(tctl.get('pairwise_accuracy_valid', 0.0)) < 0.55:
        honest_lines.append('TCTL no longer collapses, but its held-out pairwise ranking accuracy is still weak and below the prompt target.')
    if backtest:
        if float(backtest.get('return_pct', 0.0)) > 0.0:
            honest_lines.append('The one-month Backtrader replay is profitable under the current cost model, which is a meaningful improvement over the V11 collapse.')
        else:
            honest_lines.append('The one-month Backtrader replay is still not profitable after realistic execution costs, so V12 remains incomplete as a trading system.')
    if float(calibration_error or 1.0) >= 0.05:
        honest_lines.append('Confidence calibration is still loose relative to the sub-5% V12 aspiration.')
    if len(deployable) < 2:
        honest_lines.append('WFRI does not yet show two clearly deployable regime classes, so regime-aware deployment remains conservative.')

    summary = {
        'feature_consistency': {
            'pass_count': int(len(pass_features)),
            'fail_count': int(len(fail_features)),
            'pass_features': pass_features,
            'fail_features': fail_features,
        },
        'tctl': tctl,
        'sarv': sarv,
        'backtrader_month': backtest,
        'primary_success': {
            'stage2_gap_within_5pp': within_gap_target,
            'stage2_gap': stage2_gap,
        },
        'confidence_calibration_error': calibration_error,
        'deployable_regimes': deployable,
        'honest_interpretation': honest_lines,
        'v13_recommendation': 'Keep BCFE as the canonical feature path, improve TCTL pair construction and score calibration, and only expand live deployment after a passing SARV Stage 3 paper-trade window.',
    }

    V12_SUMMARY_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    V12_SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    md = [
        '# V12 Summary',
        '',
        f"- Feature audit: {len(pass_features)} pass / {len(fail_features)} fail",
        f"- TCTL pairwise accuracy (valid): {tctl.get('pairwise_accuracy_valid', 'n/a')}",
        f"- SARV Stage 1 win rate: {sarv.get('stage_1', {}).get('win_rate', 'n/a')}",
        f"- SARV Stage 2 win rate: {sarv.get('stage_2', {}).get('win_rate', 'n/a')}",
        f"- Stage 2 gap: {stage2_gap}",
        f"- Backtrader month: {backtest.get('month', 'n/a')}",
        f"- Backtrader return: {backtest.get('return_pct', 'n/a')}%",
        f"- Backtrader win rate: {backtest.get('win_rate', 'n/a')}",
        f"- Calibration error: {calibration_error}",
        f"- Deployable regimes: {', '.join(deployable) if deployable else 'none'}",
        '',
        '## Honest Interpretation',
        '',
    ]
    md.extend([f'- {line}' for line in honest_lines])
    md.extend([
        '',
        '## V13 Recommendation',
        '',
        '- Keep BCFE as the canonical feature path.',
        '- Improve TCTL pair construction and calibration before claiming live readiness.',
        '- Require a passing SARV Stage 3 paper-trade window before live deployment.',
    ])
    V12_SUMMARY_MD_PATH.write_text('\n'.join(md) + '\n', encoding='utf-8')
    print(str(V12_SUMMARY_JSON_PATH), flush=True)
    print(str(V12_SUMMARY_MD_PATH), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
