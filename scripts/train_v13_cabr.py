from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config.project_config import V13_CABR_EVALUATION_REPORT_PATH, V13_CABR_MODEL_PATH, V14_BRANCH_FEATURES_PATH, V14_CABR_EVALUATION_REPORT_PATH, V14_CABR_TEMPORAL_MODEL_PATH
from src.v13.cabr import evaluate_cabr_pairwise_accuracy, load_v13_candidate_frames, score_cabr_model, train_cabr_model


def main() -> int:
    parser = argparse.ArgumentParser(description='Train CABR for V13 or V14.')
    parser.add_argument('--version', default='v13')
    args = parser.parse_args()
    version = str(args.version).strip().lower()
    is_v14 = version == 'v14'

    archive_path = V14_BRANCH_FEATURES_PATH if is_v14 else (PROJECT_ROOT / 'outputs' / 'v10' / 'branch_features_v10_full.parquet')
    archive = pd.read_parquet(archive_path)
    train_frame, valid_frame, branch_cols, context_cols = load_v13_candidate_frames(
        archive,
        use_temporal_context=is_v14,
        n_context_bars=12,
    )
    checkpoint_path = V14_CABR_TEMPORAL_MODEL_PATH if is_v14 else V13_CABR_MODEL_PATH
    training = train_cabr_model(
        train_frame,
        valid_frame,
        branch_feature_names=branch_cols,
        context_feature_names=context_cols,
        use_temporal_context=is_v14,
        n_context_bars=12,
        checkpoint_path=checkpoint_path,
    )
    evaluation = evaluate_cabr_pairwise_accuracy(training['model'], training['valid_pairs'], device=training['device'])
    valid_scores = score_cabr_model(
        training['model'],
        valid_frame,
        branch_feature_names=branch_cols,
        context_feature_names=context_cols,
        device=training['device'],
    )
    baseline_path = PROJECT_ROOT / 'outputs' / 'v13' / 'tctl_regime_fixed_report.json'
    baseline = json.loads(baseline_path.read_text(encoding='utf-8')) if baseline_path.exists() else {}
    report = {
        'version': version,
        'branch_feature_count': len(branch_cols),
        'context_feature_count': len(context_cols),
        'train_pair_count': int(len(training['train_pairs']['pairs'])),
        'valid_pair_count': int(len(training['valid_pairs']['pairs'])),
        'heldout_pairwise_accuracy_overall': round(float(evaluation['overall_accuracy']), 6),
        'heldout_pairwise_accuracy_per_regime': evaluation['per_regime_accuracy'],
        'score_diversity': {
            'distinct_rounded_values_valid': int(len(set(np.round(valid_scores, 4).tolist()))),
            'min': round(float(valid_scores.min()), 6),
            'max': round(float(valid_scores.max()), 6),
            'mean': round(float(valid_scores.mean()), 6),
        },
        'comparison_to_v12_tctl': {
            'v12_pairwise_accuracy_valid': baseline.get('pairwise_accuracy_valid'),
            'delta_vs_v12': None if baseline.get('pairwise_accuracy_valid') is None else round(float(evaluation['overall_accuracy']) - float(baseline['pairwise_accuracy_valid']), 6),
        },
        'branch_feature_names': list(branch_cols),
        'context_feature_names': list(context_cols),
        'loss_history': [round(float(v), 6) for v in training['loss_history']],
        'valid_accuracy_history': [round(float(v), 6) for v in training['valid_accuracy_history']],
        'device': training['device'],
        'use_temporal_context': is_v14,
        'checkpoint_path': str(checkpoint_path),
    }
    report_path = V14_CABR_EVALUATION_REPORT_PATH if is_v14 else V13_CABR_EVALUATION_REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(str(report_path), flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
