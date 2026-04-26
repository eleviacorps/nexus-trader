from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config.project_config import V12_TCTL_EVALUATION_REPORT_PATH
from src.v12.tctl import (
    build_training_pairs,
    evaluate_pairwise_accuracy,
    optimize_tctl_threshold,
    prepare_tctl_candidates,
    score_tctl_model,
    train_tctl_model,
)


def main() -> int:
    frame = pd.read_parquet(PROJECT_ROOT / 'outputs' / 'v10' / 'branch_features_v10_full.parquet')
    train_candidates, valid_candidates, feature_names = prepare_tctl_candidates(frame, validation_fraction=0.2)
    training = train_tctl_model(train_candidates, feature_names=feature_names)
    model = training['model']
    valid_pairs = build_training_pairs(valid_candidates, feature_names=feature_names)
    pairwise_accuracy = evaluate_pairwise_accuracy(model, valid_candidates, valid_pairs, feature_names=feature_names, device=training['device'])
    train_scores = score_tctl_model(model, train_candidates, feature_names=feature_names, device=training['device'])
    threshold = optimize_tctl_threshold(train_scores, train_candidates['setl_target_net_unit_pnl'].to_numpy(dtype='float32'))
    valid_scores = score_tctl_model(model, valid_candidates, feature_names=feature_names, device=training['device'])

    report = {
        'feature_count': len(feature_names),
        'train_candidate_count': int(len(train_candidates)),
        'valid_candidate_count': int(len(valid_candidates)),
        'pair_count_train': int(training['pair_count']),
        'pair_count_valid': int(len(valid_pairs)),
        'pairwise_accuracy_valid': round(float(pairwise_accuracy), 6),
        'threshold': {
            'threshold': round(float(threshold.threshold), 6),
            'participation_rate': round(float(threshold.participation_rate), 6),
            'avg_unit_pnl': round(float(threshold.avg_unit_pnl), 6),
        },
        'score_summary_valid': {
            'min': round(float(valid_scores.min()), 6),
            'max': round(float(valid_scores.max()), 6),
            'mean': round(float(valid_scores.mean()), 6),
            'distinct_rounded_scores': int(len(set(valid_scores.round(4).tolist()))),
        },
        'loss_history': [round(float(value), 6) for value in training['loss_history']],
        'device': training['device'],
    }
    V12_TCTL_EVALUATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    V12_TCTL_EVALUATION_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(str(V12_TCTL_EVALUATION_REPORT_PATH), flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
