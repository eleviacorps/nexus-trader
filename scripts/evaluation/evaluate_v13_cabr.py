from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config.project_config import V13_CABR_EVALUATION_REPORT_PATH
from src.v13.cabr import (
    evaluate_cabr_pairwise_accuracy,
    load_cabr_model,
    load_v13_candidate_frames,
    score_cabr_model,
)


def main() -> int:
    archive = pd.read_parquet(PROJECT_ROOT / 'outputs' / 'v10' / 'branch_features_v10_full.parquet')
    _, valid_frame, _, _ = load_v13_candidate_frames(archive)
    model, branch_cols, context_cols, payload = load_cabr_model(map_location='cpu')
    from src.v13.cabr import build_cabr_pairs
    valid_pairs = build_cabr_pairs(valid_frame, branch_feature_names=branch_cols, context_feature_names=context_cols)
    evaluation = evaluate_cabr_pairwise_accuracy(model, valid_pairs, device='cpu')
    valid_scores = score_cabr_model(model, valid_frame, branch_feature_names=branch_cols, context_feature_names=context_cols, device='cpu')
    report = {
        'heldout_pairwise_accuracy_overall': round(float(evaluation['overall_accuracy']), 6),
        'heldout_pairwise_accuracy_per_regime': evaluation['per_regime_accuracy'],
        'score_diversity': {
            'distinct_rounded_values_valid': int(len(set(np.round(valid_scores, 4).tolist()))),
            'min': round(float(valid_scores.min()), 6),
            'max': round(float(valid_scores.max()), 6),
            'mean': round(float(valid_scores.mean()), 6),
        },
        'payload_best_accuracy': payload.get('best_accuracy'),
    }
    V13_CABR_EVALUATION_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(str(V13_CABR_EVALUATION_REPORT_PATH), flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

