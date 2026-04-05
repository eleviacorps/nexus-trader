from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config.project_config import V12_TCTL_EVALUATION_REPORT_PATH
from src.v12.tctl import build_training_pairs, evaluate_pairwise_accuracy, load_tctl_model, prepare_tctl_candidates, score_tctl_model


def main() -> int:
    frame = pd.read_parquet(PROJECT_ROOT / 'outputs' / 'v10' / 'branch_features_v10_full.parquet')
    _, valid_candidates, feature_names = prepare_tctl_candidates(frame, validation_fraction=0.2)
    model, stored_feature_names = load_tctl_model()
    feature_names = tuple(name for name in stored_feature_names if name in feature_names)
    valid_pairs = build_training_pairs(valid_candidates, feature_names=feature_names)
    pairwise_accuracy = evaluate_pairwise_accuracy(model, valid_candidates, valid_pairs, feature_names=feature_names)
    scores = score_tctl_model(model, valid_candidates, feature_names=feature_names)
    report = {
        'candidate_count': int(len(valid_candidates)),
        'pair_count': int(len(valid_pairs)),
        'pairwise_accuracy': round(float(pairwise_accuracy), 6),
        'score_min': round(float(scores.min()), 6),
        'score_max': round(float(scores.max()), 6),
        'score_mean': round(float(scores.mean()), 6),
        'distinct_rounded_scores': int(len(set(scores.round(4).tolist()))),
    }
    V12_TCTL_EVALUATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    V12_TCTL_EVALUATION_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(str(V12_TCTL_EVALUATION_REPORT_PATH), flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
