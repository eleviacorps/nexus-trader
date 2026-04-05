from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config.project_config import V14_BRANCH_FEATURES_PATH, V14_SSC_EVALUATION_REPORT_PATH
from src.v13.cabr import load_v13_candidate_frames
from src.v14.ssc import evaluate_ssc, train_ssc_model


def main() -> int:
    archive = pd.read_parquet(V14_BRANCH_FEATURES_PATH)
    train_frame, valid_frame, branch_cols, context_cols = load_v13_candidate_frames(
        archive,
        use_temporal_context=False,
    )
    training = train_ssc_model(
        train_frame,
        branch_feature_names=branch_cols,
        context_feature_names=context_cols,
    )
    evaluation = evaluate_ssc(
        training["model"],
        valid_frame,
        branch_feature_names=branch_cols,
        context_feature_names=context_cols,
        device=training["device"],
    )
    report = {
        "branch_feature_count": len(branch_cols),
        "context_feature_count": len(context_cols),
        "history": [round(float(value), 6) for value in training["history"]],
        "device": training["device"],
        **{key: round(float(value), 6) for key, value in evaluation.items()},
    }
    V14_SSC_EVALUATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    V14_SSC_EVALUATION_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(str(V14_SSC_EVALUATION_REPORT_PATH), flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
